import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPosEmb(nn.Module):
    """sin embeddings for diffusion timesteps"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """1D res block for temp convs"""

    def __init__(self, in_channels, out_channels, time_emb_dim, kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=kernel_size // 2
        )

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

        num_groups = min(8, out_channels)
        while out_channels % num_groups != 0:
            num_groups -= 1
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x, t):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        time_emb = self.time_mlp(t)[:, :, None]
        h = h + time_emb

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + self.residual_conv(x)


# what unet does instead of pooling
class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class TemporalUNet(nn.Module):
    """
    1d unet, condition on start/goal
    """

    def __init__(
        self,
        transition_dim=2,
        cond_dim=4,
        dim=128,
        dim_mults=(1, 2, 4),
        kernel_size=3,
    ):
        super().__init__()

        self.transition_dim = transition_dim
        self.cond_dim = cond_dim

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, time_dim),
        )

        dims = [dim * m for m in dim_mults]
        self.input_proj = nn.Conv1d(transition_dim, dim, 1)

        self.down_blocks = nn.ModuleList([])
        self.down_samples = nn.ModuleList([])

        in_dim = dim
        for i, out_dim in enumerate(dims):
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        ResidualBlock(in_dim, out_dim, time_dim, kernel_size),
                        ResidualBlock(out_dim, out_dim, time_dim, kernel_size),
                    ]
                )
            )
            if i < len(dims) - 1:
                self.down_samples.append(Downsample(out_dim))
            else:
                self.down_samples.append(nn.Identity())
            in_dim = out_dim

        mid_dim = dims[-1]
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim, time_dim, kernel_size)
        self.mid_block2 = ResidualBlock(mid_dim, mid_dim, time_dim, kernel_size)

        self.up_blocks = nn.ModuleList([])
        self.up_samples = nn.ModuleList([])

        reversed_dims = list(reversed(dims))
        for i, out_dim in enumerate(reversed_dims):
            skip_dim = reversed_dims[i]
            in_ch = (reversed_dims[i - 1] if i > 0 else mid_dim) + skip_dim

            self.up_blocks.append(
                nn.ModuleList(
                    [
                        ResidualBlock(in_ch, out_dim, time_dim, kernel_size),
                        ResidualBlock(out_dim, out_dim, time_dim, kernel_size),
                    ]
                )
            )
            if i < len(dims) - 1:
                self.up_samples.append(Upsample(out_dim))
            else:
                self.up_samples.append(nn.Identity())

        self.final_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2),
            nn.SiLU(),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, time, cond):
        x = x.permute(0, 2, 1)

        t_emb = self.time_mlp(time)
        c_emb = self.cond_mlp(cond)
        emb = t_emb + c_emb

        x = self.input_proj(x)

        skips = []
        for (res1, res2), downsample in zip(self.down_blocks, self.down_samples):
            x = res1(x, emb)
            x = res2(x, emb)
            skips.append(x)
            x = downsample(x)

        x = self.mid_block1(x, emb)
        x = self.mid_block2(x, emb)

        for (res1, res2), upsample in zip(self.up_blocks, self.up_samples):
            skip = skips.pop()
            # Handle size mismatch
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(
                    x, size=skip.shape[-1], mode="linear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
            x = res1(x, emb)
            x = res2(x, emb)
            x = upsample(x)

        x = self.final_conv(x)
        x = x.permute(0, 2, 1)

        return x


class GaussianDiffusion(nn.Module):
    """gaussian diffusion process"""

    def __init__(
        self,
        model,
        horizon,
        observation_dim=2,
        n_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
    ):
        super().__init__()

        self.model = model
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.n_timesteps = n_timesteps

        betas = torch.linspace(beta_start, beta_end, n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # q(x_t | x_0) calcs
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

        # posterior calcs
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance", torch.log(posterior_variance.clamp(min=1e-20))
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def q_sample(self, x_start, t, noise=None):
        """forward diffusiom"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def p_losses(self, x_start, cond, t=None):
        """copute training loss"""
        batch_size = x_start.shape[0]

        if t is None:
            t = torch.randint(0, self.n_timesteps, (batch_size,), device=x_start.device)

        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)

        noise_pred = self.model(x_noisy, t.float(), cond)
        loss = F.mse_loss(noise_pred, noise)

        return loss

    @torch.no_grad()
    def p_sample(self, x, t, cond):
        """single reverse diffusion step"""
        B = x.shape[0]
        t_batch = torch.full((B,), t, device=x.device, dtype=torch.long)

        eps = self.model(x, t_batch.float(), cond)

        sqrt_ab = self.sqrt_alphas_cumprod[t].view(1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alphas_cumprod[t].view(1, 1, 1)
        x0 = (x - sqrt_1mab * eps) / sqrt_ab

        c1 = self.posterior_mean_coef1[t].view(1, 1, 1)
        c2 = self.posterior_mean_coef2[t].view(1, 1, 1)
        mean = c1 * x0 + c2 * x

        if t > 0:
            var = self.posterior_variance[t].view(1, 1, 1)
            x = mean + torch.sqrt(var) * torch.randn_like(x)
        else:
            x = mean

        return x

    @torch.no_grad()
    def sample(self, cond, batch_size=1):
        """generate traj"""
        device = next(self.model.parameters()).device

        x = torch.randn(batch_size, self.horizon, self.observation_dim, device=device)
        for t in reversed(range(self.n_timesteps)):
            x = self.p_sample(x, t, cond)
            x[:, 0, :] = cond[:, :2]
            x[:, -1, :] = cond[:, 2:]

        return x
