import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib as mpl

from model import TemporalUNet, GaussianDiffusion
from config_utils import load_config
from train import DifferentiableCollisionLoss

EVAL_SEED = 0

# latex-ish plotting
mpl.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
)


def load_model(config, device):
    """Load trained model from checkpoint."""
    model = TemporalUNet(
        transition_dim=config["model"]["transition_dim"],
        cond_dim=4,
        dim=config["model"]["dim"],
        dim_mults=tuple(config["model"]["dim_mults"]),
        kernel_size=config["model"]["kernel_size"],
    ).to(device)

    diffusion = GaussianDiffusion(
        model=model,
        horizon=config["diffusion"]["horizon"],
        observation_dim=config["diffusion"]["observation_dim"],
        n_timesteps=config["diffusion"]["n_timesteps"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
    ).to(device)

    checkpoint = torch.load(
        config["eval"]["model_path"], map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, diffusion


def denormalize(traj, maze_size=17):
    """convert normalized traj back to grid coords"""
    return ((traj + 1) / 2) * (maze_size - 1)


def densify_trajectory(traj, num_interp=4):
    B, T, _ = traj.shape
    weights = torch.linspace(0, 1, num_interp + 1, device=traj.device)[:-1]

    p1 = traj[:, :-1, :]
    p2 = traj[:, 1:, :]

    w = weights.view(1, 1, -1, 1)
    interp = p1.unsqueeze(2) * (1 - w) + p2.unsqueeze(2) * w  # (B, T-1, W, 2)

    dense = interp.reshape(B, (T - 1) * num_interp, 2)
    dense = torch.cat([dense, traj[:, -1:, :]], dim=1)
    return dense


def collision_scores(traj, wall_field, num_interp=4):
    """
    compute per sample coll score
    """
    dense = densify_trajectory(traj, num_interp=num_interp)
    B, Td, _ = dense.shape

    grid = dense[:, :, [1, 0]].view(B, Td, 1, 2)

    wall_values = F.grid_sample(
        wall_field.expand(B, -1, -1, -1),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    wall_values = wall_values.squeeze(1).squeeze(-1)
    return wall_values.mean(dim=1)


@torch.no_grad()
def sample_snapshots(diffusion, cond_k, snapshot_ts, generator=None):
    """
    run reverse diffusion and store x_t for selected timesteps
    returns dict {t: x_t_cpu}, x_t has shape (K, T, 2)
    """
    device = next(diffusion.model.parameters()).device
    K = cond_k.shape[0]

    snapshot_ts = [int(t) for t in snapshot_ts]
    want = set(snapshot_ts)

    x = torch.randn(
        K,
        diffusion.horizon,
        diffusion.observation_dim,
        device=device,
        generator=generator,
    )
    snaps = {}

    for t in reversed(range(diffusion.n_timesteps)):
        t_batch = torch.full((K,), t, device=device, dtype=torch.long)
        eps = diffusion.model(x, t_batch.float(), cond_k)

        sqrt_ab = diffusion.sqrt_alphas_cumprod[t].view(1, 1, 1)
        sqrt_1mab = diffusion.sqrt_one_minus_alphas_cumprod[t].view(1, 1, 1)
        x0 = (x - sqrt_1mab * eps) / sqrt_ab

        c1 = diffusion.posterior_mean_coef1[t].view(1, 1, 1)
        c2 = diffusion.posterior_mean_coef2[t].view(1, 1, 1)
        mean = c1 * x0 + c2 * x

        if t > 0:
            var = diffusion.posterior_variance[t].view(1, 1, 1)
            noise = torch.randn(x.shape, device=x.device, generator=generator)
            x = mean + torch.sqrt(var) * noise

        else:
            x = mean

        # enforce endpoints each step (same as your sample())
        x[:, 0, :] = cond_k[:, :2]
        x[:, -1, :] = cond_k[:, 2:]

        if t in want:
            snaps[t] = x.detach().cpu().clone()

    return snaps


def plot_collapse_tube(
    maze,
    maze_size,
    start,
    goal,
    snaps,
    snapshot_ts,
    out_png="collapse_tube.png",
    out_pdf="collapse_tube.pdf",
):
    """
    plots K*T waypoints as a point cloud for several diffusion timesteps.
    points are colored by waypoint index along the trajectory (0..T-1).
    """
    snapshot_ts = list(snapshot_ts)
    ncols = len(snapshot_ts)
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 5))

    if ncols == 1:
        axes = [axes]

    for ax, t in zip(axes, snapshot_ts):
        x_t = snaps[t].numpy()  # (K, T, 2) normalized
        K, T, _ = x_t.shape

        xg = denormalize(x_t, maze_size=maze_size)  # (K, T, 2) grid coords
        rows = xg[:, :, 0].reshape(-1)
        cols = xg[:, :, 1].reshape(-1)

        # color by waypoint index (trajectory time), repeated K times
        c = np.tile(np.arange(T), K)

        ax.imshow(maze, cmap="binary", origin="upper")
        ax.scatter(
            cols,
            rows,
            c=c,
            cmap="inferno",
            s=140,
            alpha=0.95,
            edgecolors="none",
            vmin=0,
            vmax=T - 1,
        )

        # start/goal markers
        ax.plot(start[1], start[0], "go", markersize=12)
        ax.plot(goal[1], goal[0], "ro", markersize=12)

        ax.set_title(rf"$x_t$ at $t={t}$")
        ax.set_xlim(-0.5, maze_size - 0.5)
        ax.set_ylim(maze_size - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate():
    config = load_config()

    device = torch.device(
        config["train"]["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"using dev: {device}")

    model, diffusion = load_model(config, device)
    print("model loaded")

    maze = np.load("maze_layout.npy")
    maze_size = config["maze"]["size"]

    collision_loss = DifferentiableCollisionLoss("maze_layout.npy").to(device)
    wall_field = collision_loss.wall_field  # (1, 1, H, W)

    empty_cells = [
        (r, c) for r in range(maze_size) for c in range(maze_size) if maze[r, c] == 0
    ]

    np.random.seed(EVAL_SEED)
    torch.manual_seed(EVAL_SEED)

    num_samples = 5
    best_of_k = 8
    num_interp = 4

    start_goal_pairs = []
    for _ in range(num_samples):
        idx_start = np.random.randint(len(empty_cells))
        idx_goal = np.random.randint(len(empty_cells))
        while idx_goal == idx_start:
            idx_goal = np.random.randint(len(empty_cells))
        start_goal_pairs.append(
            (np.array(empty_cells[idx_start]), np.array(empty_cells[idx_goal]))
        )

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    Tn = diffusion.n_timesteps
    snapshot_ts = [200, 100, 50, 20, 5, 0]
    snapshot_ts = [t for t in snapshot_ts if t <= (Tn - 1)]

    saved_collapse = False

    for i, (start, goal) in enumerate(
        tqdm(start_goal_pairs, desc="generating samples")
    ):
        ax = axes[i]

        start_norm = 2 * (start / (maze_size - 1)) - 1
        goal_norm = 2 * (goal / (maze_size - 1)) - 1
        cond = torch.FloatTensor(np.concatenate([start_norm, goal_norm])).to(device)

        cond_k = cond.unsqueeze(0).repeat(best_of_k, 1)

        with torch.no_grad():
            samples = diffusion.sample(cond_k, batch_size=best_of_k)
            scores = collision_scores(samples, wall_field, num_interp=num_interp)
            best_idx = int(torch.argmin(scores).item())

        ax.imshow(maze, cmap="binary", origin="upper")

        for k in range(best_of_k):
            traj_k = denormalize(samples[k].cpu().numpy(), maze_size)
            ax.plot(traj_k[:, 1], traj_k[:, 0], linewidth=1.6, alpha=0.5)

        traj_best = denormalize(samples[best_idx].cpu().numpy(), maze_size)
        ax.plot(traj_best[:, 1], traj_best[:, 0], linewidth=3.2, alpha=1.0)

        ax.plot(start[1], start[0], "go", markersize=12)
        ax.plot(goal[1], goal[0], "ro", markersize=12)

        ax.set_title(rf"Sample {i + 1} ($K={best_of_k}$)")
        ax.set_xlim(-0.5, maze_size - 0.5)
        ax.set_ylim(maze_size - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])

        # additionally save one collapse-into-tube figure (for the first pair)
        if (not saved_collapse) and (i == 0):
            cond_best = cond.unsqueeze(0)
            g = torch.Generator(device=device)
            g.manual_seed(EVAL_SEED + 1000 * i + best_idx)

            snaps = sample_snapshots(diffusion, cond_best, snapshot_ts, generator=g)
            plot_collapse_tube(
                maze=maze,
                maze_size=maze_size,
                start=start,
                goal=goal,
                snaps=snaps,
                snapshot_ts=snapshot_ts,
                out_png="collapse_tube.png",
                out_pdf="collapse_tube.pdf",
            )
            saved_collapse = True

    plt.tight_layout()
    plt.savefig("evaluation_results.png", dpi=300, bbox_inches="tight")
    plt.savefig("evaluation_results.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    print("saved evaluation_results.png")
    if saved_collapse:
        print("saved collapse_tube.png and collapse_tube.pdf")


if __name__ == "__main__":
    evaluate()
