import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import TemporalUNet, GaussianDiffusion
from config_utils import load_config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories_path, maze_size=17):
        self.trajectories = np.load(trajectories_path)
        self.maze_size = maze_size

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj = 2 * (traj / (self.maze_size - 1)) - 1
        cond = np.concatenate([traj[0], traj[-1]])
        return {
            "trajectories": torch.from_numpy(traj).float(),
            "conditions": torch.from_numpy(cond).float(),
        }


class DifferentiableCollisionLoss(nn.Module):
    def __init__(self, maze_path):
        super().__init__()
        maze = np.load(maze_path)
        self.register_buffer(
            "wall_field",
            torch.from_numpy(maze).float().unsqueeze(0).unsqueeze(0),
        )

    def forward(self, traj, num_interp=4):
        B, T, _ = traj.shape
        w = torch.linspace(0, 1, num_interp + 1, device=traj.device)[:-1]
        p1, p2 = traj[:, :-1], traj[:, 1:]

        dense = torch.stack(
            [p1 * (1 - wi) + p2 * wi for wi in w],
            dim=2,
        ).reshape(B, -1, 2)

        dense = torch.cat([dense, traj[:, -1:]], dim=1)
        grid = dense[:, :, [1, 0]].view(B, -1, 1, 2)

        wall_vals = F.grid_sample(
            self.wall_field.expand(B, -1, -1, -1),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        return wall_vals.mean()


class DiffusionWithCollision(nn.Module):
    def __init__(self, diffusion, collision_fn, weight):
        super().__init__()
        self.diffusion = diffusion
        self.collision_fn = collision_fn
        self.weight = weight

    def loss(self, x0, cond):
        B = x0.shape[0]
        t = torch.randint(0, self.diffusion.n_timesteps, (B,), device=x0.device)

        noise = torch.randn_like(x0)
        xt = self.diffusion.q_sample(x0, t, noise)

        pred = self.diffusion.model(xt, t.float(), cond)
        diff_loss = F.mse_loss(pred, noise)

        sqrt_ab = self.diffusion.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_1mab = self.diffusion.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        x0_hat = (xt - sqrt_1mab * pred) / sqrt_ab

        coll_loss = self.collision_fn(x0_hat)
        return diff_loss + self.weight * coll_loss, diff_loss, coll_loss


def train():
    cfg = load_config()
    device = torch.device(
        cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    )

    set_seed(cfg["train"].get("seed", 0))

    dataset = TrajectoryDataset("trajectories.npy", maze_size=cfg["maze"]["size"])

    val_frac = cfg["train"].get("val_frac", 0.1)
    n_val = int(len(dataset) * val_frac)
    n_train = len(dataset) - n_val

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg["train"].get("seed", 0)),
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False
    )

    model = TemporalUNet(
        transition_dim=cfg["model"]["transition_dim"],
        cond_dim=4,
        dim=cfg["model"]["dim"],
        dim_mults=tuple(cfg["model"]["dim_mults"]),
        kernel_size=cfg["model"]["kernel_size"],
    ).to(device)

    diffusion = GaussianDiffusion(
        model=model,
        horizon=cfg["diffusion"]["horizon"],
        observation_dim=cfg["diffusion"]["observation_dim"],
        n_timesteps=cfg["diffusion"]["n_timesteps"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
    ).to(device)

    coll_fn = DifferentiableCollisionLoss("maze_layout.npy").to(device)
    wrapper = DiffusionWithCollision(
        diffusion, coll_fn, cfg["train"]["collision_weight"]
    )

    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    train_diff, train_coll = [], []
    val_diff, val_coll = [], []

    for _ in tqdm(range(cfg["train"]["epochs"]), desc="training"):
        model.train()
        td, tc = 0.0, 0.0

        for batch in train_loader:
            x = batch["trajectories"].to(device)
            c = batch["conditions"].to(device)

            opt.zero_grad()
            loss, dl, cl = wrapper.loss(x, c)
            loss.backward()
            opt.step()

            td += dl.item()
            tc += cl.item()

        train_diff.append(td / len(train_loader))
        train_coll.append(tc / len(train_loader))

        model.eval()
        vd, vc = 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["trajectories"].to(device)
                c = batch["conditions"].to(device)
                _, dl, cl = wrapper.loss(x, c)
                vd += dl.item()
                vc += cl.item()

        val_diff.append(vd / len(val_loader))
        val_coll.append(vc / len(val_loader))

    torch.save(model.state_dict(), cfg["eval"]["model_path"])

    # plot losses
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(train_diff, label="train")
    ax[0].plot(val_diff, label="val")
    ax[0].set_title("Diffusion loss")
    ax[0].legend()

    ax[1].plot(train_coll, label="train")
    ax[1].plot(val_coll, label="val")
    ax[1].set_title("Collision loss")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig("training_loss.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    train()
