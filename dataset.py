import numpy as np
import random
from tqdm import tqdm
from maze_env import MazeEnv
from utils import solve_maze


def generate_dataset(num_samples=10000, max_len=128):
    """
    generate maze traj dataset
    """
    env = MazeEnv(size=17)
    env.reset()
    np.save("maze_layout.npy", env.grid)

    trajectories = []

    empty_cells = []
    rows, cols = env.grid.shape
    for r in range(rows):
        for c in range(cols):
            if env.grid[r, c] == 0:
                empty_cells.append((r, c))

    generated = 0
    pbar = tqdm(total=num_samples, desc="generating trajectories")

    while generated < num_samples:
        start = random.choice(empty_cells)
        goal = random.choice(empty_cells)

        if start == goal:
            continue

        path = solve_maze(env, start, goal)
        if path is None:
            continue

        path = np.array(path)
        path_len = len(path)
        old_indices = np.linspace(0, 1, path_len)
        new_indices = np.linspace(0, 1, max_len)

        resampled_path = np.zeros((max_len, 2))
        resampled_path[:, 0] = np.interp(new_indices, old_indices, path[:, 0])
        resampled_path[:, 1] = np.interp(new_indices, old_indices, path[:, 1])

        trajectories.append(resampled_path)
        generated += 1
        pbar.update(1)

    pbar.close()

    trajectories = np.array(trajectories)
    np.save("trajectories.npy", trajectories)


if __name__ == "__main__":
    generate_dataset()
