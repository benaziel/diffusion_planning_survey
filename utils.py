import numpy as np
from collections import deque


def solve_maze(env, start, goal):
    """ bfs solver """
    rows, cols = env.grid.shape
    queue = deque([(start, [start])])
    visited = set()
    visited.add(start)

    while queue:
        (r, c), path = queue.popleft()

        if (r, c) == goal:
            return path

        # action space: up, down, left, right
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols:
                if env.grid[nr, nc] == 0 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [(nr, nc)]))

    return None
