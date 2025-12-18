import numpy as np


class MazeEnv:
    def __init__(self, size=17):  # note that it should be odd otherwise it has a lip
        self.size = size
        self.grid = None

    def reset(self):
        self.grid = self._generate_maze()
        return self.grid

    def _generate_maze(self):
        grid = np.ones((self.size, self.size), dtype=np.int32)
        start_r, start_c = 1, 1
        grid[start_r, start_c] = 0

        stack = [(start_r, start_c)]
        directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]

        while stack:
            r, c = stack[-1]
            neighbors = []
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 1 <= nr < self.size - 1 and 1 <= nc < self.size - 1:
                    if grid[nr, nc] == 1:  # unvisited
                        neighbors.append((nr, nc, dr, dc))

            if neighbors:
                nr, nc, dr, dc = neighbors[np.random.randint(len(neighbors))]
                grid[r + dr // 2, c + dc // 2] = 0
                grid[nr, nc] = 0
                stack.append((nr, nc))
            else:
                stack.pop()

        return grid

    def get_empty_cells(self):
        empty = []
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r, c] == 0:
                    empty.append((r, c))
        return empty

    def is_valid(self, r, c):
        return 0 <= r < self.size and 0 <= c < self.size and self.grid[r, c] == 0
