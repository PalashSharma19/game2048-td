"""
2048 game engine.

A board is a 1-D NumPy array of 16 uint8 values in row-major order
(cell index = 4 * row + col). Each value is the tile's exponent:
0 = empty, 1 = 2, 2 = 4, 3 = 8, ..., 11 = 2048.

All hot-path functions are compiled with Numba so that training on
hundreds of thousands of games is feasible in pure Python.
"""
import numpy as np
from numba import njit

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTION_NAMES = ("up", "right", "down", "left")

# For each action, the four lines of the board, each listed starting from the
# edge that tiles slide *towards*. This lets one routine handle all directions.
LINES = np.array([
    [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]],    # up
    [[3, 2, 1, 0], [7, 6, 5, 4], [11, 10, 9, 8], [15, 14, 13, 12]],    # right
    [[12, 8, 4, 0], [13, 9, 5, 1], [14, 10, 6, 2], [15, 11, 7, 3]],    # down
    [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],    # left
], dtype=np.int64)


@njit(cache=True)
def move(board, action, out):
    """Slide `board` in direction `action`, writing the afterstate into `out`.

    Returns the reward (sum of newly merged tile values), or -1 if the move
    does not change the board (an illegal move). Each tile merges at most
    once per move, as in the original game.
    """
    reward = 0
    changed = False
    for l in range(4):
        idx = LINES[action, l]
        k = 0          # next write position along the line
        pending = 0    # tile waiting to see if the next tile matches it
        for j in range(4):
            v = np.int64(board[idx[j]])
            if v == 0:
                continue
            if pending == 0:
                pending = v
            elif pending == v:
                out[idx[k]] = v + 1
                reward += 1 << (v + 1)
                k += 1
                pending = 0
            else:
                out[idx[k]] = pending
                k += 1
                pending = v
        if pending != 0:
            out[idx[k]] = pending
            k += 1
        while k < 4:
            out[idx[k]] = 0
            k += 1
        for j in range(4):
            if out[idx[j]] != board[idx[j]]:
                changed = True
    return reward if changed else -1


@njit(cache=True)
def spawn_tile(board):
    """Place a 2 (p = 0.9) or a 4 (p = 0.1) on a uniformly random empty cell."""
    n_empty = 0
    for i in range(16):
        if board[i] == 0:
            n_empty += 1
    if n_empty == 0:
        return False
    k = np.random.randint(n_empty)
    for i in range(16):
        if board[i] == 0:
            if k == 0:
                board[i] = 1 if np.random.random() < 0.9 else 2
                return True
            k -= 1
    return False


@njit(cache=True)
def new_board():
    board = np.zeros(16, np.uint8)
    spawn_tile(board)
    spawn_tile(board)
    return board


@njit(cache=True)
def seed(s):
    """Seed Numba's random generator (separate from NumPy's)."""
    np.random.seed(s)


# ---------- helpers for humans (not performance critical) ----------

def to_grid(board):
    """Convert exponent board to a 4x4 grid of actual tile values."""
    b = np.asarray(board, dtype=np.int64)
    return np.where(b > 0, 2 ** b, 0).reshape(4, 4)


def from_grid(grid):
    """Convert a 4x4 grid of tile values (0, 2, 4, ...) to an exponent board."""
    g = np.asarray(grid, dtype=np.int64).reshape(16)
    out = np.zeros(16, np.uint8)
    nz = g > 0
    out[nz] = np.log2(g[nz]).astype(np.uint8)
    return out


def render(board):
    grid = to_grid(board)
    lines = []
    for row in grid:
        lines.append(" ".join(f"{v:>5}" if v else "    ." for v in row))
    return "\n".join(lines)
