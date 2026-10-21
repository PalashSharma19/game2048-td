"""
N-tuple network value function.

Each n-tuple is a fixed group of cells. The exponents in those cells form a
base-16 index into a lookup table of weights. The board's value is the sum of
the looked-up weights over every tuple and every one of the board's 8
symmetries (4 rotations x optional mirror), so each table is shared by
symmetric positions of its tuple. This weight sharing makes learning much
faster, because one experience updates 8 positions at once.
"""
import json
import numpy as np
from numba import njit

# Cell indices are row-major on the 4x4 board:
#    0  1  2  3
#    4  5  6  7
#    8  9 10 11
#   12 13 14 15
NETWORKS = {
    # Four 6-tuples (two "line + 2" shapes, two 2x3 rectangles). The standard
    # strong configuration from the 2048 TD literature. Memory: ~268 MB.
    "large": [
        [0, 1, 2, 3, 4, 5],
        [4, 5, 6, 7, 8, 9],
        [0, 1, 2, 4, 5, 6],
        [4, 5, 6, 8, 9, 10],
    ],
    # Five 4-tuples (edge row, inner row, three 2x2 squares). Learns quickly
    # and uses ~1.3 MB, but plateaus lower. Good for fast experiments.
    "small": [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 4, 5],
        [1, 2, 5, 6],
        [5, 6, 9, 10],
    ],
}


def symmetric_patterns(patterns):
    """Return an int array (n_patterns, 8, tuple_len) of the 8 symmetric
    variants of every pattern."""
    out = []
    for p in patterns:
        variants = []
        for flip in (False, True):
            for rot in range(4):
                cells = []
                for i in p:
                    r, c = divmod(i, 4)
                    if flip:
                        c = 3 - c
                    for _ in range(rot):
                        r, c = c, 3 - r      # rotate 90 degrees clockwise
                    cells.append(4 * r + c)
                variants.append(cells)
        out.append(variants)
    return np.array(out, dtype=np.int64)


@njit(cache=True)
def evaluate(board, sym, weights):
    """V(board): sum of table entries over all tuples and symmetries."""
    v = 0.0
    P, S, L = sym.shape
    for p in range(P):
        for s in range(S):
            idx = 0
            for k in range(L):
                c = board[sym[p, s, k]]
                if c > 15:
                    c = 15
                idx = idx * 16 + c
            v += weights[p, idx]
    return v


@njit(cache=True)
def update(board, sym, weights, delta):
    """Add `delta` to every table entry that contributes to V(board)."""
    P, S, L = sym.shape
    for p in range(P):
        for s in range(S):
            idx = 0
            for k in range(L):
                c = board[sym[p, s, k]]
                if c > 15:
                    c = 15
                idx = idx * 16 + c
            weights[p, idx] += delta


class NTupleNetwork:
    def __init__(self, name="large", patterns=None):
        self.name = name
        self.patterns = patterns if patterns is not None else NETWORKS[name]
        lengths = {len(p) for p in self.patterns}
        if len(lengths) != 1:
            raise ValueError("All tuples in one network must have the same length")
        self.tuple_len = lengths.pop()
        self.sym = symmetric_patterns(self.patterns)
        self.weights = np.zeros((len(self.patterns), 16 ** self.tuple_len), np.float32)

    @property
    def n_symmetries(self):
        return self.sym.shape[1]

    def value(self, board):
        return evaluate(board, self.sym, self.weights)

    def save(self, path):
        np.savez_compressed(path, weights=self.weights,
                            meta=json.dumps({"name": self.name, "patterns": self.patterns}))

    @classmethod
    def load(cls, path):
        data = np.load(path)
        meta = json.loads(str(data["meta"]))
        net = cls(meta["name"], meta["patterns"])
        net.weights = data["weights"].astype(np.float32)
        return net
