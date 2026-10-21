import numpy as np
import pytest

from game2048 import move, from_grid, to_grid
from game2048.board import UP, RIGHT, DOWN, LEFT
from game2048.ntuple import symmetric_patterns, NETWORKS, NTupleNetwork


def do(grid, action):
    out = np.zeros(16, np.uint8)
    r = move(from_grid(grid), action, out)
    return to_grid(out).tolist(), r


def row(vals):
    return [vals, [0] * 4, [0] * 4, [0] * 4]


@pytest.mark.parametrize("before, after, reward", [
    ([2, 2, 0, 0], [4, 0, 0, 0], 4),
    ([2, 2, 2, 2], [4, 4, 0, 0], 8),
    ([2, 2, 4, 0], [4, 4, 0, 0], 4),       # merged tile must not merge again
    ([4, 2, 2, 0], [4, 4, 0, 0], 4),
    ([0, 0, 0, 2], [2, 0, 0, 0], 0),
    ([2, 4, 8, 16], [2, 4, 8, 16], -1),    # nothing moves -> illegal
    ([8, 0, 8, 8], [16, 8, 0, 0], 16),
])
def test_left(before, after, reward):
    g, r = do(row(before), LEFT)
    assert g[0] == after and r == reward


def test_right():
    g, r = do(row([2, 2, 2, 0]), RIGHT)
    assert g[0] == [0, 0, 2, 4] and r == 4


def test_up_down():
    grid = [[2, 0, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0]]
    g, r = do(grid, UP)
    assert [g[i][0] for i in range(4)] == [4, 4, 0, 0] and r == 4
    g, r = do(grid, DOWN)
    assert [g[i][0] for i in range(4)] == [0, 0, 4, 4] and r == 4


def test_symmetries_are_distinct_and_valid():
    for patterns in NETWORKS.values():
        sym = symmetric_patterns(patterns)
        assert sym.shape[1] == 8
        for p in sym:
            assert len({tuple(v) for v in p}) == 8
            assert p.min() >= 0 and p.max() <= 15


def test_value_is_symmetry_invariant():
    net = NTupleNetwork("small")
    rng = np.random.default_rng(0)
    net.weights[:] = rng.normal(size=net.weights.shape).astype(np.float32)
    b = rng.integers(0, 8, 16).astype(np.uint8)
    g = b.reshape(4, 4)
    v = net.value(b)
    for k in range(4):
        for flip in (False, True):
            t = np.rot90(g, k)
            t = np.fliplr(t) if flip else t
            assert np.isclose(net.value(np.ascontiguousarray(t).reshape(16)), v, rtol=1e-4)
