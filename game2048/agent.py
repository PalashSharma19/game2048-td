"""
TD(0) afterstate learning for 2048.

Notation for one time step t:
    s_t   : board before the move (a new tile has already spawned)
    a_t   : chosen action
    r_t   : reward from merges in that move
    s'_t  : afterstate, the board right after sliding, before a tile spawns

Policy (greedy, no epsilon is needed because tile spawns already give
plenty of exploration):
    a_t = argmax_a [ r(s_t, a) + V(s'(s_t, a)) ]

TD(0) update applied after every move, once the next afterstate is known:
    V(s'_{t-1}) <- V(s'_{t-1}) + alpha * ( r_t + V(s'_t) - V(s'_{t-1}) )
When the game ends (no legal move), the target is 0.
"""
import numpy as np
from numba import njit

from .board import move, spawn_tile, new_board
from .ntuple import evaluate, update


@njit(cache=True)
def greedy_action(board, sym, weights, buf):
    """Return (action, reward) maximizing r + V(afterstate); (-1, 0) if none.
    The chosen afterstate is left in buf[action]."""
    best_a = -1
    best_r = 0
    best_v = -1e30
    for a in range(4):
        r = move(board, a, buf[a])
        if r < 0:
            continue
        v = r + evaluate(buf[a], sym, weights)
        if v > best_v:
            best_v = v
            best_a = a
            best_r = r
    return best_a, best_r


@njit(cache=True)
def play_game(sym, weights, alpha, learn):
    """Play one full game with the greedy policy.

    If `learn` is True, apply the online TD(0) afterstate update after every
    move. Returns (score, max_tile_exponent, n_moves).
    """
    board = new_board()
    buf = np.zeros((4, 16), np.uint8)
    prev_after = np.zeros(16, np.uint8)
    has_prev = False
    n_feat = sym.shape[1]          # step size is split across the 8 symmetries
    score = 0
    n_moves = 0

    while True:
        a, r = greedy_action(board, sym, weights, buf)
        if a < 0:
            # Terminal: the previous afterstate leads to game over, target 0.
            if learn and has_prev:
                err = 0.0 - evaluate(prev_after, sym, weights)
                update(prev_after, sym, weights, alpha * err / n_feat)
            break

        after = buf[a]
        if learn and has_prev:
            target = r + evaluate(after, sym, weights)
            err = target - evaluate(prev_after, sym, weights)
            update(prev_after, sym, weights, alpha * err / n_feat)

        prev_after[:] = after
        has_prev = True
        score += r
        n_moves += 1
        board[:] = after
        spawn_tile(board)

    return score, board.max(), n_moves


@njit(cache=True)
def run_games(n, sym, weights, alpha, learn):
    """Play `n` games; return arrays of scores, max tile exponents, move counts."""
    scores = np.zeros(n, np.int64)
    tiles = np.zeros(n, np.int64)
    moves = np.zeros(n, np.int64)
    for i in range(n):
        s, t, m = play_game(sym, weights, alpha, learn)
        scores[i] = s
        tiles[i] = t
        moves[i] = m
    return scores, tiles, moves


@njit(cache=True)
def run_random_games(n):
    """Baseline: choose uniformly among legal moves."""
    scores = np.zeros(n, np.int64)
    tiles = np.zeros(n, np.int64)
    moves = np.zeros(n, np.int64)
    buf = np.zeros((4, 16), np.uint8)
    legal = np.zeros(4, np.int64)
    rewards = np.zeros(4, np.int64)
    for i in range(n):
        board = new_board()
        score = 0
        m = 0
        while True:
            n_legal = 0
            for a in range(4):
                r = move(board, a, buf[a])
                if r >= 0:
                    legal[n_legal] = a
                    rewards[n_legal] = r
                    n_legal += 1
            if n_legal == 0:
                break
            k = np.random.randint(n_legal)
            board[:] = buf[legal[k]]
            score += rewards[k]
            m += 1
            spawn_tile(board)
        scores[i] = score
        tiles[i] = board.max()
        moves[i] = m
    return scores, tiles, moves


def summarize(scores, tiles, moves):
    """Metrics used throughout the project."""
    n = len(scores)
    reach = {}
    for e in range(7, 17):                      # 128 ... 65536
        frac = float(np.mean(tiles >= e))
        if frac > 0 or e <= 11:
            reach[str(2 ** e)] = frac
    return {
        "games": int(n),
        "avg_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "max_score": int(np.max(scores)),
        "avg_moves": float(np.mean(moves)),
        "max_tile": int(2 ** np.max(tiles)),
        "win_rate_2048": float(np.mean(tiles >= 11)),
        "reach_rate": reach,
    }
