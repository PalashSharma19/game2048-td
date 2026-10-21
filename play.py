"""
Watch the trained agent play one game in the terminal.

Example:
    python play.py --weights runs/large/weights.npz --delay 0.05
"""
import argparse
import time

import numpy as np

from game2048 import (NTupleNetwork, greedy_action, new_board, spawn_tile,
                      render, seed, ACTION_NAMES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--delay", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    net = NTupleNetwork.load(args.weights)
    if args.seed is not None:
        seed(args.seed)
    board = new_board()
    buf = np.zeros((4, 16), np.uint8)
    score, moves = 0, 0
    while True:
        a, r = greedy_action(board, net.sym, net.weights, buf)
        if a < 0:
            break
        board[:] = buf[a]
        score += r
        moves += 1
        spawn_tile(board)
        print("\033[H\033[J", end="")
        print(f"Move {moves}  |  {ACTION_NAMES[a]:<5}  |  score {score}\n")
        print(render(board))
        time.sleep(args.delay)
    print(f"\nGame over. Score {score}, max tile {2 ** int(board.max())}, {moves} moves.")


if __name__ == "__main__":
    main()
