"""
Train a 2048 agent with TD(0) afterstate learning.

Examples:
    python train.py --network small --games 20000            # quick run (~minutes)
    python train.py --network large --games 200000           # full run
    python train.py --resume runs/large/weights.npz --games 100000
"""
import argparse
import csv
import os
import time

import numpy as np

from game2048 import NTupleNetwork, run_games, seed, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="large", choices=["large", "small"])
    ap.add_argument("--games", type=int, default=100_000, help="training games")
    ap.add_argument("--alpha", type=float, default=0.1, help="learning rate")
    ap.add_argument("--chunk", type=int, default=1000, help="games per log line")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output folder (default runs/<network>)")
    ap.add_argument("--resume", default=None, help="weights .npz to continue from")
    args = ap.parse_args()

    out = args.out or os.path.join("runs", args.network)
    os.makedirs(out, exist_ok=True)
    seed(args.seed)

    net = NTupleNetwork.load(args.resume) if args.resume else NTupleNetwork(args.network)
    print(f"Network '{net.name}': {len(net.patterns)} tuples of length {net.tuple_len}, "
          f"{net.weights.size:,} weights ({net.weights.nbytes / 1e6:.0f} MB)")

    log_path = os.path.join(out, "train_log.csv")
    start_game = 0
    if args.resume and os.path.exists(log_path):
        with open(log_path) as f:
            rows = list(csv.DictReader(f))
            if rows:
                start_game = int(rows[-1]["games"])
    new_log = not (args.resume and os.path.exists(log_path))
    f = open(log_path, "w" if new_log else "a", newline="")
    writer = csv.writer(f)
    if new_log:
        writer.writerow(["games", "avg_score", "max_score", "win_rate_2048",
                         "reach_1024", "reach_4096", "avg_moves", "elapsed_s"])

    t0 = time.time()
    done = 0
    print("Compiling (first chunk is slower)...")
    while done < args.games:
        n = min(args.chunk, args.games - done)
        scores, tiles, moves = run_games(n, net.sym, net.weights, args.alpha, True)
        done += n
        s = summarize(scores, tiles, moves)
        total = start_game + done
        elapsed = time.time() - t0
        writer.writerow([total, round(s["avg_score"], 1), s["max_score"],
                         round(s["win_rate_2048"], 4),
                         round(float(np.mean(tiles >= 10)), 4),
                         round(float(np.mean(tiles >= 12)), 4),
                         round(s["avg_moves"], 1), round(elapsed, 1)])
        f.flush()
        print(f"games {total:>8,} | avg score {s['avg_score']:>9.0f} | "
              f"max {s['max_score']:>7} | 2048 rate {100 * s['win_rate_2048']:5.1f}% | "
              f"{elapsed / 60:6.1f} min")
        if done % (args.chunk * 10) == 0 or done == args.games:
            net.save(os.path.join(out, "weights.npz"))
    f.close()
    print(f"Saved weights to {os.path.join(out, 'weights.npz')}")


if __name__ == "__main__":
    main()
