"""
Plot learning curves and the tile distribution.

Example:
    python plot.py --log runs/large/train_log.csv --results results.json
"""
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--results", default=None)
    ap.add_argument("--out", default="learning_curve.png")
    args = ap.parse_args()

    with open(args.log) as f:
        rows = list(csv.DictReader(f))
    games = [int(r["games"]) for r in rows]
    score = [float(r["avg_score"]) for r in rows]
    win = [100 * float(r["win_rate_2048"]) for r in rows]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(games, score)
    ax[0].set(xlabel="Training games", ylabel="Average score", title="Average score per chunk")
    ax[1].plot(games, win, color="tab:green")
    ax[1].set(xlabel="Training games", ylabel="% games reaching 2048",
              title="2048 success rate", ylim=(0, 100))
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")

    if args.results:
        with open(args.results) as f:
            res = json.load(f)
        fig, a = plt.subplots(figsize=(8, 4.5))
        names = list(res.keys())
        tiles = sorted({int(t) for r in res.values() for t in r["reach_rate"]})
        width = 0.8 / len(names)
        for i, n in enumerate(names):
            vals = [100 * res[n]["reach_rate"].get(str(t), 0.0) for t in tiles]
            a.bar([j + i * width for j in range(len(tiles))], vals, width, label=n)
        a.set_xticks([j + width * (len(names) - 1) / 2 for j in range(len(tiles))])
        a.set_xticklabels([str(t) for t in tiles])
        a.set(xlabel="Tile", ylabel="% of games reaching tile", title="Tile reach rates")
        a.legend()
        a.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig("tile_distribution.png", dpi=150)
        print("Saved tile_distribution.png")


if __name__ == "__main__":
    main()
