"""
Evaluate a trained agent (learning switched off) and the random baseline.

Example:
    python evaluate.py --weights runs/large/weights.npz --games 1000
"""
import argparse
import json
import os

from game2048 import NTupleNetwork, run_games, run_random_games, seed, summarize


def report(name, s):
    print(f"\n=== {name} ({s['games']} games) ===")
    print(f"Average score : {s['avg_score']:.0f} (std {s['std_score']:.0f})")
    print(f"Best score    : {s['max_score']}")
    print(f"Max tile      : {s['max_tile']}")
    print(f"Average moves : {s['avg_moves']:.0f}")
    print("Reached tile  :")
    for tile, frac in s["reach_rate"].items():
        print(f"   >= {tile:>6}: {100 * frac:6.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="trained weights .npz")
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    results = {}
    seed(args.seed)
    results["random"] = summarize(*run_random_games(args.games))
    report("Random baseline", results["random"])

    if args.weights:
        net = NTupleNetwork.load(args.weights)
        seed(args.seed)
        results["td_agent"] = summarize(*run_games(args.games, net.sym, net.weights, 0.0, False))
        report(f"TD agent [{net.name}]", results["td_agent"])

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
