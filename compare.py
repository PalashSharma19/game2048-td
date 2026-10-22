import argparse
import csv
import json
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from game2048 import NTupleNetwork, run_games, run_random_games, seed


AGENT_COLORS = {
    "Zero-weight greedy": "#4C78A8",
    "Trained TD agent": "#F58518",
    "Random-move baseline": "#54A24B",
}
TILE_THRESHOLDS = (512, 1024, 2048, 4096, 8192, 16384)


def mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    margin = 1.96 * float(np.std(values, ddof=1)) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return [mean - margin, mean + margin]


def wilson_interval(successes, n):
    if n == 0:
        return [0.0, 0.0]
    z = 1.96
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def summarize_agent(data):
    scores = np.asarray(data["score"], dtype=np.float64)
    exponents = np.asarray(data["max_tile"], dtype=np.int64)
    moves = np.asarray(data["moves"], dtype=np.float64)
    n = len(scores)
    max_values = np.exp2(exponents)
    reached_2048 = exponents >= 11
    return {
        "games": int(n),
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "std_score": float(np.std(scores, ddof=1)) if n > 1 else 0.0,
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "mean_moves": float(np.mean(moves)),
        "mean_max_tile_exponent": float(np.mean(exponents)),
        "highest_tile": int(np.max(max_values)),
        "reach_rates_percent": {str(tile): 100.0 * float(np.mean(exponents >= int(math.log2(tile)))) for tile in TILE_THRESHOLDS},
        "reach_2048_ci95_percent": [100.0 * x for x in wilson_interval(int(np.sum(reached_2048)), n)],
        "mean_score_ci95": mean_ci(scores),
    }


def collect_agent_results(weights_path, games, seed_value, include_random=False):
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"weights file not found: {weights_path}")
    trained = NTupleNetwork.load(weights_path)
    zero = NTupleNetwork("small")
    results = {}
    specs = [("Zero-weight greedy", zero)]
    specs.append(("Trained TD agent", trained))
    if include_random:
        specs.append(("Random-move baseline", None))
    for name, network in specs:
        print(f"Running {name}: {games} games...", flush=True)
        seed(seed_value)
        if network is None:
            scores, tiles, moves = run_random_games(games)
        else:
            scores, tiles, moves = run_games(games, network.sym, network.weights, 0.0, False)
        results[name] = {
            "score": np.asarray(scores, dtype=np.int64),
            "max_tile": np.asarray(tiles, dtype=np.int64),
            "moves": np.asarray(moves, dtype=np.int64),
        }
    return results


def mann_whitney_numpy(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    n1, n2 = len(left), len(right)
    combined = np.concatenate((left, right))
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(len(combined), dtype=np.float64)
    sorted_values = combined[order]
    i = 0
    tie_sum = 0.0
    while i < len(sorted_values):
        j = i + 1
        while j < len(sorted_values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        count = j - i
        tie_sum += count ** 3 - count
        i = j
    u1 = float(np.sum(ranks[:n1]) - n1 * (n1 + 1) / 2)
    u2 = float(n1 * n2 - u1)
    mean_u = n1 * n2 / 2.0
    total = n1 + n2
    variance = n1 * n2 / 12.0 * (total + 1 - tie_sum / (total * (total - 1))) if total > 1 else 0.0
    if variance <= 0:
        p_value = 1.0
        z_score = 0.0
    else:
        z_score = (abs(u1 - mean_u) - 0.5) / math.sqrt(variance)
        z_score = max(0.0, z_score)
        p_value = math.erfc(z_score / math.sqrt(2.0))
    pooled_denominator = n1 + n2 - 2
    pooled_sd = math.sqrt(((n1 - 1) * np.var(left, ddof=1) + (n2 - 1) * np.var(right, ddof=1)) / pooled_denominator) if n1 > 1 and n2 > 1 and pooled_denominator > 0 else 0.0
    cohens_d = float((np.mean(left) - np.mean(right)) / pooled_sd) if pooled_sd else 0.0
    common_language = float((np.sum(left[:, None] > right[None, :]) + 0.5 * np.sum(left[:, None] == right[None, :])) / (n1 * n2))
    return {"u_statistic": min(u1, u2), "p_value_two_sided_normal_approx": float(p_value), "z_score": float(z_score), "cohens_d_trained_minus_greedy": cohens_d, "common_language_trained_beats_greedy": common_language}


def write_raw_csv(results, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["agent", "game_index", "score", "max_tile", "moves"])
        for name, data in results.items():
            for i, (score, exponent, moves) in enumerate(zip(data["score"], data["max_tile"], data["moves"]), start=1):
                writer.writerow([name, i, int(score), int(2 ** int(exponent)), int(moves)])


def write_summaries(results, out_dir):
    summaries = {name: summarize_agent(data) for name, data in results.items()}
    csv_path = os.path.join(out_dir, "summary.csv")
    fields = ["agent", "games", "mean_score", "median_score", "std_score", "min_score", "max_score", "mean_moves", "mean_max_tile_exponent", "highest_tile", "mean_score_ci95_low", "mean_score_ci95_high", "reach_2048_ci95_low_percent", "reach_2048_ci95_high_percent"]
    fields += [f"reach_{tile}_percent" for tile in TILE_THRESHOLDS]
    with open(csv_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, summary in summaries.items():
            row = {"agent": name, **{key: value for key, value in summary.items() if not isinstance(value, (dict, list))}}
            row["mean_score_ci95_low"], row["mean_score_ci95_high"] = summary["mean_score_ci95"]
            row["reach_2048_ci95_low_percent"], row["reach_2048_ci95_high_percent"] = summary["reach_2048_ci95_percent"]
            for tile in TILE_THRESHOLDS:
                row[f"reach_{tile}_percent"] = summary["reach_rates_percent"][str(tile)]
            writer.writerow(row)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as stream:
        json.dump(summaries, stream, indent=2)
    return summaries


def print_summary_table(summaries):
    headers = ("Agent", "Games", "Mean score", "Median", "Std dev", "Min", "Max", "Mean moves", "Mean max exp", "Highest tile", "2048 rate")
    print("\n" + " | ".join(headers))
    print(" | ".join("---" for _ in headers))
    for name, summary in summaries.items():
        rate = summary["reach_rates_percent"]["2048"]
        print(f"{name} | {summary['games']} | {summary['mean_score']:.2f} | {summary['median_score']:.2f} | {summary['std_score']:.2f} | {summary['min_score']:.0f} | {summary['max_score']:.0f} | {summary['mean_moves']:.2f} | {summary['mean_max_tile_exponent']:.2f} | {summary['highest_tile']} | {rate:.2f}%")
        print(f"  mean score 95% CI: [{summary['mean_score_ci95'][0]:.2f}, {summary['mean_score_ci95'][1]:.2f}], 2048 Wilson 95% CI: [{summary['reach_2048_ci95_percent'][0]:.2f}%, {summary['reach_2048_ci95_percent'][1]:.2f}%]")
        for tile, percent in summary["reach_rates_percent"].items():
            print(f"  reach {tile}: {percent:.2f}%")


def plot_figures(results, summaries, out_dir, log_path=None):
    names = list(results)
    colors = [AGENT_COLORS[name] for name in names]
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#F5F7FA")
    means = [summaries[name]["mean_score"] for name in names]
    lower = [means[i] - summaries[name]["mean_score_ci95"][0] for i, name in enumerate(names)]
    upper = [summaries[name]["mean_score_ci95"][1] - means[i] for i, name in enumerate(names)]
    bars = ax.bar(names, means, color=colors, yerr=np.array([lower, upper]), capsize=5)
    ax.bar_label(bars, fmt="%.0f", padding=3)
    ax.set(title="Mean score by agent", xlabel="Agent", ylabel="Score (points)")
    ax.set_facecolor("#F5F7FA")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_mean_score.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#F5F7FA")
    x = np.arange(len(TILE_THRESHOLDS))
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        values = [summaries[name]["reach_rates_percent"][str(tile)] for tile in TILE_THRESHOLDS]
        bars = ax.bar(x + i * width, values, width, color=AGENT_COLORS[name], label=name)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)
    ax.set_xticks(x + width * (len(names) - 1) / 2, [str(tile) for tile in TILE_THRESHOLDS])
    ax.set(title="Tile reach rates", xlabel="Tile threshold", ylabel="Games reaching tile (%)")
    ax.set_facecolor("#F5F7FA")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_reach_rates.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    all_exponents = np.concatenate([data["max_tile"] for data in results.values()])
    tile_exponents = np.arange(7, int(np.max(all_exponents)) + 1)
    tile_labels = [str(2 ** int(e)) for e in tile_exponents]
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#F5F7FA")
    x = np.arange(len(tile_exponents))
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        values = [100 * float(np.mean(results[name]["max_tile"] == exponent)) for exponent in tile_exponents]
        bars = ax.bar(x + i * width, values, width, color=AGENT_COLORS[name], label=name)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)
    ax.set_xticks(x + width * (len(names) - 1) / 2, tile_labels, rotation=35, ha="right")
    ax.set(title="Final highest-tile distribution", xlabel="Highest tile in game", ylabel="Games ending at tile (%)")
    ax.set_facecolor("#F5F7FA")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_tile_distribution.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#F5F7FA")
    box = ax.boxplot([results[name]["score"] for name in names], tick_labels=names, patch_artist=True, showmeans=True)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set(title="Per-game score distribution", xlabel="Agent", ylabel="Score (points)")
    ax.set_facecolor("#F5F7FA")
    ax.legend(handles=[Patch(facecolor=AGENT_COLORS[name], alpha=0.7, label=name) for name in names])
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_score_boxplot.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#F5F7FA")
    combined_scores = np.concatenate([data["score"] for data in results.values()])
    use_log = np.min(combined_scores) > 0 and np.max(combined_scores) / np.min(combined_scores) > 100
    for name in names:
        ax.hist(results[name]["score"], bins=30, alpha=0.45, color=AGENT_COLORS[name], label=name)
    if use_log:
        ax.set_xscale("log")
    ax.set(title="Per-game score histogram", xlabel="Score (points)", ylabel="Games")
    ax.set_facecolor("#F5F7FA")
    ax.legend()
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_score_histogram.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#F5F7FA")
    for name in names:
        values = np.sort(results[name]["score"])
        y = np.arange(1, len(values) + 1) / len(values)
        ax.step(values, y, where="post", color=AGENT_COLORS[name], label=name)
    ax.set(title="Empirical cumulative distribution of scores", xlabel="Score (points)", ylabel="Fraction of games at or below score", ylim=(0, 1))
    ax.set_facecolor("#F5F7FA")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_score_ecdf.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#F5F7FA")
    box = ax.boxplot([results[name]["moves"] for name in names], tick_labels=names, patch_artist=True, showmeans=True)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set(title="Game length distribution", xlabel="Agent", ylabel="Moves per game")
    ax.set_facecolor("#F5F7FA")
    ax.legend(handles=[Patch(facecolor=AGENT_COLORS[name], alpha=0.7, label=name) for name in names])
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_moves_distribution.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#F5F7FA")
    for name in names:
        scores = results[name]["score"]
        moves = results[name]["moves"]
        ax.scatter(moves, scores, alpha=0.45, s=16, color=AGENT_COLORS[name], label=name)
        if len(np.unique(moves)) > 1:
            slope, intercept = np.polyfit(moves, scores, 1)
            xs = np.linspace(float(np.min(moves)), float(np.max(moves)), 100)
            ax.plot(xs, slope * xs + intercept, color=AGENT_COLORS[name], linewidth=2)
    ax.set(title="Score versus game length", xlabel="Moves per game", ylabel="Score (points)")
    ax.set_facecolor("#F5F7FA")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_score_vs_moves.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    if log_path:
        with open(log_path, newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if rows:
            keys = {re.sub(r"[^a-z0-9]", "", key.lower()): key for key in rows[0] if key}
            games_key = next((keys[k] for k in ("games", "traininggames", "totalgames") if k in keys), None)
            score_key = next((keys[k] for k in ("avgscore", "meanscore", "averagescore") if k in keys), None)
            if games_key and score_key:
                training_games = np.asarray([float(row[games_key]) for row in rows])
                training_scores = np.asarray([float(row[score_key]) for row in rows])
                baseline = summaries["Zero-weight greedy"]["mean_score"]
                fig, ax = plt.subplots(figsize=(9, 5), facecolor="#F5F7FA")
                ax.plot(training_games, training_scores, color=AGENT_COLORS["Trained TD agent"], label="Trained TD agent")
                ax.axhline(baseline, color=AGENT_COLORS["Zero-weight greedy"], linestyle="--", label="Zero-weight greedy baseline")
                crossing = np.flatnonzero(training_scores >= baseline)
                if len(crossing):
                    index = int(crossing[0])
                    ax.scatter([training_games[index]], [training_scores[index]], color=AGENT_COLORS["Trained TD agent"], zorder=3, label="First baseline crossing")
                ax.set(title="Training learning curve", xlabel="Training games", ylabel="Average score (points)")
                ax.set_facecolor("#F5F7FA")
                ax.legend()
                ax.grid(alpha=0.25)
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, "fig_learning_curve.png"), dpi=200, bbox_inches="tight")
                plt.close(fig)
            else:
                print("Skipping learning curve: log needs games and average-score columns.")
        else:
            print("Skipping learning curve: training log is empty.")

    fig, axes = plt.subplots(2, 2, figsize=(13.33, 7.5), facecolor="#F5F7FA")
    bars = axes[0, 0].bar(names, means, color=colors, yerr=np.array([lower, upper]), capsize=4)
    axes[0, 0].bar_label(bars, fmt="%.0f", padding=2, fontsize=8)
    axes[0, 0].set(title="Mean score", ylabel="Score (points)")
    x = np.arange(len(TILE_THRESHOLDS))
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        axes[0, 1].bar(x + i * width, [summaries[name]["reach_rates_percent"][str(t)] for t in TILE_THRESHOLDS], width, color=AGENT_COLORS[name], label=name)
    axes[0, 1].set_xticks(x + width * (len(names) - 1) / 2, [str(t) for t in TILE_THRESHOLDS], rotation=30)
    axes[0, 1].set(title="Tile reach rates", xlabel="Tile", ylabel="Games reaching tile (%)")
    box = axes[1, 0].boxplot([results[name]["score"] for name in names], tick_labels=names, patch_artist=True, showmeans=True)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
    axes[1, 0].set(title="Score distribution", ylabel="Score (points)")
    for name in names:
        values = np.sort(results[name]["score"])
        axes[1, 1].step(values, np.arange(1, len(values) + 1) / len(values), where="post", color=AGENT_COLORS[name], label=name)
    axes[1, 1].set(title="Score ECDF", xlabel="Score (points)", ylabel="Fraction at or below score", ylim=(0, 1))
    for axis in axes.flat:
        axis.set_facecolor("#F5F7FA")
        axis.grid(alpha=0.22, axis="y")
    axes[0, 1].legend(fontsize=8)
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("2048 agent comparison", fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_dashboard.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="compare_results")
    parser.add_argument("--log", default=None)
    parser.add_argument("--include-random", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.games < 1:
        raise SystemExit("--games must be at least 1")
    os.makedirs(args.out, exist_ok=True)
    try:
        results = collect_agent_results(args.weights, args.games, args.seed, args.include_random)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    write_raw_csv(results, os.path.join(args.out, "games.csv"))
    summaries = write_summaries(results, args.out)
    print_summary_table(summaries)
    score_test = mann_whitney_numpy(results["Trained TD agent"]["score"], results["Zero-weight greedy"]["score"])
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as stream:
        json.dump(score_test, stream, indent=2)
    ratio = summaries["Trained TD agent"]["mean_score"] / summaries["Zero-weight greedy"]["mean_score"] if summaries["Zero-weight greedy"]["mean_score"] else None
    difference = summaries["Trained TD agent"]["mean_score"] - summaries["Zero-weight greedy"]["mean_score"]
    print(f"\nTrained over greedy mean score: difference {difference:.2f}; ratio {ratio:.4f}" if ratio is not None else f"\nTrained over greedy mean score: difference {difference:.2f}; ratio undefined because the greedy mean is zero")
    print(f"Mann–Whitney U={score_test['u_statistic']:.2f}, approximate two-sided p={score_test['p_value_two_sided_normal_approx']:.6g}, Cohen's d={score_test['cohens_d_trained_minus_greedy']:.4f}, common-language effect={100 * score_test['common_language_trained_beats_greedy']:.2f}%.")
    print(f"The trained agent scores higher than the greedy agent in {100 * score_test['common_language_trained_beats_greedy']:.2f}% of random pairings (ties count as half).")
    plot_figures(results, summaries, args.out, args.log)
    print(f"\nSaved comparison outputs to {args.out}")


if __name__ == "__main__":
    main()
