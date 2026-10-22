import csv
import json

import numpy as np

import compare
from game2048 import NTupleNetwork, run_games, seed


def test_zero_weight_agent_is_reproducible():
    network = NTupleNetwork("small")
    seed(314)
    first = run_games(3, network.sym, network.weights, 0.0, False)
    seed(314)
    second = run_games(3, network.sym, network.weights, 0.0, False)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)


def test_comparison_does_not_mutate_loaded_weights(tmp_path, monkeypatch):
    weights_path = tmp_path / "model.input"
    weights_path.write_text("placeholder", encoding="utf-8")
    loaded = NTupleNetwork("small")
    loaded.weights[0, 1] = 7.0
    before = loaded.weights.copy()
    monkeypatch.setattr(compare.NTupleNetwork, "load", lambda path: loaded)
    compare.collect_agent_results(str(weights_path), 2, 7)
    np.testing.assert_array_equal(loaded.weights, before)


def test_summary_has_equal_games_per_agent():
    data = {
        "score": np.array([0, 4]),
        "max_tile": np.array([1, 2]),
        "moves": np.array([3, 5]),
    }
    summary = {name: compare.summarize_agent(data) for name in ("first", "second", "third")}
    assert {item["games"] for item in summary.values()} == {2}


def test_compare_cli_creates_outputs_with_mocked_network(tmp_path, monkeypatch):
    weights_path = tmp_path / "model.input"
    weights_path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(compare.NTupleNetwork, "load", lambda path: NTupleNetwork("small"))
    out_dir = tmp_path / "comparison"
    compare.main(["--weights", str(weights_path), "--games", "2", "--seed", "4", "--out", str(out_dir), "--include-random"])
    expected = ["games.csv", "summary.csv", "summary.json", "stats.json", "fig_mean_score.png", "fig_reach_rates.png", "fig_tile_distribution.png", "fig_score_boxplot.png", "fig_score_histogram.png", "fig_score_ecdf.png", "fig_moves_distribution.png", "fig_score_vs_moves.png", "fig_dashboard.png"]
    assert all((out_dir / name).is_file() for name in expected)
    assert not (out_dir / "fig_learning_curve.png").exists()
    with open(out_dir / "summary.json", encoding="utf-8") as stream:
        summaries = json.load(stream)
    assert {value["games"] for value in summaries.values()} == {2}
    with open(out_dir / "games.csv", newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 6
