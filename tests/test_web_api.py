import numpy as np
import pytest

from game2048.agent import greedy_action
from game2048.board import new_board
from game2048.ntuple import NTupleNetwork, update
from web.inspect_agent import GameSession, changed_entries, decision


def test_instrumented_decision_matches_greedy_and_update_matches_engine():
    net = NTupleNetwork("small")
    rng = np.random.default_rng(14)
    net.weights[:] = rng.normal(0, 0.01, net.weights.shape).astype(np.float32)
    board = new_board()
    buf = np.zeros((4, 16), np.uint8)
    normal_action, _ = greedy_action(board, net.sym, net.weights, buf)
    inspected_action, _, _ = decision(board, net)
    assert inspected_action == normal_action

    after = buf[normal_action].copy()
    delta = 17.25
    expected = net.weights.copy()
    update(after, net.sym, expected, 0.1 * delta / net.n_symmetries)
    changed_entries(after, net, delta)
    assert np.array_equal(net.weights, expected)


def test_undo_restores_weights_and_redo_replays_identically():
    net = NTupleNetwork("small")
    session = GameSession(net, "small")
    session.reset("step", "fresh")
    original = session.net.weights.copy()
    generated = []
    for _ in range(5):
        record = session.agent_step(detail=True, learning=True)
        if record.get("terminal"):
            break
        generated.append(record)
    final_weights = session.net.weights.copy()
    final_state = session.state()
    for _ in generated:
        session.undo()
    assert np.array_equal(session.net.weights, original)
    for expected_record in generated:
        actual_record = session.agent_step(detail=True, learning=True)
        assert actual_record["chosen"] == expected_record["chosen"]
        assert actual_record["spawn"] == expected_record["spawn"]
        assert actual_record["state"] == expected_record["state"]
    assert np.array_equal(session.net.weights, final_weights)
    assert session.state() == final_state


def test_zero_fresh_network_ties_without_merges_and_first_td_delta_is_reward():
    session = GameSession(NTupleNetwork("small"), "small")
    session.reset("step", "fresh")
    session.board[:] = 0
    session.board[5] = 2
    session.board[10] = 1
    first = session.agent_step(detail=True, learning=True)
    legal = [item for item in first["evaluations"] if item["legal"]]
    assert len(legal) == 4
    assert {item["score"] for item in legal} == {0.0}
    assert first["chosen"] == "up"
    assert first["update"] is None
    second = session.agent_step(detail=True, learning=True)
    assert second["update"]["delta"] == second["update"]["reward"]


def test_flask_api_state_validation_and_agent_step():
    pytest.importorskip("flask")
    from web import server

    server.session = GameSession(NTupleNetwork("small"), "small")
    client = server.app.test_client()
    state = client.get("/api/state").get_json()
    assert len(state["board"]) == 4 and len(state["board"][0]) == 4
    assert all(isinstance(tile, int) for row in state["board"] for tile in row)
    invalid = client.post("/api/human/move", json={"direction": "diagonal"})
    assert invalid.status_code == 400
    client.post("/api/reset", json={"mode": "step", "weights_source": "fresh"})
    step = client.post("/api/agent/step?detail=0", json={}).get_json()
    assert step["chosen"] in ("up", "right", "down", "left")
    assert "state" in step and "afterstate" in step
