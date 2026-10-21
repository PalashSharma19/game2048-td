import numpy as np

from game2048.agent import greedy_action
from game2048.board import ACTION_NAMES, move, new_board, spawn_tile
from game2048.ntuple import NTupleNetwork, evaluate, update


ALPHA = 0.1


def lookup_entries(board, net):
    entries = []
    for p in range(net.sym.shape[0]):
        lookups = []
        total = 0.0
        for s in range(net.sym.shape[1]):
            cells = net.sym[p, s]
            pattern = [int(board[c]) for c in cells]
            index = 0
            for exponent in pattern:
                index = index * 16 + min(exponent, 15)
            weight = float(net.weights[p, index])
            total += weight
            lookups.append({"tuple_id": p, "symmetry_id": s, "index": index, "pattern": pattern, "cells": [int(c) for c in cells], "weight": weight})
        entries.append({"tuple_id": p, "sum": total, "lookups": lookups})
    return entries


def value_breakdown(board, net):
    tuples = lookup_entries(board, net)
    return float(evaluate(board, net.sym, net.weights)), tuples


def decision(board, net, detail=True):
    buf = np.zeros((4, 16), np.uint8)
    action, reward = greedy_action(board, net.sym, net.weights, buf)
    candidates = []
    for a in range(4):
        after = np.zeros(16, np.uint8)
        r = int(move(board, a, after))
        if r < 0:
            candidates.append({"action": ACTION_NAMES[a], "legal": False, "reward": 0, "after": after.copy(), "value": None, "score": None, "tuples": []})
        else:
            if detail:
                val, breakdown = value_breakdown(after, net)
            else:
                val, breakdown = float(evaluate(after, net.sym, net.weights)), []
            candidates.append({"action": ACTION_NAMES[a], "legal": True, "reward": r, "after": after.copy(), "value": val, "score": r + val, "tuples": breakdown})
    return action, reward, candidates


def changed_entries(board, net, delta, alpha=ALPHA):
    frequencies = {}
    for p in range(net.sym.shape[0]):
        for s in range(net.sym.shape[1]):
            index = 0
            for c in net.sym[p, s]:
                index = index * 16 + min(int(board[c]), 15)
            frequencies.setdefault((p, index), []).append(s)
    old = {(p, index): float(net.weights[p, index]) for p, index in frequencies}
    update(board, net.sym, net.weights, alpha * delta / net.n_symmetries)
    result = []
    for (p, index), symmetries in frequencies.items():
        new = float(net.weights[p, index])
        result.append({"tuple_id": p, "index": index, "count": len(symmetries), "symmetry_ids": symmetries, "old": old[(p, index)], "new": new, "change": new - old[(p, index)]})
    return result


class GameSession:
    def __init__(self, trained, network_name):
        self.trained = trained
        self.fresh = NTupleNetwork("small")
        self.network_name = network_name
        self.mode = "human"
        self.weights_source = "trained"
        self.learning = True
        self.reset("human", "trained")

    @property
    def net(self):
        return self.trained if self.weights_source == "trained" else self.fresh

    def reset(self, mode=None, weights_source=None):
        if mode is not None:
            self.mode = mode
        if weights_source is not None:
            self.weights_source = weights_source
        self.board = new_board()
        self.score = 0
        self.moves = 0
        self.prev_after = None
        self.game_over = False
        self.terminal_finalized = False
        self.reached_2048 = bool(np.any(self.board >= 11))
        self.reached_2048_ever = self.reached_2048
        self.history = []
        self.cursor = 0
        self.td_errors = []
        self.updates = 0
        return self.state()

    def state(self):
        exponents = np.asarray(self.board, dtype=np.uint8).reshape(4, 4)
        return {"mode": self.mode, "weights_source": self.weights_source, "board": self._values(exponents), "exponents": exponents.tolist(), "score": int(self.score), "moves": int(self.moves), "max_tile": int(2 ** int(self.board.max())), "game_over": bool(self.game_over), "reached_2048": bool(self.reached_2048), "reached_2048_ever": bool(self.reached_2048_ever), "terminal_finalized": bool(self.terminal_finalized), "learning": bool(self.learning), "td_updates": int(self.updates), "td_errors": self.td_errors[-100:]}

    def continue_after_2048(self):
        self.reached_2048 = False

    def set_learning(self, learning):
        self.learning = bool(learning)

    @staticmethod
    def _values(exponents):
        return np.where(exponents > 0, np.left_shift(1, exponents.astype(np.int64)), 0).tolist()

    def human_move(self, direction):
        if direction not in ACTION_NAMES:
            raise ValueError("direction must be up, right, down, or left")
        after = np.zeros(16, np.uint8)
        reward = int(move(self.board, ACTION_NAMES.index(direction), after))
        if reward < 0 or self.game_over:
            return {"legal": False, "reward": 0, "state": self.state()}
        self.board = after
        self.score += reward
        self.moves += 1
        spawn = self._spawn()
        self._check_over()
        reached_now = not self.reached_2048_ever and bool(np.any(self.board >= 11))
        self.reached_2048_ever = self.reached_2048_ever or bool(np.any(self.board >= 11))
        self.reached_2048 = reached_now
        return {"legal": True, "reward": reward, "spawn": spawn, "reached_2048_now": reached_now, "state": self.state()}

    def _spawn(self):
        before = self.board.copy()
        spawn_tile(self.board)
        positions = np.flatnonzero((before == 0) & (self.board != 0))
        if len(positions) == 0:
            return None
        return {"position": int(positions[0]), "value": int(2 ** int(self.board[positions[0]]))}

    def _check_over(self):
        buf = np.zeros((4, 16), np.uint8)
        self.game_over = all(int(move(self.board, a, buf[a])) < 0 for a in range(4))

    def agent_step(self, detail=True, learning=None):
        if learning is not None:
            self.learning = bool(learning)
        if self.cursor < len(self.history):
            entry = self.history[self.cursor]
            self._apply_entry(entry, True)
            self.cursor += 1
            return entry["record"]
        if self.game_over and self.terminal_finalized:
            return {"terminal": True, "state": self.state(), "message": "Game over"}
        before_board = self.board.copy()
        before_score, before_moves = self.score, self.moves
        before_prev = None if self.prev_after is None else self.prev_after.copy()
        before_reached = self.reached_2048
        before_reached_ever = self.reached_2048_ever
        before_game_over = self.game_over
        before_terminal_finalized = self.terminal_finalized
        action, reward, candidates = decision(self.board, self.net, detail)
        if action < 0:
            update_detail = None
            if self.prev_after is not None:
                old_value = float(evaluate(self.prev_after, self.net.sym, self.net.weights))
                delta = -old_value
                diffs = changed_entries(self.prev_after, self.net, delta) if self.learning else []
                update_detail = {"terminal": True, "previous_after": self.prev_after.copy(), "previous_value_before": old_value, "target": 0.0, "delta": delta, "alpha": ALPHA, "step_size": ALPHA * delta / self.net.n_symmetries, "diffs": diffs, "value_after": float(evaluate(self.prev_after, self.net.sym, self.net.weights)), "applied": self.learning}
                if self.learning:
                    self.updates += 1
                self.td_errors.append(delta)
            self.game_over = True
            self.terminal_finalized = True
            record = {"terminal": True, "update": update_detail, "state": self.state()}
            entry = self._make_entry(record, before_board, before_score, before_moves, before_prev, before_reached, before_reached_ever, before_game_over, before_terminal_finalized, None, update_detail)
            self.history.append(entry)
            self.cursor += 1
            return record
        after = candidates[action]["after"].copy()
        update_detail = None
        if self.prev_after is not None:
            old_value, old_breakdown = value_breakdown(self.prev_after, self.net)
            next_value = float(evaluate(after, self.net.sym, self.net.weights))
            delta = reward + next_value - old_value
            diffs = changed_entries(self.prev_after, self.net, delta) if self.learning else []
            update_detail = {"terminal": False, "previous_after": self.prev_after.copy(), "previous_value_before": old_value, "target": reward + next_value, "reward": reward, "next_value": next_value, "delta": delta, "alpha": ALPHA, "step_size": ALPHA * delta / self.net.n_symmetries, "diffs": diffs, "value_after": float(evaluate(self.prev_after, self.net.sym, self.net.weights)), "previous_tuples": old_breakdown, "applied": self.learning}
            if self.learning:
                self.updates += 1
            self.td_errors.append(delta)
        self.board = after.copy()
        self.prev_after = after.copy()
        self.score += reward
        self.moves += 1
        spawn = self._spawn()
        self._check_over()
        self.terminal_finalized = False
        before_reached_now = not before_reached_ever and bool(np.any(self.board >= 11))
        self.reached_2048_ever = self.reached_2048_ever or bool(np.any(self.board >= 11))
        self.reached_2048 = before_reached_now
        scores = [candidate["score"] for candidate in candidates if candidate["legal"]]
        best_score = max(scores)
        tied = [candidate["action"] for candidate in candidates if candidate["legal"] and candidate["score"] == best_score]
        record = {"terminal": False, "before": before_board, "evaluations": candidates if detail else [{k: v for k, v in candidate.items() if k != "tuples"} for candidate in candidates], "chosen": ACTION_NAMES[action], "tied_moves": tied, "tie_break": "first in Up, Right, Down, Left order", "afterstate": after, "spawn": spawn, "update": update_detail, "reached_2048_now": before_reached_now, "state": self.state()}
        entry = self._make_entry(record, before_board, before_score, before_moves, before_prev, before_reached, before_reached_ever, before_game_over, before_terminal_finalized, spawn, update_detail)
        self.history.append(entry)
        self.cursor += 1
        return record

    def _make_entry(self, record, board, score, moves, prev, reached, reached_ever, game_over, terminal_finalized, spawn, update_detail):
        diffs = [] if update_detail is None else update_detail["diffs"]
        return {"record": record, "before_board": board, "before_score": score, "before_moves": moves, "before_prev": prev, "before_reached": reached, "before_reached_ever": reached_ever, "before_game_over": game_over, "before_terminal_finalized": terminal_finalized, "spawn": spawn, "diffs": diffs, "learning": self.learning}

    def _apply_entry(self, entry, redo):
        if redo:
            for diff in entry["diffs"]:
                self.net.weights[diff["tuple_id"], diff["index"]] = diff["new"]
            state = entry["record"].get("state")
            if state:
                self.board = np.asarray(state["exponents"], np.uint8).reshape(16)
                self.score = state["score"]
                self.moves = state["moves"]
                self.game_over = state["game_over"]
                self.reached_2048 = state["reached_2048"]
                self.reached_2048_ever = state["reached_2048_ever"]
                self.terminal_finalized = state["terminal_finalized"]
                update_detail = entry["record"].get("update")
                self.prev_after = None if entry["record"].get("terminal") else (None if entry.get("after") is None else entry["after"].copy())
                if self.prev_after is None and not entry["record"].get("terminal"):
                    self.prev_after = np.asarray(entry["record"]["afterstate"], np.uint8).copy()
                if update_detail:
                    if update_detail.get("applied"):
                        self.updates += 1
                    self.td_errors.append(update_detail["delta"])
            return
        for diff in entry["diffs"]:
            self.net.weights[diff["tuple_id"], diff["index"]] = diff["old"]
        self.board = entry["before_board"].copy()
        self.score = entry["before_score"]
        self.moves = entry["before_moves"]
        self.prev_after = None if entry["before_prev"] is None else entry["before_prev"].copy()
        self.reached_2048 = entry["before_reached"]
        self.reached_2048_ever = entry["before_reached_ever"]
        self.game_over = entry["before_game_over"]
        self.terminal_finalized = entry["before_terminal_finalized"]
        update_detail = entry["record"].get("update")
        if update_detail:
            if update_detail.get("applied"):
                self.updates = max(0, self.updates - 1)
            if self.td_errors:
                self.td_errors.pop()

    def undo(self):
        if self.cursor <= 0:
            raise ValueError("There is no previous step")
        self.cursor -= 1
        entry = self.history[self.cursor]
        self._apply_entry(entry, False)
        return {"record": entry["record"], "state": self.state()}
