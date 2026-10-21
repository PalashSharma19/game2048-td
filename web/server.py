import argparse
import os
import sys

import numpy as np
from flask import Flask, jsonify, request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from game2048 import NTupleNetwork
if __package__:
    from .inspect_agent import GameSession
else:
    from inspect_agent import GameSession


app = Flask(__name__, static_folder="static", static_url_path="")
session = None


def clean(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def error(message, status=400):
    return jsonify({"error": message}), status


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/state")
def get_state():
    return jsonify(clean(session.state()))


@app.post("/api/reset")
def reset():
    body = request.get_json(silent=True) or {}
    mode = body.get("mode", session.mode)
    source = body.get("weights_source", session.weights_source)
    if mode not in ("human", "step", "autoplay"):
        return error("mode must be human, step, or autoplay")
    if source not in ("trained", "fresh"):
        return error("weights_source must be trained or fresh")
    session.learning = bool(body.get("learning", session.learning))
    return jsonify(clean(session.reset(mode, source)))


@app.post("/api/human/move")
def human_move():
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(clean(session.human_move(body.get("direction"))))
    except ValueError as exc:
        return error(str(exc))


@app.post("/api/continue")
def continue_game():
    session.continue_after_2048()
    return jsonify(clean(session.state()))


@app.post("/api/settings")
def settings():
    body = request.get_json(silent=True) or {}
    session.set_learning(bool(body.get("learning", session.learning)))
    return jsonify(clean(session.state()))


@app.post("/api/agent/step")
def agent_step():
    body = request.get_json(silent=True) or {}
    detail = request.args.get("detail", body.get("detail", True))
    if isinstance(detail, str):
        detail = detail.lower() not in ("0", "false", "no")
    learning = body.get("learning", request.args.get("learning"))
    if isinstance(learning, str):
        learning = learning.lower() in ("1", "true", "yes", "on")
    try:
        return jsonify(clean(session.agent_step(bool(detail), learning)))
    except Exception as exc:
        return error(str(exc))


@app.post("/api/agent/undo")
def agent_undo():
    try:
        return jsonify(clean(session.undo()))
    except ValueError as exc:
        return error(str(exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=os.path.join("runs", "large", "weights.npz"))
    parser.add_argument("--network", choices=("large", "small"), default=None)
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        parser.error("weights file not found: " + args.weights)
    try:
        trained = NTupleNetwork.load(args.weights)
    except Exception as exc:
        parser.error("could not load weights file: " + str(exc))
    if args.network and args.network != trained.name:
        parser.error("--network is " + args.network + " but the weights contain network '" + trained.name + "'")
    global session
    session = GameSession(trained, trained.name)
    print("Loaded " + trained.name + " network from " + args.weights)
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
