# 2048 TD(0) project audit

Read-only source audit. No training was run.

1. **The TD update divides `alpha * err` by `n_feat`, which is `sym.shape[1]` (8 symmetries); default training alpha is 0.1.**

   `game2048/agent.py:55,69–72`
   ```python
   n_feat = sym.shape[1]          # step size is split across the 8 symmetries
   ...
   if learn and has_prev:
       target = r + evaluate(after, sym, weights)
       err = target - evaluate(prev_after, sym, weights)
       update(prev_after, sym, weights, alpha * err / n_feat)
   ```

   `train.py:23`
   ```python
   ap.add_argument("--alpha", type=float, default=0.1, help="learning rate")
   ```

2. **Delta is `r + V(after) - V(previous afterstate)`; the previous board is stored, but its value is recomputed with current weights when the update runs.**

   `game2048/agent.py:53,68–74`
   ```python
   prev_after = np.zeros(16, np.uint8)
   ...
   after = buf[a]
   if learn and has_prev:
       target = r + evaluate(after, sym, weights)
       err = target - evaluate(prev_after, sym, weights)
       update(prev_after, sym, weights, alpha * err / n_feat)

   prev_after[:] = after
   ```

3. **At game over, if learning is enabled and a prior afterstate exists, the code applies a final update with target 0; otherwise it skips that update.**

   `game2048/agent.py:61–66`
   ```python
   if a < 0:
       # Terminal: the previous afterstate leads to game over, target 0.
       if learn and has_prev:
           err = 0.0 - evaluate(prev_after, sym, weights)
           update(prev_after, sym, weights, alpha * err / n_feat)
       break
   ```

4. **Equal scores are resolved in favor of the first legal move encountered, and the action order is up, right, down, left.**

   `game2048/agent.py:32–40`; `game2048/board.py:14–15`
   ```python
   for a in range(4):
       r = move(board, a, buf[a])
       if r < 0:
           continue
       v = r + evaluate(buf[a], sym, weights)
       if v > best_v:
           best_v = v
           best_a = a
           best_r = r
   ```
   ```python
   UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
   ACTION_NAMES = ("up", "right", "down", "left")
   ```

5. **The learned agent is greedy and has no epsilon-greedy branch; a separate random baseline samples a legal move uniformly.**

   `game2048/agent.py:10–12,32–40,121`
   ```python
   Policy (greedy, no epsilon is needed because tile spawns already give
   plenty of exploration):
       a_t = argmax_a [ r(s_t, a) + V(s'(s_t, a)) ]
   ...
   for a in range(4):
       ...
       if v > best_v:
   ...
   k = np.random.randint(n_legal)
   ```

6. **No gamma variable or discount multiplier appears in the update, so it uses the undiscounted next value (implicitly gamma = 1).**

   `game2048/agent.py:70–72`
   ```python
   target = r + evaluate(after, sym, weights)
   err = target - evaluate(prev_after, sym, weights)
   update(prev_after, sym, weights, alpha * err / n_feat)
   ```

7. **The board is a one-dimensional NumPy array with 16 `uint8` entries storing exponents, not a packed 64-bit integer.**

   `game2048/board.py:4–6,88–92`
   ```python
   A board is a 1-D NumPy array of 16 uint8 values in row-major order
   (cell index = 4 * row + col). Each value is the tile's exponent:
   0 = empty, 1 = 2, 2 = 4, 3 = 8, ..., 11 = 2048.
   ...
   def new_board():
       board = np.zeros(16, np.uint8)
   ```

8. **Each merge contributes the value of the newly created tile to the move reward, and the reward sums those merged tile values.**

   `game2048/board.py:31–33,47–50`
   ```python
   Returns the reward (sum of newly merged tile values), or -1 if the move
   does not change the board (an illegal move). Each tile merges at most
   once per move, as in the original game.
   ...
   reward += 1 << (v + 1)
   ```

9. **The spawn chance is 90% for a 2 and 10% for a 4, at a uniformly selected empty cell.**

   `game2048/board.py:69–81`
   ```python
   """Place a 2 (p = 0.9) or a 4 (p = 0.1) on a uniformly random empty cell."""
   ...
   board[i] = 1 if np.random.random() < 0.9 else 2
   ```

10. **`large` contains four 6-cell tuples, `small` contains five 4-cell tuples, and weights use `float32`.**

    `game2048/ntuple.py:20–38,103`
    ```python
    NETWORKS = {
        "large": [
            [0, 1, 2, 3, 4, 5],
            [4, 5, 6, 7, 8, 9],
            [0, 1, 2, 4, 5, 6],
            [4, 5, 6, 8, 9, 10],
        ],
        "small": [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 4, 5],
            [1, 2, 5, 6],
            [5, 6, 9, 10],
        ],
    }
    ...
    self.weights = np.zeros((len(self.patterns), 16 ** self.tuple_len), np.float32)
    ```

11. **The defaults are 100,000 games, alpha 0.1, and network `large`.**

    `train.py:21–23`
    ```python
    ap.add_argument("--network", default="large", choices=["large", "small"])
    ap.add_argument("--games", type=int, default=100_000, help="training games")
    ap.add_argument("--alpha", type=float, default=0.1, help="learning rate")
    ```

12. **Weights are written to `<out>/weights.npz` every ten chunks and at the final chunk (default `<out>` is `runs/<network>`); the save overwrites the destination, and without `--resume` the network starts at zero even if that file already exists. Thus the example command starts from zero and will overwrite `runs/large/weights.npz` when it saves.**

    `train.py:26–34,55–57,71–72`; `game2048/ntuple.py:112–114`
    ```python
    ap.add_argument("--out", default=None, help="output folder (default runs/<network>)")
    ap.add_argument("--resume", default=None, help="weights .npz to continue from")
    ...
    out = args.out or os.path.join("runs", args.network)
    ...
    net = NTupleNetwork.load(args.resume) if args.resume else NTupleNetwork(args.network)
    ...
    scores, tiles, moves = run_games(n, net.sym, net.weights, args.alpha, True)
    ...
    if done % (args.chunk * 10) == 0 or done == args.games:
        net.save(os.path.join(out, "weights.npz"))
    ```
    ```python
    def save(self, path):
        np.savez_compressed(path, weights=self.weights,
                            meta=json.dumps({"name": self.name, "patterns": self.patterns}))
    ```

13. **The web server loads weights once in `main()` at startup and gives that in-memory network to the session; resets do not load the file.**

    `web/server.py:112–123`
    ```python
    if not os.path.isfile(args.weights):
        parser.error("weights file not found: " + args.weights)
    try:
        trained = NTupleNetwork.load(args.weights)
    ...
    session = GameSession(trained, trained.name)
    ```

14. **The weights-write hits found are `net.save(...)` in training and its implementation via `np.savez_compressed`; web code has no weights save path. Other write-mode hits found are the training CSV and evaluation JSON outputs.**

    `train.py:46,72`; `game2048/ntuple.py:112–114`; `evaluate.py:44–45`
    ```python
    f = open(log_path, "w" if new_log else "a", newline="")
    ...
    net.save(os.path.join(out, "weights.npz"))
    ```
    ```python
    def save(self, path):
        np.savez_compressed(path, weights=self.weights,
                            meta=json.dumps({"name": self.name, "patterns": self.patterns}))
    ```
    ```python
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    ```

15. **The reset endpoint changes mode/source and resets game state while retaining the same trained network object in memory; switching sources does not reload or copy its weights.**

    `web/server.py:49–59`; `web/inspect_agent.py:70–88`
    ```python
    source = body.get("weights_source", session.weights_source)
    ...
    return jsonify(clean(session.reset(mode, source)))
    ```
    ```python
    self.trained = trained
    ...
    def net(self):
        return self.trained if self.weights_source == "trained" else self.fresh

    def reset(self, mode=None, weights_source=None):
        if mode is not None:
            self.mode = mode
        if weights_source is not None:
            self.weights_source = weights_source
        self.board = new_board()
    ```

16. **One zero-initialized fresh network is created when each `GameSession` is constructed and reused across resets; the source selector chooses either that object or the trained object.**

    `web/inspect_agent.py:70–81,83–88`; `game2048/ntuple.py:103`
    ```python
    self.trained = trained
    self.fresh = NTupleNetwork("small")
    ...
    def net(self):
        return self.trained if self.weights_source == "trained" else self.fresh
    ```
    ```python
    self.weights = np.zeros((len(self.patterns), 16 ** self.tuple_len), np.float32)
    ```

17. **When Learning is ON for the trained source, learned weights persist in memory across Reset Board and Play Again; reset does not restore original weights, and no original-weight restore path was found.**

    `web/inspect_agent.py:61,70–100`; `web/static/app.js:10,13`
    ```python
    update(board, net.sym, net.weights, alpha * delta / net.n_symmetries)
    ...
    self.trained = trained
    ...
    def reset(self, mode=None, weights_source=None):
        ...
        self.board = new_board()
        self.score = 0
        self.moves = 0
        self.prev_after = None
    ```
    ```javascript
    ...$("#play-again").onclick=()=>reset(activeMode);...
    async function reset(mode=activeMode,learning=$("#learning").checked){...const s=await api("/api/reset",{mode,weights_source:$("#weights-source").value,learning});...}
    ```

18. **The session starts with Learning ON; changing to Step-by-Step sets it ON and changing to Autoplay sets it OFF. This mode default is independent of trained versus fresh weights.**

    `web/inspect_agent.py:74–77`; `web/static/app.js:27`
    ```python
    self.mode = "human"
    self.weights_source = "trained"
    self.learning = True
    ```
    ```javascript
    $$('.mode-button').forEach(b=>b.addEventListener("click",()=>{if(b.dataset.mode!==activeMode)askSwitch(()=>reset(b.dataset.mode,b.dataset.mode==="autoplay"?false:true))}));
    ```

19. **Evaluation passes `alpha=0.0` and `learn=False`, so it does not update the weights object; it writes the results JSON to `--out` (default `results.json`), not the weights file.**

    `evaluate.py:38–45`
    ```python
    if args.weights:
        net = NTupleNetwork.load(args.weights)
        seed(args.seed)
        results["td_agent"] = summarize(*run_games(args.games, net.sym, net.weights, 0.0, False))
        report(f"TD agent [{net.name}]", results["td_agent"])

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    ```

## Surprising or risky findings

- A normal `train.py` run does not detect or load an existing `runs/<network>/weights.npz`; it creates a new network unless `--resume` is given, then may overwrite that path during saving.
- Web learning can change the trained weights in memory, but web code does not persist those changes to disk.
- Evaluation does not modify the weights file, but it does overwrite its results output file when that path already exists.
- In the UI, the initial session is `human` mode; agent mode buttons set Step-by-Step learning ON and Autoplay OFF when switching modes. The checkbox itself is populated from the session state.
