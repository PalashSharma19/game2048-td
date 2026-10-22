# Complete Project Guide: 2048 with TD(0) and N-Tuple Networks

This guide explains the project from the game representation through the learning algorithm, command-line tools, trained weights, and local web interface. It describes the implementation in this repository, including the actual Python engine and API data formats.

## 1. What the project makes

This project implements the 2048 puzzle and an agent that learns to play it by playing games against itself. The agent uses temporal-difference learning, specifically TD(0), to estimate how promising a board is. The estimate is represented by an n-tuple network: a collection of small patterns of board cells, each with a learned lookup table.

The repository includes:

- A fast Python game engine, with hot functions compiled by Numba.
- A symmetric n-tuple value function and weight persistence.
- A greedy afterstate agent and self-play training loop.
- Evaluation, terminal playback, and plotting tools.
- A local Flask web UI for human play, detailed agent inspection, and autoplay.
- Tests for game mechanics, network symmetry, instrumentation, TD updates, history replay, and the web API.

The web interface is a client for the Python engine. The browser sends moves to Python and displays its responses. It does not contain a second implementation of the 2048 rules or the agent's value function.

## 2. Repository map

```text
game2048/
  __init__.py       Public package exports
  board.py          Board representation, moves, merging, random spawns
  ntuple.py         Tuple layouts, symmetries, value evaluation, weight updates
  agent.py          Greedy policy, TD(0) self-play, random baseline, metrics

web/
  server.py         Flask app, command-line startup, JSON API
  inspect_agent.py  Web session, decision breakdown, update diffs, undo/redo
  static/
    index.html      Page structure and controls
    style.css       Dark responsive layout, tile colors, animation styles
    app.js          Browser events, API calls, rendering, autoplay timer

tests/
  test_game.py      Original game and n-tuple symmetry tests
  test_web_api.py   Web instrumentation, learning, undo/redo, and API tests

train.py            Train or resume an agent and write logs/weights
evaluate.py         Compare a random policy and trained agent
play.py             Watch one trained game in a terminal
plot.py             Plot training and evaluation summaries
requirements.txt    Python dependencies
README.md           Project overview and quick-start instructions
PROJECT_GUIDE.md    This detailed guide

runs/large/
  weights.npz       Included trained large-network weights
  train_log.csv     Training metrics for the included run
```

The repository also contains generated plot images and evaluation output. They are results/artifacts, not required by the engine or the web UI.

## 3. Board representation and game rules

### The board is an exponent array

The engine does **not** store the board as a single 64-bit bitboard. It stores it as a NumPy array of 16 unsigned 8-bit integers (`uint8`) in row-major order. Each array value is the exponent of the tile:

| Array value | Tile shown to a person |
|---:|---:|
| 0 | Empty |
| 1 | 2 |
| 2 | 4 |
| 3 | 8 |
| 11 | 2048 |

For a non-empty cell with exponent `e`, the visible tile value is `2**e`. The row-major cell number is `4 * row + column`:

```text
 0  1  2  3
 4  5  6  7
 8  9 10 11
12 13 14 15
```

`to_grid` converts the internal exponent array to a 4×4 array of displayed tile values. `from_grid` performs the inverse conversion. The web API uses displayed tile values for `board` and also provides the exponent grid as `exponents`.

### Actions, sliding, merging, and rewards

The action integers and names are:

```text
0 = up
1 = right
2 = down
3 = left
```

`board.py` defines the four groups of lines and orders each line from the edge tiles move toward. `move(board, action, out)` processes those lines. It skips empty cells, compacts tiles, and combines a pair of equal adjacent exponents once. A merged pair with exponent `e` becomes exponent `e+1` and contributes `2**(e+1)` to the move reward. A newly created tile cannot merge a second time during that same move.

`move` writes the after-move board into a caller-provided output array. It returns the integer merge reward, or `-1` if the board did not change. The negative result is how the rest of the project identifies an illegal move. An illegal move does not cause a spawn.

### Random spawns and game start

`spawn_tile` chooses one empty cell uniformly at random. It puts a tile with value 2 there with probability 0.9 or value 4 with probability 0.1. In the internal exponent representation, these are 1 and 2.

`new_board` starts from an empty array and calls `spawn_tile` twice. The `seed` helper seeds Numba's random generator; it is separate from NumPy's normal Python random generator.

The game is over when all four calls to `move` report that the board cannot change. The maximum tile is stored internally as an exponent; command-line summaries and the UI convert it to the visible value.

## 4. The n-tuple network

### What a tuple is

A tuple is a fixed ordered list of cell indices. For one board and one tuple, the exponents in those cells form a base-16 number. That number is an index into that tuple's weight table.

For a tuple with cell exponents `(e0, e1, ..., eL-1)`, the index is:

```text
index = (((e0 * 16 + e1) * 16 + e2) ... )
```

For example, the four exponents `(1, 2, 3, 4)` form `0x1234`, which is decimal `4660`. Each tuple has a separate table. Cells with exponent greater than 15 are clipped to 15 before indexing, consistently in both evaluation and updates.

### Network layouts and memory

The layouts are defined in `game2048/ntuple.py`:

- **large**: four 6-cell tuples. Each table has `16**6` float32 entries, so the network has `4 * 16**6 = 67,108,864` weights (about 268 MB in memory).
- **small**: five 4-cell tuples. The network has `5 * 16**4 = 327,680` float32 weights (about 1.3 MB in memory).

The patterns overlap. Overlap lets the value function represent several local views of a board. The small network is useful for quick experiments and the web UI's zero-initialized fresh agent. The large network uses more memory and is the stronger provided model.

The actual row-major cell lists are:

```text
large:
  [0, 1, 2, 3, 4, 5]       top row plus the first two cells of row 2
  [4, 5, 6, 7, 8, 9]       second row plus the first two cells of row 3
  [0, 1, 2, 4, 5, 6]       2-by-3 rectangle at the upper left
  [4, 5, 6, 8, 9, 10]      2-by-3 rectangle in rows 2 and 3

small:
  [0, 1, 2, 3]             first row
  [4, 5, 6, 7]             second row
  [0, 1, 4, 5]             2-by-2 square at the upper left
  [1, 2, 5, 6]             2-by-2 square at the upper middle
  [5, 6, 9, 10]            2-by-2 square in the middle
```

### Eight-way symmetry sharing

`symmetric_patterns` generates eight views of every tuple: four rotations, and four rotations after a horizontal reflection. `NTupleNetwork.sym` stores the resulting cell indices.

The value of a board is the sum of the looked-up weights for every tuple and every symmetry:

```text
V(board) = sum over tuples p and symmetries s of weights[p, index(board, p, s)]
```

The same tuple table is therefore used for rotated/reflected versions of its shape. This both reduces the number of separate parameters and means one learning event can touch up to eight symmetric lookup positions per tuple.

`evaluate` performs this sum in Numba-compiled code. `NTupleNetwork.value` is the Python method that calls it.

### Saving and loading weights

`NTupleNetwork.save(path)` writes a compressed `.npz` containing:

- `weights`: the float32 tables.
- `meta`: JSON text containing the network name and tuple patterns.

`NTupleNetwork.load(path)` reads both. The metadata is how the web server infers whether the file contains a `large` or `small` network. The web server loads the trained file once at startup. Learning through the web UI changes in-memory weights only; it does not overwrite the `.npz` file.

## 5. The agent and its learning algorithm

### Afterstates

The normal board given to the agent already includes the most recent random tile spawn. For each legal action, the agent first computes a deterministic slide and merge. That board before the next random spawn is called the **afterstate**. The merge reward is also known at that point.

The agent learns a value for afterstates. It does not need to enumerate possible future random spawns just to choose its immediate action.

### Greedy action selection

`greedy_action(board, sym, weights, buf)` tests actions in this order:

```text
Up, Right, Down, Left
```

For each legal action `a`, it calculates:

```text
action_score(a) = immediate_merge_reward(a) + V(afterstate(a))
```

It returns the action with the greatest score and that action's merge reward. Because it only replaces the current best when it finds a strictly greater score, exact ties are resolved in favor of the first action in the order above. If no legal action exists, it returns `(-1, 0)`.

### TD(0) update timing and formula

The code remembers the previous afterstate, `A(t-1)`. Once it has chosen the next move and has the new afterstate `A(t)`, it can compute the new move reward `r(t)` and update the previous estimate:

```text
delta = r(t) + V(A(t)) - V(A(t-1))
V(A(t-1)) <- V(A(t-1)) + alpha * delta
```

In the implementation, the update is distributed across the eight symmetry lookups. Each lookup receives:

```text
per-lookup step = alpha * delta / 8
```

The default learning rate is `alpha = 0.1`. `ntuple.update` recomputes the same tuple and symmetry indices as evaluation and adds that step to the corresponding weights. If multiple symmetries of the same tuple resolve to one table index, that entry gets the step multiple times; the web inspection groups those repeated hits when presenting a net weight change.

The first move of a game has no previous afterstate, so it has no TD update. When the agent discovers that there are no legal moves, the last afterstate's target is zero:

```text
delta_terminal = 0 - V(A_last)
```

No terminal update is applied if learning is disabled.

As a numeric example, suppose the previous afterstate is valued at 100, the current move earns reward 4, and its afterstate is valued at 120. Then `delta = 4 + 120 - 100 = 24`. At `alpha = 0.1`, each symmetry lookup receives `0.1 * 24 / 8 = 0.3`. The next estimate of the previous afterstate moves upward because the target was greater than its old estimate.

### Full-game training path

`play_game` is the Numba-compiled loop used for training and evaluation:

1. Create a board with two random tiles.
2. Select a greedy legal action.
3. If this is not the first move and learning is enabled, update the previous afterstate using the current reward and afterstate value.
4. Store the chosen afterstate as the new previous afterstate.
5. Add its merge reward to the score.
6. Replace the board with the afterstate and spawn a random tile.
7. Repeat until there is no legal action; apply the terminal update if learning is enabled.

`run_games` repeats that loop for multiple games and returns NumPy arrays of scores, maximum tile exponents, and move counts. `run_random_games` is a baseline that chooses uniformly from the available legal moves rather than using the learned value function.

## 6. Training, evaluation, playback, and plotting

### Training: `train.py`

Example:

```bash
python train.py --network large --games 100000
```

Options include:

- `--network large|small`: choose the layout; default is `large`.
- `--games`: number of self-play games; default is 100,000.
- `--alpha`: learning rate; default is 0.1.
- `--chunk`: games per progress/log row; default is 1,000.
- `--seed`: seed Numba's random generator; default is 0.
- `--out`: output folder; default is `runs/<network>`.
- `--resume`: load a saved `.npz` and continue training.

The script prints network size and progress. It writes a CSV with average score, maximum score, 2048 rate, tile reach rates, average moves, and elapsed time. It saves `weights.npz` periodically (every ten chunks) and at the end. The first chunk can be slower while Numba compiles hot functions.

### Evaluation: `evaluate.py`

Example:

```bash
python evaluate.py --weights runs/large/weights.npz --games 1000
```

Evaluation runs a random baseline and, if a weights file was supplied, the trained agent with learning disabled (`alpha=0`, `learn=False`). It reports average and maximum score, maximum tile, average moves, and the fraction of games reaching several tile values. It writes JSON results, by default to `results.json`.

### Terminal playback: `play.py`

Example:

```bash
python play.py --weights runs/large/weights.npz --delay 0.05
```

This displays one game at a time in the terminal using the trained greedy policy. `--seed` can set the Numba random generator for a reproducible starting sequence, and `--delay` controls the pause between moves.

### Plotting: `plot.py`

Example:

```bash
python plot.py --log runs/large/train_log.csv --results results.json
```

It plots average score and 2048 success rate from the training CSV. If evaluation JSON is supplied, it also plots the tile reach distributions. The learning curve output defaults to `learning_curve.png`; the reach-rate comparison is written to `tile_distribution.png`.

## 7. Local web application

### Starting the server

Install dependencies and start the server from the repository root:

```bash
python web/server.py --weights runs/large/weights.npz --network large
```

Open **http://localhost:5000**. The server binds to `127.0.0.1`, loads weights once at startup, and keeps one in-memory session. The `--network` flag is optional because the network name is read from the weights metadata. The included large weights file is a compressed file on disk but expands to the large network's in-memory table size.

The HTML, CSS, and JavaScript are served from `web/static/`. There is no npm, build step, framework, or external CDN request.

### Three modes

- **Play Yourself**: arrow keys, WASD, touch swipe, and on-screen direction buttons send human moves to the Python engine.
- **RL Agent: Step-by-Step**: Next requests one detailed agent step. Previous undoes one step. The UI displays all four legal/illegal evaluations, afterstates, reward, value and score, then any TD update and weight changes.
- **RL Agent: Autoplay**: a browser timer asks Python for one lightweight step at a time. The speed selector controls the interval from one to twenty moves per second. It pauses on a 2048 overlay or game over.

Both agent modes can use the loaded trained weights or the separate fresh small network initialized to zero. Learning can be switched on or off. Autoplay starts with learning off; Step-by-Step starts with it on. Web learning changes only the server's in-memory network. It does not save those changes to the training `.npz` file.

Mode changes ask before resetting the board. Reset Board asks for confirmation while a game is in progress. Reaching 2048 pauses the run for a choice to continue or start over. Game over displays the final score and maximum tile. The browser stores the best score in `localStorage`.

### Server/API responsibilities

`web/server.py` owns the Flask routes and a single module-level `GameSession`. JSON board values are converted at the API boundary. The main endpoints are:

| Endpoint | Purpose |
|---|---|
| `GET /api/state` | Read the current mode-independent board and statistics |
| `POST /api/reset` | Start a board in a selected mode and weights source |
| `POST /api/human/move` | Send one direction to the Python engine |
| `POST /api/agent/step?detail=1` | Perform a detailed step for the inspector |
| `POST /api/agent/step?detail=0` | Perform a lighter step for autoplay |
| `POST /api/agent/undo` | Undo one inspected agent step |
| `POST /api/continue` | Dismiss the 2048 pause and continue the same board |
| `POST /api/settings` | Change the learning switch without resetting the board |

Requests with invalid directions, modes, or sources return JSON errors with a 400 status.

### How the web instrumentation stays consistent

`web/inspect_agent.py` builds the explanatory record around existing engine calls:

- It asks the project's `greedy_action` function to select the move. This preserves the normal policy and tie ordering.
- It uses `move` for each candidate action and `evaluate` for afterstate values.
- For detailed mode, it follows the network's stored `sym` indices to report each tuple, symmetry, base-16 index, exponent pattern, and weight.
- For learning, it calls the project's `update` function. It records old/new values only for the touched entries.

History stores the move record, spawn position/value, board/session fields, and only the changed weight entries. Undo restores the old values; asking for Next after undo replays the saved entry rather than drawing a new random spawn. Once the history cursor reaches the end, Next generates a new action and spawn.

The front end displays boards from the server response. Its exponent-to-value helper accepts both 16-cell flat arrays (used in decision records) and 4×4 arrays (used in the API's current board state).

## 8. Tests and what they check

Run all tests with:

```bash
python -m pytest tests
```

`tests/test_game.py` checks 2048 merge behavior, rewards, illegal moves, direction handling, valid distinct symmetries, and value symmetry invariance.

`tests/test_web_api.py` checks that instrumentation selects the same action as `greedy_action`, that instrumented updates produce the same weights as `ntuple.update`, that undo restores weights and redo reproduces board/spawn/weights, that zero weights produce tied scores for non-merging legal moves, and that the Flask API returns valid JSON and rejects an invalid direction.

## 9. Dependencies and design choices

- **NumPy** stores board arrays and weight tables.
- **Numba** compiles the core loops for moves, evaluation, updates, and self-play.
- **Flask** serves the local UI and JSON API.
- **Matplotlib** makes experiment plots.
- **Pytest** runs the test suite.

The browser uses ordinary HTML, CSS, and JavaScript. It does not calculate policy values or merge rewards. The project has no external font, script, or stylesheet downloads.

## 10. Suggested reading order

For someone learning the code, this order follows the dependencies:

1. `game2048/board.py`: how a board is stored, moved, rewarded, and spawned.
2. `game2048/ntuple.py`: tuple shapes, symmetry, value lookup, and weight update.
3. `game2048/agent.py`: greedy decisions and the TD(0) game loop.
4. `train.py`, then `evaluate.py`, `play.py`, and `plot.py`: how the core is used.
5. `web/inspect_agent.py`: how one training-style step becomes an inspectable record.
6. `web/server.py`: HTTP endpoints and server startup.
7. `web/static/index.html`, `style.css`, and `app.js`: what the user sees and how browser actions call the API.
8. `tests/`: executable examples of expected game and learning behavior.

## 11. Important boundaries

- The engine board is an exponent array, not a packed 64-bit bitboard.
- A web-session reset clears board/session history and TD summary. It does not reload or zero either network's existing in-memory weights.
- The trained weights file is loaded once at server start. Web learning is not persisted to disk.
- “Fresh agent” means a separate small network whose weights begin at zero; once it learns, those weights remain in memory until the server process ends.
- The server is designed for one local user/session, not multiple isolated users or production hosting.
- The web UI is an interactive teaching and play tool. The training/evaluation scripts remain the batch experiment path.
