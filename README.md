# Learning to Play 2048 with TD(0) Afterstate Learning and N-Tuple Networks

An autonomous 2048 agent that learns entirely through self-play, with no hand-crafted
evaluation heuristics. The value of afterstates is learned with TD(0) and approximated
by an n-tuple network.

## Project structure

```
game2048/
  board.py     game engine: slides, merges, rewards, tile spawning (Numba-compiled)
  ntuple.py    n-tuple network with 8-way symmetric weight sharing
  agent.py     greedy afterstate policy, online TD(0) training, random baseline, metrics
train.py       train the agent and log learning curves to CSV
evaluate.py    evaluate a trained agent vs. the random baseline (learning off)
plot.py        learning-curve and tile-distribution figures
play.py        watch the trained agent play in the terminal
tests/         unit tests for game rules and network symmetry
```

## Setup

```bash
pip install -r requirements.txt
python -m pytest tests          # 11 tests should pass
```

## Running the experiments

```bash
# 1. Train (large = four 6-tuples, ~268 MB RAM; small = five 4-tuples, ~1 MB)
python train.py --network large --games 100000

# 2. Evaluate over 1000 games against the random baseline
python evaluate.py --weights runs/large/weights.npz --games 1000

# 3. Figures for the report
python plot.py --log runs/large/train_log.csv --results results.json

# 4. Watch it play
python play.py --weights runs/large/weights.npz
```

Training can be stopped and continued with `--resume runs/large/weights.npz`.
Weights are saved every 10 chunks and at the end.

## Web UI

Install the requirements, then start the local Flask server:

```bash
python web/server.py --weights runs/large/weights.npz --network large
```

Open http://localhost:5000. The server binds to localhost and loads the weights once at startup. The network type is read from the `.npz` metadata, so `--network` is optional.

The UI has three modes. **Play Yourself** sends human moves to the Python engine. **RL Agent: Step-by-Step** displays each greedy move evaluation and the TD(0) update, with undo and deterministic redo. **RL Agent: Autoplay** runs the same policy on a timer. Both agent modes can use the loaded trained weights or a separate zero-initialized small network, and can apply or skip learning.

## Method

**MDP.** The state is the board, the actions are up/right/down/left, and the reward is the
sum of tiles created by merges in the move. After each move a 2 (p = 0.9) or 4 (p = 0.1)
spawns on a random empty cell.

**Afterstates.** A move is split into a deterministic slide, which produces the afterstate s′,
and a random spawn. The agent learns V(s′), so choosing an action needs no expectation
over spawns:

    a_t = argmax_a [ r(s_t, a) + V(s′(s_t, a)) ]

**TD(0) update** (online, after every move):

    V(s′_{t−1}) ← V(s′_{t−1}) + α [ r_t + V(s′_t) − V(s′_{t−1}) ]

When no legal move remains, the target is 0. The policy is purely greedy; random tile spawns
provide enough exploration.

**N-tuple network.** Each tuple is a fixed group of cells whose tile exponents form a base-16
index into a weight table. V(s′) is the sum of the indexed weights over all tuples and all 8
board symmetries, so each table is shared across rotations and reflections. The step size
α is split evenly across the 8 symmetric features (default α = 0.1).

| Network | Tuples | Weights | Memory |
|---|---|---|---|
| large | four 6-tuples (two 2×3 rectangles, two row-plus-two shapes) | 67M | ~268 MB |
| small | five 4-tuples (two rows, three 2×2 squares) | 330K | ~1.3 MB |

## Sanity-check results (from development runs, your numbers will vary)

| Agent | Training games | Avg score | 2048 rate |
|---|---|---|---|
| Random baseline | – | ~1,000 | 0% (max tile 256) |
| Small network | 20,000 | ~18,000 | ~29% |
| Large network | 30,000 | ~53,000 | ~88% (training games) |

Training speed is roughly 5,000–10,000 games per minute early on, slowing as games get longer.
Run the large network for 100k+ games and report your own final evaluation numbers.

## Suggested report outline

1. Introduction: 2048 as an RL benchmark
2. MDP formulation and the afterstate decomposition
3. TD(0) and n-tuple function approximation, including symmetry sharing
4. Experimental setup: networks, α, number of training games, evaluation protocol, seeds
5. Results: learning curves, tile-reach rates, comparison with the random baseline,
   and small vs. large network
6. Discussion: why afterstates help, limitations of greedy 1-ply play
7. Conclusion and future work

## Possible extensions

- Ablations: α values, symmetry sharing on vs. off, small vs. large network
- Expectimax search (2–3 ply) on top of the learned value function at evaluation time
- TD(λ), temporal coherence learning, or multi-stage networks
- Optimistic weight initialization
