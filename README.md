# Adaptive Traffic Signal Control (RL Summative)

A reinforcement learning agent that learns to control the traffic signals at
a single Lagos-style four-way intersection. Traffic (queue lengths and wait
times) is simulated by a custom Gymnasium environment; four algorithms
(DQN, PPO, A2C, and a REINFORCE-style policy gradient) are trained against
it with Stable-Baselines3, swept over hyperparameters, and compared against
a random-policy baseline. A Panda3D renderer provides a live visual
demonstration of a trained agent operating the intersection.

The mission framing: Lagos intersections are notorious for exactly the
failure mode this project targets — fixed-time or poorly-tuned signals that
let one direction's queue grow unbounded while a perpendicular approach sits
on green with nothing arriving. The environment models that failure mode
directly (a "gridlock" termination condition) so the agent is explicitly
rewarded for avoiding it.

## Environment

`environment/custom_env.py` defines `TrafficSignalEnv`, a custom
`gymnasium.Env` for one intersection with four approaches (N, S, E, W).

- **Action space** — `Discrete(5)`, one signal phase per step:
  - `0` NS-through, `1` EW-through, `2` NS-left, `3` EW-left, `4` all-red
- **Observation space** — `Box(10,)`: queue length per direction (N, S, E,
  W), the current phase and time spent in it, and average wait time per
  direction.
- **Reward** — negative total queue length and negative total wait time
  (so idle queues are actively penalized every step), a penalty for
  switching phases (and a larger additional penalty specifically for
  switching to all-red), and a bonus for fully clearing a direction's
  queue.
- **Episode end** — truncates at 500 steps, or terminates early
  ("gridlock") if any direction's queue exceeds 20 vehicles.

`environment/rendering.py` provides `TrafficRenderer`, a Panda3D scene
(angled aerial camera, buildings/streetlights/sidewalks/crosswalks for
scene context, four traffic signals, and per-direction vehicle queues) kept
in sync with the environment's observations — used by `play.py` and
`tests/test_rendering.py`, entirely independent of the training code.

## Setup

Prerequisites: [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
installed, and Python 3.11+ (uv will fetch a matching interpreter
automatically if you don't have one).

```
git clone https://github.com/Glorycodess/glorypaul_rl_summative.git
cd glorypaul_rl_summative
uv sync
uv run main.py
```

`git clone` downloads the repository; `cd` into it before running anything
else below, since every command assumes it's run from the project root.
`uv sync` installs the locked dependency set (Gymnasium, Stable-Baselines3,
Panda3D, PyTorch, TensorBoard, Matplotlib) into `.venv`. `uv run main.py` is
a smoke test that just prints a hello-world line — use it to confirm the
environment is installed and runnable before moving on to training.

## Training

```
uv run python -m training.pg_training --algorithm ppo
uv run python -m training.pg_training --algorithm a2c
uv run python -m training.pg_training --algorithm reinforce
uv run python -m training.dqn_training
```

Both accept `--timesteps` (default `50000`). `pg_training.py`'s
`--algorithm` flag selects which Stable-Baselines3 model class to build;
`dqn_training.py` always trains DQN. Each run saves to
`models/pg/<algorithm>_baseline.zip` (or `models/dqn/dqn_baseline.zip` for
DQN) and logs to `logs/<algorithm>_baseline` for TensorBoard.

REINFORCE has no native Stable-Baselines3 implementation, so
`pg_training.py` approximates it as a PPO agent with its update collapsed
to a single unclipped policy-gradient step per rollout (`n_epochs=1`,
`batch_size == n_steps`, `clip_range=10.0`, `gae_lambda=1.0`) — the
reasoning is documented inline in `training/pg_training.py`.

### Hyperparameter sweep

```
uv run python -m training.run_sweep              # full 40-run sweep
uv run python -m training.run_sweep --dry-run     # 2-run pipeline check
```

Sweeps all four algorithms, 10 configurations each (40 runs total), 50,000
timesteps and a 20-episode evaluation per run:

| Algorithm | Grid | Values |
|---|---|---|
| DQN | learning rate × gamma | lr ∈ {0.0001, 0.0003, 0.0005, 0.001, 0.005} × gamma ∈ {0.95, 0.99} |
| PPO | learning rate × clip range | lr ∈ {0.0001, 0.0003, 0.0005, 0.001, 0.005} × clip ∈ {0.1, 0.3} |
| A2C | learning rate × n_steps | lr ∈ {0.0001, 0.0003, 0.0005, 0.001, 0.005} × n_steps ∈ {5, 20} |
| REINFORCE | learning rate × gamma | lr ∈ {0.0001, 0.0003, 0.0005, 0.001, 0.005} × gamma ∈ {0.95, 0.99} |

The best config per algorithm (by eval reward, gridlock rate as tiebreaker)
is saved to `models/<pg|dqn>/<algorithm>_best.zip`. Full per-run results
are written incrementally to `logs/sweep_results.json`; per-algorithm
hyperparameter tables and comparison plots built from that file live in
`assets/tables/` and `assets/plots/` (see below).

### Generating report assets

```
uv run python -m training.generate_report_assets
```

Reads `logs/sweep_results.json` and the saved `*_best` models, then writes:
- `assets/tables/<algorithm>_hyperparameter_table.{md,csv}` — every sweep
  run for that algorithm
- `assets/tables/summary_comparison.md` — best config per algorithm vs. the
  random baseline
- `assets/tables/generalization_test.md` — each best model re-evaluated
  from both an empty-queue start and a randomized-queue start
- `assets/plots/01_reward_curves_grid.png` … `05_generalization_test.png` —
  training reward curves, DQN loss, PPO/A2C/REINFORCE entropy, a
  cross-algorithm convergence comparison, and the generalization test

### Evaluating a model

```
uv run python -m training.evaluate_ppo
```

Runs the saved PPO baseline (`models/pg/ppo_baseline.zip`) for 50 episodes
with a deterministic policy and reports gridlock rate, average episode
length, and average reward, writing the full per-episode breakdown to
`logs/ppo_baseline_eval.json`.

## Watching a trained agent live

```
uv run python play.py                                     # PPO baseline, live window, ~10 fps
uv run python play.py --model models/dqn/dqn_best --algorithm dqn
uv run python play.py --episodes 3 --fps 15
```

Defaults to `models/pg/ppo_baseline` inferred as a PPO model; pass
`--model` to point at any saved `.zip` (baseline or `*_best`) and
`--algorithm {ppo,a2c,reinforce,dqn}` if it can't be inferred from the
path. `--episodes` (default `1`) sets how many full episodes to play;
`--fps` (default `10`) sets the playback speed of the live Panda3D window.

## Tests

```
uv run python -m tests.test_env                # check_env + 50-episode random-policy baseline
uv run python -m tests.test_rendering          # headless renderer smoke test
uv run python -m tests.test_rendering --live   # live window, random policy
uv run python -m tests.test_rendering --capture  # headless, dumps a PNG per step
```

`test_env.py` validates the environment against Gymnasium's `check_env`
and writes a fresh random-policy baseline to `logs/random_baseline.json`.
`test_rendering.py` exercises the renderer's camera framing, signal/vehicle
sync, and a short synced episode without needing a live window.

## Results

Best saved model per algorithm vs. the random-policy baseline (50
episodes each; see `logs/random_baseline_official.json` and
`assets/tables/summary_comparison.md`):

| Method | Key hyperparameters | Avg. reward | Avg. episode length | Gridlock rate |
|---|---|---:|---:|---:|
| Random policy | n/a | -14,154.4 | 265.6 | 96% |
| DQN (best) | lr=0.001, gamma=0.95 | **-740.5** | 500.0 | 0% |
| PPO (best) | lr=0.0003, clip=0.3 | -768.0 | 500.0 | 0% |
| A2C (best) | lr=0.005, n_steps=5 | -886.4 | 500.0 | 0% |
| REINFORCE (best) | lr=0.005, gamma=0.95 | -1147.4 | 500.0 | 0% |

DQN has the best overall reward, with PPO close behind; all four trained
algorithms eliminate gridlock entirely (0% vs. the random policy's 96%) and
run the full 500-step episode. `assets/tables/generalization_test.md`
re-evaluates each best model from a randomized (non-empty) starting queue
state and shows the same ranking holds up, at a uniformly higher wait cost.

`models/pg/ppo_best_v2.zip` is a separate experimental PPO model trained
with heavier phase-switch penalties (see `logs/reward_reweight_comparison.json`
for the before/after comparison); it is not part of the sweep and isn't
included in the table above.

## Project structure

```
glorypaul_rl_summative/
├── pyproject.toml               dependency + project metadata (uv)
├── uv.lock                      locked dependency versions
├── README.md
├── main.py                      installation smoke test
├── play.py                      loads a trained model, plays it live in the renderer
├── environment/
│   ├── __init__.py
│   ├── custom_env.py            TrafficSignalEnv (Gymnasium environment)
│   └── rendering.py             TrafficRenderer (Panda3D visualization)
├── training/
│   ├── __init__.py
│   ├── dqn_training.py          trains DQN
│   ├── pg_training.py           trains PPO / A2C / REINFORCE (--algorithm)
│   ├── run_sweep.py             40-run hyperparameter sweep, saves *_best models
│   ├── evaluate_ppo.py          50-episode evaluation of the saved PPO baseline
│   └── generate_report_assets.py   builds assets/tables and assets/plots
├── tests/
│   ├── __init__.py
│   ├── test_env.py              check_env + random-policy baseline
│   └── test_rendering.py        renderer smoke tests / demo driver
├── models/
│   ├── dqn/                     dqn_baseline.zip, dqn_best.zip
│   └── pg/                      ppo/a2c/reinforce baseline + best (+ ppo_best_v2 experiment)
├── assets/
│   ├── plots/                   5 PNG figures used in the report
│   └── tables/                  per-algorithm hyperparameter tables, summary + generalization tables
└── logs/                        generated, regenerable: sweep results, eval JSON,
                                   TensorBoard event dirs, monitor/progress CSVs,
                                   and renderer screenshots/frame captures
```
