# Adaptive Traffic Signal Control (RL Summative)

A reinforcement learning agent that controls the traffic signals at a single
Lagos-style four-way intersection, trained with Stable-Baselines3 against a
custom Gymnasium environment, with a live Panda3D visualization.

## Environment

`environment/custom_env.py` defines `TrafficSignalEnv`, a custom
`gymnasium.Env` for one intersection with four approaches (N, S, E, W).

- **Action space** — `Discrete(5)`, one signal phase per step:
  - `0` NS-through, `1` EW-through, `2` NS-left, `3` EW-left, `4` all-red
- **Observation space** — `Box(10,)`: queue length and average wait time per
  direction (N, S, E, W), plus the current phase and time spent in it.
- **Reward** — penalizes total queue length and total wait time, penalizes
  switching phases (and switching to all-red specifically), and rewards
  fully clearing a direction's queue.
- **Episode end** — truncates at 500 steps, or terminates early if any
  direction's queue exceeds the gridlock threshold (20 vehicles).

## Project structure

```
environment/     TrafficSignalEnv (custom_env.py) and the Panda3D
                  visualization (rendering.py)
training/         training scripts, one per algorithm family
    pg_training.py    PPO / A2C / REINFORCE (--algorithm flag)
    dqn_training.py   DQN
    evaluate_ppo.py   evaluates a saved PPO model over 50 episodes
models/pg/        saved policy-gradient models (ppo_baseline, a2c_baseline,
                  reinforce_baseline)
models/dqn/       saved DQN models
tests/            test_env.py (check_env + random-policy baseline),
                  test_rendering.py (renderer smoke tests / demo driver)
logs/             tensorboard logs and evaluation JSON results
play.py           loads a trained model and runs it live in the Panda3D
                  renderer
```

## Setup

```
uv sync
```

## Training

```
uv run python -m training.pg_training --algorithm ppo
uv run python -m training.pg_training --algorithm a2c
uv run python -m training.pg_training --algorithm reinforce
uv run python -m training.dqn_training
```

Each saves to `models/pg/<algorithm>_baseline` (or `models/dqn/dqn_baseline`
for DQN) and logs to `logs/<algorithm>_baseline` for TensorBoard.

REINFORCE has no native Stable-Baselines3 implementation, so
`pg_training.py` approximates it as a PPO agent with its update collapsed to
a single unclipped policy-gradient step per rollout (`n_epochs=1`,
`batch_size == n_steps`, `clip_range=10.0`, `gae_lambda=1.0`) — the reasoning
is documented inline in `training/pg_training.py`.

## Evaluation

```
uv run python -m training.evaluate_ppo
```

Runs the saved PPO baseline for 50 episodes and reports gridlock rate,
average episode length, and average reward. Current results:

| Policy            | Gridlock rate | Avg. episode length | Avg. reward |
|-------------------|--------------:|---------------------:|------------:|
| Random             | 96% (48/50)   | 265.6 steps          | -14,154.4   |
| PPO baseline       | 0% (0/50)     | 500.0 steps          | -876.7      |

## Visualization / demo

```
uv run python play.py                                   # PPO baseline, live window
uv run python play.py --model models/dqn/dqn_baseline --algorithm dqn
uv run python -m tests.test_rendering                     # headless renderer smoke test
uv run python -m tests.test_rendering --live               # live window, random policy
uv run python -m tests.test_rendering --capture             # dumps per-step PNGs for a demo video
```

The renderer (`environment/rendering.py`) shows the intersection from an
angled aerial view: four traffic signals that change color with the active
phase, cars that spawn/despawn per direction to match live queue length, and
a surrounding street scene (buildings, sidewalks, streetlights, lane
markings, crosswalks).

## Tests

```
uv run python -m tests.test_env          # check_env + random-policy baseline
uv run python -m tests.test_rendering    # renderer smoke test
```
