"""Hyperparameter sweep for DQN / PPO / A2C / REINFORCE on TrafficSignalEnv.

Usage:
    uv run python -m training.run_sweep              # full 40-run sweep
    uv run python -m training.run_sweep --dry-run     # 2-run pipeline check
"""

import argparse
import json
import os
import traceback

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.monitor import Monitor

from environment.custom_env import TrafficSignalEnv

TIMESTEPS = 50_000
EVAL_EPISODES = 20

LEARNING_RATES = [0.0001, 0.0003, 0.0005, 0.001, 0.005]

GRIDS = {
    "dqn": [{"lr": lr, "gamma": g} for g in (0.95, 0.99) for lr in LEARNING_RATES],
    "ppo": [{"lr": lr, "clip": c} for c in (0.1, 0.3) for lr in LEARNING_RATES],
    "a2c": [{"lr": lr, "n_steps": n} for n in (5, 20) for lr in LEARNING_RATES],
    "reinforce": [{"lr": lr, "gamma": g} for g in (0.95, 0.99) for lr in LEARNING_RATES],
}

BEST_MODEL_PATHS = {
    "dqn": "models/dqn/dqn_best",
    "ppo": "models/pg/ppo_best",
    "a2c": "models/pg/a2c_best",
    "reinforce": "models/pg/reinforce_best",
}

RESULTS_PATH = "logs/sweep_results.json"


def build_model(algorithm, env, params):
    if algorithm == "dqn":
        return DQN("MlpPolicy", env, learning_rate=params["lr"], gamma=params["gamma"], verbose=0)

    if algorithm == "ppo":
        return PPO("MlpPolicy", env, learning_rate=params["lr"], clip_range=params["clip"], verbose=0)

    if algorithm == "a2c":
        return A2C("MlpPolicy", env, learning_rate=params["lr"], n_steps=params["n_steps"], verbose=0)

    if algorithm == "reinforce":
        return PPO(
            "MlpPolicy",
            env,
            learning_rate=params["lr"],
            gamma=params["gamma"],
            n_steps=2048,
            batch_size=2048,
            n_epochs=1,
            gae_lambda=1.0,
            clip_range=10.0,
            verbose=0,
        )

    raise ValueError(f"Unknown algorithm: {algorithm}")


def evaluate(model, num_episodes=EVAL_EPISODES):
    env = TrafficSignalEnv()
    episode_lengths = []
    episode_rewards = []
    gridlock_count = 0

    for _ in range(num_episodes):
        obs, _info = env.reset()
        total_reward = 0.0
        steps = 0
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(int(action))
            total_reward += reward
            steps += 1
            if terminated:
                gridlock_count += 1
                break
            if truncated:
                break
        episode_lengths.append(steps)
        episode_rewards.append(total_reward)

    return {
        "avg_eval_reward": sum(episode_rewards) / num_episodes,
        "avg_eval_length": sum(episode_lengths) / num_episodes,
        "gridlock_rate": gridlock_count / num_episodes,
    }


def format_params(params):
    return " ".join(f"{k}={v}" for k, v in params.items())


def is_better(candidate, current_best):
    if current_best is None:
        return True
    if candidate["avg_eval_reward"] != current_best["avg_eval_reward"]:
        return candidate["avg_eval_reward"] > current_best["avg_eval_reward"]
    return candidate["gridlock_rate"] < current_best["gridlock_rate"]


def run_sweep(grids, results_path, model_paths, timesteps=TIMESTEPS):
    total_runs = sum(len(grid) for grid in grids.values())
    results = []
    global_run = 0

    for algorithm, grid in grids.items():
        best_record = None
        best_model = None

        for algo_run, params in enumerate(grid, start=1):
            global_run += 1
            print(f"Run {global_run}/{total_runs}: {algorithm.upper()} {format_params(params)}...", flush=True)

            record = {
                "global_run": global_run,
                "algorithm": algorithm,
                "algo_run": algo_run,
                "params": params,
                "timesteps": timesteps,
                "eval_episodes": EVAL_EPISODES,
                "status": "ok",
            }

            try:
                env = Monitor(TrafficSignalEnv())
                model = build_model(algorithm, env, params)
                model.learn(total_timesteps=timesteps)
                eval_stats = evaluate(model)
                record.update(eval_stats)
                print(
                    f"  -> eval reward: {eval_stats['avg_eval_reward']:.1f}, "
                    f"avg length: {eval_stats['avg_eval_length']:.1f}, "
                    f"gridlock: {eval_stats['gridlock_rate'] * 100:.0f}%"
                )
                if is_better(record, best_record):
                    best_record = record
                    best_model = model
            except Exception as exc:
                record["status"] = "failed"
                record["error"] = str(exc)
                print(f"  -> FAILED: {exc}")
                traceback.print_exc()

            results.append(record)
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

        if best_model is not None:
            save_path = model_paths[algorithm]
            best_model.save(save_path)
            print(
                f"Best {algorithm.upper()} config: {format_params(best_record['params'])} "
                f"(eval reward {best_record['avg_eval_reward']:.1f}) -> saved to {save_path}"
            )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run only DQN run 1 and PPO run 1, writing to throwaway paths, to sanity-check the pipeline",
    )
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)

    if args.dry_run:
        grids = {"dqn": GRIDS["dqn"][:1], "ppo": GRIDS["ppo"][:1]}
        results_path = "logs/sweep_results_dryrun.json"
        model_paths = {
            "dqn": "models/dqn/_dryrun_dqn_best",
            "ppo": "models/pg/_dryrun_ppo_best",
        }
        print("DRY RUN: 2 configs only, writing to throwaway paths.\n")
    else:
        grids = GRIDS
        results_path = RESULTS_PATH
        model_paths = BEST_MODEL_PATHS

    run_sweep(grids, results_path, model_paths)
    print(f"\nAll runs complete. Results written to {results_path}")


if __name__ == "__main__":
    main()
