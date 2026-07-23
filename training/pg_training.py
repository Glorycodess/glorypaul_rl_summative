"""Trains a policy-gradient agent (PPO, A2C, or REINFORCE) on TrafficSignalEnv.

Usage:
    uv run python -m training.pg_training --algorithm ppo
    uv run python -m training.pg_training --algorithm a2c
    uv run python -m training.pg_training --algorithm reinforce
"""

import argparse

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor

from environment.custom_env import TrafficSignalEnv

ALGORITHMS = ["ppo", "a2c", "reinforce"]

MODEL_PATHS = {
    "ppo": "models/pg/ppo_baseline",
    "a2c": "models/pg/a2c_baseline",
    "reinforce": "models/pg/reinforce_baseline",
}

TENSORBOARD_LOGS = {
    "ppo": "logs/ppo_baseline",
    "a2c": "logs/a2c_baseline",
    "reinforce": "logs/reinforce_baseline",
}


def build_model(algorithm, env):
    if algorithm == "ppo":
        return PPO("MlpPolicy", env, verbose=1, tensorboard_log=TENSORBOARD_LOGS["ppo"])

    if algorithm == "a2c":
        return A2C("MlpPolicy", env, verbose=1, tensorboard_log=TENSORBOARD_LOGS["a2c"])

    if algorithm == "reinforce":
        # SB3 has no native REINFORCE, so this approximates vanilla Monte-Carlo
        # policy gradient (REINFORCE with a value-function baseline) by
        # collapsing PPO's update to a single unclipped policy-gradient step:
        #   - batch_size == n_steps, n_epochs=1: one gradient step per rollout,
        #     computed while the policy still matches the one that generated
        #     the data, so the PPO probability ratio is ~1 by construction and
        #     the clipped surrogate objective reduces to plain REINFORCE.
        #   - clip_range=10.0: set high enough to never bind, since clipping
        #     is exactly what distinguishes PPO's update from REINFORCE's.
        #   - gae_lambda=1.0: the advantage reduces to the full Monte-Carlo
        #     return minus baseline, instead of PPO's default bootstrapped GAE.
        return PPO(
            "MlpPolicy",
            env,
            n_steps=2048,
            batch_size=2048,
            n_epochs=1,
            gae_lambda=1.0,
            clip_range=10.0,
            verbose=1,
            tensorboard_log=TENSORBOARD_LOGS["reinforce"],
        )

    raise ValueError(f"Unknown algorithm: {algorithm}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="ppo")
    parser.add_argument("--timesteps", type=int, default=50_000)
    args = parser.parse_args()

    env = Monitor(TrafficSignalEnv())
    model = build_model(args.algorithm, env)
    model.learn(total_timesteps=args.timesteps)

    save_path = MODEL_PATHS[args.algorithm]
    model.save(save_path)
    print(f"Training complete. Model saved to {save_path}")


if __name__ == "__main__":
    main()
