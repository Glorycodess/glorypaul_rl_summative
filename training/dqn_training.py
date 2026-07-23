"""Trains a DQN agent on TrafficSignalEnv.

Usage:
    uv run python -m training.dqn_training
"""

import argparse

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from environment.custom_env import TrafficSignalEnv

MODEL_PATH = "models/dqn/dqn_baseline"
TENSORBOARD_LOG = "logs/dqn_baseline"


def build_model(env):
    return DQN("MlpPolicy", env, verbose=1, tensorboard_log=TENSORBOARD_LOG)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50_000)
    args = parser.parse_args()

    env = Monitor(TrafficSignalEnv())
    model = build_model(env)
    model.learn(total_timesteps=args.timesteps)

    model.save(MODEL_PATH)
    print(f"Training complete. Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
