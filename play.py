"""Runs a trained agent against TrafficSignalEnv with the Panda3D renderer
live, for visual demonstration / demo-video recording.

Usage:
    uv run python play.py
    uv run python play.py --model models/dqn/dqn_baseline --algorithm dqn
    uv run python play.py --episodes 3 --fps 15
"""

import argparse

from stable_baselines3 import PPO, A2C, DQN

from environment.custom_env import TrafficSignalEnv
from environment.rendering import TrafficRenderer

DEFAULT_MODEL = "models/pg/ppo_baseline"

ALGO_CLASSES = {
    "ppo": PPO,
    "a2c": A2C,
    "reinforce": PPO,  # "reinforce" is trained as a specially-configured PPO; same load class.
    "dqn": DQN,
}


def infer_algorithm(model_path):
    name = model_path.lower()
    for key in ("dqn", "a2c", "reinforce", "ppo"):
        if key in name:
            return key
    return "ppo"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, help="path to a trained SB3 model (without .zip)")
    parser.add_argument(
        "--algorithm",
        choices=sorted(ALGO_CLASSES),
        default=None,
        help="SB3 algorithm class to load --model with (default: inferred from the model path)",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    algorithm = args.algorithm or infer_algorithm(args.model)
    model_cls = ALGO_CLASSES[algorithm]
    model = model_cls.load(args.model)
    print(f"Loaded {algorithm.upper()} model from {args.model}")

    def policy(obs):
        action, _states = model.predict(obs, deterministic=True)
        return int(action)

    env = TrafficSignalEnv()
    renderer = TrafficRenderer(headless=False)

    for episode in range(args.episodes):
        steps = renderer.run_episode(env, policy=policy, max_steps=env.max_steps, fps=args.fps)
        print(f"Episode {episode + 1}/{args.episodes}: {steps} steps")


if __name__ == "__main__":
    main()
