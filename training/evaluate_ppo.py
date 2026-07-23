import json

from stable_baselines3 import PPO
from environment.custom_env import TrafficSignalEnv

env = TrafficSignalEnv()
model = PPO.load("models/pg/ppo_baseline")

num_episodes = 50
episode_lengths = []
episode_rewards = []
gridlock_count = 0

for episode in range(num_episodes):
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0

    while True:
        action, _states = model.predict(obs, deterministic=True)
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if terminated:
            gridlock_count += 1
            break
        if truncated:
            break

    episode_lengths.append(steps)
    episode_rewards.append(total_reward)

    print(f"Episode {episode + 1}: steps={steps}, total_reward={total_reward:.1f}")

print()
print(f"Gridlock rate: {gridlock_count}/{num_episodes} episodes")
print(f"Average episode length: {sum(episode_lengths) / num_episodes:.1f} steps")
print(f"Average episode reward: {sum(episode_rewards) / num_episodes:.1f}")

results = {
    "model": "ppo_baseline",
    "num_episodes": num_episodes,
    "gridlock_count": gridlock_count,
    "gridlock_rate": gridlock_count / num_episodes,
    "avg_episode_length": sum(episode_lengths) / num_episodes,
    "avg_episode_reward": sum(episode_rewards) / num_episodes,
    "episode_lengths": episode_lengths,
    "episode_rewards": episode_rewards,
}

with open("logs/ppo_baseline_eval.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to logs/ppo_baseline_eval.json")