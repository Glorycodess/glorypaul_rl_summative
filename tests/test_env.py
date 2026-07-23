import json

from stable_baselines3.common.env_checker import check_env
from environment.custom_env import TrafficSignalEnv

env = TrafficSignalEnv()
check_env(env)
print("Environment passed check_env validation")
print()

num_episodes = 50
episode_lengths = []
episode_rewards = []
gridlock_count = 0

for episode in range(num_episodes):
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    ended_by_gridlock = False

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if terminated:
            ended_by_gridlock = True
            gridlock_count += 1
            break
        if truncated:
            break

    episode_lengths.append(steps)
    episode_rewards.append(total_reward)

    print(f"Episode {episode + 1}: steps={steps}, total_reward={total_reward:.1f}, "
          f"ended_by_gridlock={ended_by_gridlock}")

print()
print(f"Gridlock rate: {gridlock_count}/{num_episodes} episodes")
print(f"Average episode length: {sum(episode_lengths) / num_episodes:.1f} steps")
print(f"Average episode reward: {sum(episode_rewards) / num_episodes:.1f}")

baseline_results = {
    "arrival_prob": env.arrival_prob,
    "gridlock_threshold": env.gridlock_threshold,
    "num_episodes": num_episodes,
    "gridlock_count": gridlock_count,
    "gridlock_rate": gridlock_count / num_episodes,
    "avg_episode_length": sum(episode_lengths) / num_episodes,
    "avg_episode_reward": sum(episode_rewards) / num_episodes,
    "episode_lengths": episode_lengths,
    "episode_rewards": episode_rewards,
}

with open("logs/random_baseline.json", "w") as f:
    json.dump(baseline_results, f, indent=2)

print("\nBaseline results saved to logs/random_baseline.json")