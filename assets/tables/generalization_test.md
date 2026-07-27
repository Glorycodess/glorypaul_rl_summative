# Generalization Test (30 episodes, randomized start states)

| Algorithm | Best Hyperparameters | Empty-Start Reward | Empty-Start Gridlock | Randomized-Start Reward | Randomized-Start Gridlock |
|---|---|---|---|---|---|
| DQN | lr=0.001, gamma=0.95 | -740.5 | 0% | -924.6 | 0% |
| PPO | lr=0.0003, clip=0.3 | -768.0 | 0% | -986.0 | 0% |
| A2C | lr=0.005, n_steps=5 | -886.4 | 0% | -1073.2 | 0% |
| REINFORCE | lr=0.005, gamma=0.95 | -1147.4 | 0% | -1345.6 | 0% |
