# Summary Comparison

| Method | Key Hyperparameters | Avg Reward | Avg Episode Length | Gridlock Rate |
|---|---|---|---|---|
| Random policy | n/a | -14154.4 | 265.6 | 96% |
| DQN (best) | lr=0.001, gamma=0.95 | -740.5 | 500.0 | 0% |
| PPO (best) | lr=0.0003, clip=0.3 | -768.0 | 500.0 | 0% |
| A2C (best) | lr=0.005, n_steps=5 | -886.4 | 500.0 | 0% |
| REINFORCE (best) | lr=0.005, gamma=0.95 | -1147.4 | 500.0 | 0% |
