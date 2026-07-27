"""Generates Phase 6 report assets (hyperparameter tables + plots) from the
Phase 5 sweep results (logs/sweep_results.json) and the saved *_best models.

Usage:
    uv run python -m training.generate_report_assets
"""

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, X_TIMESTEPS

from environment.custom_env import TrafficSignalEnv
from training.run_sweep import build_model, is_better, TIMESTEPS

SWEEP_RESULTS_PATH = "logs/sweep_results.json"
RANDOM_BASELINE_PATH = "logs/random_baseline_official.json"

PLOTS_DIR = "assets/plots"
TABLES_DIR = "assets/tables"
REPORT_RUNS_DIR = "logs/report_runs"

ALGORITHMS = ("dqn", "ppo", "a2c", "reinforce")

BEST_MODEL_PATHS = {
    "dqn": "models/dqn/dqn_best",
    "ppo": "models/pg/ppo_best",
    "a2c": "models/pg/a2c_best",
    "reinforce": "models/pg/reinforce_best",
}

MODEL_CLASSES = {"dqn": DQN, "ppo": PPO, "a2c": A2C, "reinforce": PPO}

HYPERPARAM_COLUMNS = {
    "dqn": [("lr", "Learning Rate"), ("gamma", "Gamma")],
    "ppo": [("lr", "Learning Rate"), ("clip", "Clip Range")],
    "a2c": [("lr", "Learning Rate"), ("n_steps", "N Steps")],
    "reinforce": [("lr", "Learning Rate"), ("gamma", "Gamma")],
}

PLOT_COLORS = {"dqn": "tab:blue", "ppo": "tab:green", "a2c": "tab:orange", "reinforce": "tab:purple"}

GENERALIZATION_EPISODES = 30
MAX_PLOTTED_POINTS = 2000


def fmt_pct(x):
    return f"{x * 100:.0f}%"


def fmt_params(params):
    return ", ".join(f"{k}={v}" for k, v in params.items())


def load_sweep_results():
    with open(SWEEP_RESULTS_PATH) as f:
        return json.load(f)


def best_per_algorithm(results):
    best = {}
    for r in results:
        if r["status"] != "ok":
            continue
        algo = r["algorithm"]
        if algo not in best or is_better(r, best[algo]):
            best[algo] = r
    return best


def write_hyperparameter_table(algorithm, results):
    rows = sorted(
        (r for r in results if r["algorithm"] == algorithm and r["status"] == "ok"),
        key=lambda r: r["algo_run"],
    )
    columns = HYPERPARAM_COLUMNS[algorithm]
    headers = ["Run"] + [label for _, label in columns] + ["Avg Eval Reward", "Avg Episode Length", "Gridlock Rate"]

    csv_rows = [headers]
    for r in rows:
        values = [str(r["algo_run"])]
        values += [str(r["params"][key]) for key, _ in columns]
        values += [f"{r['avg_eval_reward']:.1f}", f"{r['avg_eval_length']:.1f}", fmt_pct(r["gridlock_rate"])]
        csv_rows.append(values)

    md_lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for values in csv_rows[1:]:
        md_lines.append("| " + " | ".join(values) + " |")

    md_path = os.path.join(TABLES_DIR, f"{algorithm}_hyperparameter_table.md")
    csv_path = os.path.join(TABLES_DIR, f"{algorithm}_hyperparameter_table.csv")

    with open(md_path, "w") as f:
        f.write(f"# {algorithm.upper()} Hyperparameter Sweep\n\n")
        f.write("\n".join(md_lines) + "\n")

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)

    return md_path, csv_path


def load_random_baseline():
    with open(RANDOM_BASELINE_PATH) as f:
        return json.load(f)


def write_summary_comparison(best):
    baseline = load_random_baseline()
    headers = ["Method", "Key Hyperparameters", "Avg Reward", "Avg Episode Length", "Gridlock Rate"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]

    lines.append(
        "| Random policy | n/a | "
        f"{baseline['avg_episode_reward']:.1f} | {baseline['avg_episode_length']:.1f} | "
        f"{fmt_pct(baseline['gridlock_rate'])} |"
    )
    for algo in ALGORITHMS:
        r = best[algo]
        lines.append(
            f"| {algo.upper()} (best) | {fmt_params(r['params'])} | "
            f"{r['avg_eval_reward']:.1f} | {r['avg_eval_length']:.1f} | "
            f"{fmt_pct(r['gridlock_rate'])} |"
        )

    path = os.path.join(TABLES_DIR, "summary_comparison.md")
    with open(path, "w") as f:
        f.write("# Summary Comparison\n\n")
        f.write("\n".join(lines) + "\n")
    return path


def write_generalization_table(best, gen_results):
    headers = [
        "Algorithm", "Best Hyperparameters",
        "Empty-Start Reward", "Empty-Start Gridlock",
        "Randomized-Start Reward", "Randomized-Start Gridlock",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for algo in ALGORITHMS:
        b, g = best[algo], gen_results[algo]
        lines.append(
            f"| {algo.upper()} | {fmt_params(b['params'])} | "
            f"{b['avg_eval_reward']:.1f} | {fmt_pct(b['gridlock_rate'])} | "
            f"{g['avg_reward']:.1f} | {fmt_pct(g['gridlock_rate'])} |"
        )

    path = os.path.join(TABLES_DIR, "generalization_test.md")
    with open(path, "w") as f:
        f.write(f"# Generalization Test ({GENERALIZATION_EPISODES} episodes, randomized start states)\n\n")
        f.write("\n".join(lines) + "\n")
    return path


def retrain_best_with_logging(algorithm, params):
    run_dir = os.path.join(REPORT_RUNS_DIR, f"{algorithm}_best")
    os.makedirs(run_dir, exist_ok=True)

    env = Monitor(TrafficSignalEnv(), filename=run_dir)
    model = build_model(algorithm, env, params)
    model.set_logger(configure(run_dir, ["csv"]))
    model.learn(total_timesteps=TIMESTEPS, log_interval=1)

    return run_dir


def load_reward_curve(run_dir):
    x, y = ts2xy(load_results(run_dir), X_TIMESTEPS)
    return np.asarray(x), np.asarray(y)


def load_progress_rows(run_dir):
    path = os.path.join(run_dir, "progress.csv")
    with open(path) as f:
        return list(csv.DictReader(f))


def extract_series(rows, y_key, x_key="time/total_timesteps"):
    xs, ys = [], []
    for r in rows:
        xv, yv = r.get(x_key), r.get(y_key)
        if not xv or not yv:
            continue
        try:
            xs.append(float(xv))
            ys.append(float(yv))
        except ValueError:
            continue
    xs, ys = np.asarray(xs), np.asarray(ys)
    if len(xs) > MAX_PLOTTED_POINTS:
        idx = np.linspace(0, len(xs) - 1, MAX_PLOTTED_POINTS).astype(int)
        xs, ys = xs[idx], ys[idx]
    return xs, ys


def smooth_xy(x, y, window=10):
    if len(y) < window:
        return x, y
    y_smooth = np.convolve(y, np.ones(window) / window, mode="valid")
    return x[window - 1:], y_smooth


def plot_reward_curves_grid(curves, best):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, algo in zip(axes.flat, ALGORITHMS):
        x, y = curves[algo]
        color = PLOT_COLORS[algo]
        ax.plot(x, y, alpha=0.25, color=color)
        xs, ys = smooth_xy(x, y)
        ax.plot(xs, ys, color=color, linewidth=2)
        ax.set_title(f"{algo.upper()} ({fmt_params(best[algo]['params'])})")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Episode Reward")
        ax.grid(alpha=0.3)
    fig.suptitle("Training Reward Curves — Best Config per Algorithm")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "01_reward_curves_grid.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_dqn_loss(rows_dqn, best_dqn):
    xs, ys = extract_series(rows_dqn, "train/loss")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, color=PLOT_COLORS["dqn"])
    ax.set_title(f"DQN Training Loss ({fmt_params(best_dqn['params'])})")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "02_dqn_loss.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_entropy_curves(progress_rows, best):
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in ("ppo", "a2c", "reinforce"):
        xs, ys = extract_series(progress_rows[algo], "train/entropy_loss")
        ax.plot(xs, ys, label=f"{algo.upper()} ({fmt_params(best[algo]['params'])})", color=PLOT_COLORS[algo])
    ax.set_title("Policy Entropy Loss During Training")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Entropy Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "03_entropy_curves.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_convergence_overlay(curves, best):
    fig, ax = plt.subplots(figsize=(9, 6))
    for algo in ALGORITHMS:
        x, y = curves[algo]
        xs, ys = smooth_xy(x, y)
        ax.plot(xs, ys, label=f"{algo.upper()} ({fmt_params(best[algo]['params'])})", color=PLOT_COLORS[algo])
    ax.set_title("Convergence Comparison (smoothed episode reward)")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Reward (moving average)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "04_convergence_comparison.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_generalization(best, gen_results):
    x = np.arange(len(ALGORITHMS))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    empty_rewards = [best[a]["avg_eval_reward"] for a in ALGORITHMS]
    rand_rewards = [gen_results[a]["avg_reward"] for a in ALGORITHMS]
    axes[0].bar(x - width / 2, empty_rewards, width, label="Empty start")
    axes[0].bar(x + width / 2, rand_rewards, width, label="Randomized start")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([a.upper() for a in ALGORITHMS])
    axes[0].set_ylabel("Avg Episode Reward")
    axes[0].set_title("Reward: Empty vs Randomized Start")
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="y")

    empty_gridlock = [best[a]["gridlock_rate"] * 100 for a in ALGORITHMS]
    rand_gridlock = [gen_results[a]["gridlock_rate"] * 100 for a in ALGORITHMS]
    axes[1].bar(x - width / 2, empty_gridlock, width, label="Empty start")
    axes[1].bar(x + width / 2, rand_gridlock, width, label="Randomized start")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([a.upper() for a in ALGORITHMS])
    axes[1].set_ylabel("Gridlock Rate (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Gridlock Rate: Empty vs Randomized Start")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")

    fig.suptitle(f"Generalization Test: Randomized Start States ({GENERALIZATION_EPISODES} episodes)")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "05_generalization_test.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def run_generalization_test(best):
    results = {}
    for algo in ALGORITHMS:
        model = MODEL_CLASSES[algo].load(BEST_MODEL_PATHS[algo])
        env = TrafficSignalEnv()

        episode_rewards, episode_lengths = [], []
        gridlock_count = 0

        for _ in range(GENERALIZATION_EPISODES):
            obs, _info = env.reset(randomize_start=True)
            total_reward, steps = 0.0, 0
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
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

        results[algo] = {
            "avg_reward": sum(episode_rewards) / GENERALIZATION_EPISODES,
            "avg_length": sum(episode_lengths) / GENERALIZATION_EPISODES,
            "gridlock_rate": gridlock_count / GENERALIZATION_EPISODES,
        }
        print(
            f"  {algo}: reward {results[algo]['avg_reward']:.1f}, "
            f"length {results[algo]['avg_length']:.1f}, "
            f"gridlock {results[algo]['gridlock_rate'] * 100:.0f}%"
        )

    return results


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(REPORT_RUNS_DIR, exist_ok=True)

    print("Loading sweep results...")
    results = load_sweep_results()
    best = best_per_algorithm(results)
    for algo in ALGORITHMS:
        r = best[algo]
        print(f"  best {algo}: {r['params']} (reward {r['avg_eval_reward']:.1f}, gridlock {r['gridlock_rate'] * 100:.0f}%)")

    print("\nWriting hyperparameter tables...")
    for algo in ALGORITHMS:
        md_path, csv_path = write_hyperparameter_table(algo, results)
        print(f"  {algo}: {md_path}, {csv_path}")
    print(f"  summary: {write_summary_comparison(best)}")

    print("\nRetraining best configs with logging enabled (for training curves)...")
    run_dirs = {}
    for algo in ALGORITHMS:
        print(f"  retraining {algo}: {best[algo]['params']}...", flush=True)
        run_dirs[algo] = retrain_best_with_logging(algo, best[algo]["params"])

    print("\nBuilding plots 1-4 (training curves)...")
    curves = {algo: load_reward_curve(run_dirs[algo]) for algo in ALGORITHMS}
    progress_rows = {algo: load_progress_rows(run_dirs[algo]) for algo in ALGORITHMS}

    print(f"  {plot_reward_curves_grid(curves, best)}")
    print(f"  {plot_dqn_loss(progress_rows['dqn'], best['dqn'])}")
    print(f"  {plot_entropy_curves(progress_rows, best)}")
    print(f"  {plot_convergence_overlay(curves, best)}")

    print(f"\nRunning generalization test ({GENERALIZATION_EPISODES} episodes/model, randomized start)...")
    gen_results = run_generalization_test(best)

    print(f"\n  {plot_generalization(best, gen_results)}")
    print(f"  {write_generalization_table(best, gen_results)}")

    gen_results_path = "logs/generalization_results.json"
    with open(gen_results_path, "w") as f:
        json.dump(gen_results, f, indent=2)
    print(f"  {gen_results_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
