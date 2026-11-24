"""Builds every figure and table for the report from the saved training logs.

    uv run python -m training.analysis

Outputs (in assets/ and logs/):
  cumulative_rewards.png   reward per simulated day for the best run of each algorithm
  training_stability.png   DQN loss and policy entropy of REINFORCE, PPO and A2C
  convergence.png          smoothed learning curves with the episode each method converged
  sweep_overview.png       evaluated reward of all 10 runs of each algorithm
  generalization.png       best models tested on unseen start conditions
  logs/best_models.json, logs/convergence.csv, logs/generalization.csv
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from environment import BatterySwapStationEnv
from training.common import (EVAL_SEEDS, GENERALISATION_SCENARIOS, LOG_DIR, MODEL_DIR, ROOT, evaluate,
                             load_policy_net)

ASSETS = ROOT / "assets"
ALGOS = ["dqn", "reinforce", "ppo", "a2c"]
NAMES = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}
COLORS = {"dqn": "#2f6fb0", "reinforce": "#c0504d", "ppo": "#3a9a5b", "a2c": "#e0a030"}
SUBDIR = {"dqn": "dqn", "reinforce": "pg", "ppo": "pg", "a2c": "pg"}

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})


def moving_average(x, n=20):
    x = np.asarray(x, dtype=float)
    if len(x) < n:
        return x
    c = np.cumsum(np.insert(x, 0, 0))
    ma = (c[n:] - c[:-n]) / n
    return np.concatenate([np.cumsum(x[:n - 1]) / np.arange(1, n), ma])


def load_runs(algo):
    return [json.loads(p.read_text()) for p in sorted((LOG_DIR / algo).glob("run_*.json"))]


# ------------------------------------------------------------------ baselines
def rule_policy_net():
    """Simple rule used as a reference: always try to start a normal charge."""
    import torch

    class Always(torch.nn.Module):
        def forward(self, obs):
            out = torch.zeros(6)
            out[1] = 1.0
            return out
    return Always()


def random_eval(options=None):
    env = BatterySwapStationEnv(randomize_demand=False)
    rng = np.random.default_rng(0)
    rewards = []
    for s in EVAL_SEEDS:
        env.reset(seed=s, options=options or {})
        done = False
        while not done:
            _, _, te, tr, info = env.step(int(rng.integers(6)))
            done = te or tr
        rewards.append(info["reward"])
    return float(np.mean(rewards)), float(np.std(rewards))


# ------------------------------------------------------------------ figures
def fig_cumulative(best):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for ax, algo in zip(axes.flat, ALGOS):
        r = best[algo]["log"]["episode_rewards"]
        ax.plot(r, color=COLORS[algo], alpha=0.25, lw=0.8, label="reward per day")
        ax.plot(moving_average(r), color=COLORS[algo], lw=2, label="20-day moving average")
        ax.set_title(f"{NAMES[algo]} (best run {best[algo]['run']})")
        ax.set_xlabel("Episode (simulated day)")
        ax.set_ylabel("Total reward per day")
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Cumulative reward per episode during training, best run of each method")
    fig.tight_layout()
    fig.savefig(ASSETS / "cumulative_rewards.png")
    plt.close(fig)


def fig_stability(best, runs):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax = axes[0, 0]
    for r in runs["dqn"]:
        c = r["curves"]
        if "loss" in c:
            ax.plot(c["loss_frame"], moving_average(c["loss"], 50), color="#9db6d3", lw=0.7)
    c = best["dqn"]["log"]["curves"]
    ax.plot(c["loss_frame"], moving_average(c["loss"], 50), color=COLORS["dqn"], lw=2,
            label=f"best run {best['dqn']['run']}")
    ax.set_title("DQN objective: TD loss (smooth L1)")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Loss (50-update average)")
    ax.legend(fontsize=8)
    for ax, algo in zip([axes[0, 1], axes[1, 0], axes[1, 1]], ["reinforce", "ppo", "a2c"]):
        for r in runs[algo]:
            c = r["curves"]
            ax.plot(c["loss_frame"], moving_average(c["entropy"], 5), color="#cccccc", lw=0.7)
        c = best[algo]["log"]["curves"]
        ax.plot(c["loss_frame"], moving_average(c["entropy"], 5), color=COLORS[algo], lw=2,
                label=f"best run {best[algo]['run']}")
        ax.axhline(np.log(6), ls="--", color="grey", lw=1, label="uniform random policy")
        ax.set_title(f"{NAMES[algo]}: policy entropy")
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Entropy (nats)")
        ax.legend(fontsize=8)
    fig.suptitle("Training stability (grey lines: the other runs of the sweep)")
    fig.tight_layout()
    fig.savefig(ASSETS / "training_stability.png")
    plt.close(fig)


def convergence_episode(rewards, window=20, band=0.9):
    """First episode where the moving average reaches 90% of the way from its start to its final
    level and stays there for the rest of training."""
    ma = moving_average(rewards, window)
    if len(ma) < 2 * window:
        return None, ma
    start, final = ma[:window].mean(), ma[-50:].mean()
    threshold = start + band * (final - start)
    above = ma >= threshold
    for i in range(len(ma)):
        if above[i:].mean() >= 0.95:
            return i, ma
    return None, ma


def fig_convergence(best):
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for algo in ALGOS:
        r = best[algo]["log"]["episode_rewards"]
        ep, ma = convergence_episode(r)
        frames_per_ep = best[algo]["log"]["frames"] / max(len(r), 1)
        axes[0].plot(ma, color=COLORS[algo], lw=2, label=NAMES[algo])
        if ep is not None:
            axes[0].scatter([ep], [ma[ep]], color=COLORS[algo], s=50, zorder=5, edgecolor="black")
        rows.append(dict(algorithm=NAMES[algo], best_run=best[algo]["run"],
                         converged_at_episode=ep,
                         converged_at_steps=None if ep is None else int(ep * frames_per_ep),
                         final_training_reward=round(float(np.mean(r[-50:])), 2),
                         std_last_50=round(float(np.std(r[-50:])), 2),
                         eval_reward=round(best[algo]["eval"]["mean_reward"], 2)))
    axes[0].set_title("Smoothed reward per day (dot = converged)")
    axes[0].set_xlabel("Episode (simulated day)")
    axes[0].set_ylabel("20-day moving average reward")
    axes[0].legend()
    df = pd.DataFrame(rows)
    eps = df["converged_at_episode"].fillna(0)
    axes[1].bar(df["algorithm"], eps, color=[COLORS[a] for a in ALGOS])
    for i, v in enumerate(df["converged_at_episode"]):
        axes[1].text(i, eps[i], "not converged" if pd.isna(v) else f"{int(v)}", ha="center", va="bottom")
    axes[1].set_title("Episodes needed to reach stable performance")
    axes[1].set_ylabel("Episodes")
    fig.tight_layout()
    fig.savefig(ASSETS / "convergence.png")
    plt.close(fig)
    df.to_csv(LOG_DIR / "convergence.csv", index=False)
    print("\nConvergence\n", df.to_string(index=False))


def fig_sweeps(runs):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, algo in zip(axes, ALGOS):
        rs = sorted(runs[algo], key=lambda r: r["run"])
        means = [r["evaluation"]["mean_reward"] for r in rs]
        stds = [r["evaluation"]["std_reward"] for r in rs]
        ax.bar([str(r["run"]) for r in rs], means, yerr=stds, color=COLORS[algo], alpha=0.85, capsize=2)
        ax.set_title(NAMES[algo])
        ax.set_xlabel("Run")
    axes[0].set_ylabel("Evaluated reward per day (20 unseen days)")
    fig.suptitle("Hyperparameter sweeps: evaluated reward of each run")
    fig.tight_layout()
    fig.savefig(ASSETS / "sweep_overview.png")
    plt.close(fig)


def generalisation(best):
    rows = []
    for name, opts in GENERALISATION_SCENARIOS.items():
        row = {"scenario": name}
        for algo in ALGOS:
            row[NAMES[algo]] = round(evaluate(best[algo]["net"], options=opts)["mean_reward"], 2)
        row["Rule: always charge"] = round(evaluate(rule_policy_net(), options=opts)["mean_reward"], 2)
        row["Random"] = round(random_eval(opts)[0], 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(LOG_DIR / "generalization.csv", index=False)
    print("\nGeneralisation (mean reward per day on 20 unseen days)\n", df.to_string(index=False))

    cols = [NAMES[a] for a in ALGOS] + ["Rule: always charge"]
    colors = [COLORS[a] for a in ALGOS] + ["#888888"]
    x = np.arange(len(df))
    w = 0.16
    fig, ax = plt.subplots(figsize=(13, 4.8))
    for k, (c, col) in enumerate(zip(cols, colors)):
        ax.bar(x + (k - 2) * w, df[c], w, label=c, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels(df["scenario"], rotation=12)
    ax.set_ylabel("Mean reward per day")
    ax.set_title("Generalisation: best models on start conditions never seen in training")
    ax.axhline(0, color="black", lw=0.8)
    ax.legend(ncol=5, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(ASSETS / "generalization.png")
    plt.close(fig)


def main():
    ASSETS.mkdir(exist_ok=True)
    runs = {a: load_runs(a) for a in ALGOS}
    missing = [a for a in ALGOS if not runs[a]]
    if missing:
        raise SystemExit(f"No training logs for {missing}. Train them first (see README).")

    best = {}
    for algo in ALGOS:
        top = max(runs[algo], key=lambda r: r["evaluation"]["mean_reward"])
        path = MODEL_DIR / SUBDIR[algo] / f"{algo}_run_{top['run']:02d}.pt"
        net, _ = load_policy_net(path)
        best[algo] = dict(run=top["run"], log=top, eval=top["evaluation"], net=net, path=path)
    summary = {a: dict(run=b["run"], model=str(b["path"].relative_to(ROOT)).replace("\\", "/"),
                       evaluation=b["eval"], config=b["log"]["config"]) for a, b in best.items()}
    overall = max(summary, key=lambda a: summary[a]["evaluation"]["mean_reward"])
    (LOG_DIR / "best_models.json").write_text(json.dumps(dict(best_overall=overall, models=summary), indent=2))
    print("Best run per algorithm:")
    for a in ALGOS:
        e = summary[a]["evaluation"]
        print(f"  {NAMES[a]:<10} run {summary[a]['run']:>2}  reward {e['mean_reward']:6.2f} ± {e['std_reward']:.2f}"
              f"  served {e['served']:.1f}  lost {e['lost']:.1f}")
    print(f"Best overall: {NAMES[overall]}")

    fig_cumulative(best)
    fig_stability(best, runs)
    fig_convergence(best)
    fig_sweeps(runs)
    generalisation(best)
    print(f"\nFigures saved in {ASSETS}")


if __name__ == "__main__":
    main()
