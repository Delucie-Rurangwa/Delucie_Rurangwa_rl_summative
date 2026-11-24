"""Shared pieces for every algorithm: environment factory, networks, evaluation and saving.

All four algorithms train on exactly the same environment (BatterySwapStationEnv),
with the same observation, action space, reward and training budget, so their
results can be compared fairly.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("RL_WARNINGS", "False")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torchrl")

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchrl.envs import Compose, GymWrapper, RewardSum, StepCounter, TransformedEnv
from torchrl.modules import MLP

from environment import BatterySwapStationEnv

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
MODEL_DIR = ROOT / "models"

OBS_DIM = BatterySwapStationEnv.OBS_SIZE
N_ACTIONS = 6

# Seeds never used during training: used to evaluate every run the same way.
EVAL_SEEDS = list(range(10_000, 10_020))

# Unseen start conditions used for the generalisation test.
GENERALISATION_SCENARIOS = {
    "normal day": {},
    "empty batteries at midnight": {"init_soc": "low"},
    "all batteries full at midnight": {"init_soc": "full"},
    "busy day (+30% riders)": {"demand_scale": 1.3},
    "quiet day (-30% riders)": {"demand_scale": 0.7},
    "busy day + empty batteries": {"demand_scale": 1.3, "init_soc": "low"},
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(seed: int = 0) -> TransformedEnv:
    """TorchRL version of the custom Gymnasium environment."""
    env = TransformedEnv(
        GymWrapper(BatterySwapStationEnv()),
        Compose(StepCounter(), RewardSum()),   # adds step_count and episode_reward
    )
    env.set_seed(seed)
    return env


def mlp(out_features: int, hidden: int, layers: int = 2, activation=nn.Tanh) -> MLP:
    return MLP(in_features=OBS_DIM, out_features=out_features,
               num_cells=[hidden] * layers, activation_class=activation)


def finished_episode_rewards(data) -> list[float]:
    """Total reward of every episode that ended inside a collected batch."""
    done = data["next", "done"].squeeze(-1)
    if not done.any():
        return []
    return data["next", "episode_reward"][done].squeeze(-1).tolist()


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate(net: nn.Module, seeds=EVAL_SEEDS, options: dict | None = None) -> dict:
    """Run the greedy policy (argmax of Q-values or of policy logits) for one day per seed."""
    net.eval()
    env = BatterySwapStationEnv(randomize_demand=False)
    rewards, served, lost, cost, failed, wait = [], [], [], [], [], []
    for s in seeds:
        obs, _ = env.reset(seed=s, options=options or {})
        done = False
        while not done:
            action = int(net(torch.as_tensor(obs)).argmax().item())
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        rewards.append(info["reward"])
        served.append(info["served"])
        lost.append(info["lost"])
        cost.append(info["energy_cost"])
        failed.append(float(term))
        wait.append(info["wait_steps"] * 10 / max(info["served"] + info["lost"], 1))
    net.train()
    return dict(mean_reward=float(np.mean(rewards)), std_reward=float(np.std(rewards)),
                served=float(np.mean(served)), lost=float(np.mean(lost)),
                energy_cost=float(np.mean(cost)), avg_wait_min=float(np.mean(wait)),
                fail_rate=float(np.mean(failed)))


# ------------------------------------------------------------------ saving
class RunLogger:
    """Collects curves during one training run and writes them to logs/<algo>/run_XX.json."""

    def __init__(self, algo: str, run_id: int, config: dict):
        self.algo, self.run_id, self.config = algo, run_id, config
        self.episode_rewards: list[float] = []
        self.curves: dict[str, list[float]] = {}
        self.start = time.time()

    def episodes(self, rewards):
        self.episode_rewards.extend(rewards)

    def log(self, key: str, value: float):
        self.curves.setdefault(key, []).append(float(value))

    def last_mean(self, n=20):
        r = self.episode_rewards[-n:]
        return float(np.mean(r)) if r else float("nan")

    def save(self, evaluation: dict, frames: int) -> dict:
        out = dict(algo=self.algo, run=self.run_id, config=self.config, frames=frames,
                   minutes=(time.time() - self.start) / 60, episode_rewards=self.episode_rewards,
                   curves=self.curves, evaluation=evaluation)
        path = LOG_DIR / self.algo
        path.mkdir(parents=True, exist_ok=True)
        (path / f"run_{self.run_id:02d}.json").write_text(json.dumps(out))
        return out


def save_model(net: nn.Module, algo: str, subdir: str, config: dict, evaluation: dict, run_id: int):
    path = MODEL_DIR / subdir
    path.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=net.state_dict(), algo=algo, config=config, run=run_id,
                    evaluation=evaluation, hidden=config["hidden"],
                    activation=config.get("activation", "tanh")),
               path / f"{algo}_run_{run_id:02d}.pt")


def load_policy_net(path: Path) -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    act = nn.ReLU if ckpt["activation"] == "relu" else nn.Tanh
    net = mlp(N_ACTIONS, ckpt["hidden"], activation=act)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt


def write_sweep_table(algo: str, results: list[dict], columns: list[str]):
    """CSV with one row per run: the hyperparameters that were changed and the evaluated outcome."""
    import pandas as pd
    rows = []
    for r in results:
        row = {"run": r["run"]}
        row.update({c: r["config"][c] for c in columns})
        row.update({"mean_reward": round(r["evaluation"]["mean_reward"], 2),
                    "std_reward": round(r["evaluation"]["std_reward"], 2),
                    "served": round(r["evaluation"]["served"], 1),
                    "lost": round(r["evaluation"]["lost"], 1),
                    "minutes": round(r["minutes"], 1)})
        rows.append(row)
    df = pd.DataFrame(rows)
    LOG_DIR.mkdir(exist_ok=True)
    df.to_csv(LOG_DIR / f"{algo}_sweep.csv", index=False)
    print(df.to_string(index=False))
    return df
