"""Run the battery-swap station with the best trained agent.

    uv run main.py                      # best agent overall, one simulated day, with the window
    uv run main.py --algo ppo           # best PPO agent (dqn, reinforce, ppo, a2c)
    uv run main.py --episodes 3         # several days
    uv run main.py --busy               # unseen busy day (+30% riders)
    uv run main.py --random             # untrained random agent, for comparison
    uv run main.py --no-render          # terminal output only

Training and analysis:
    uv run python -m training.dqn_training
    uv run python -m training.pg_training --algo all
    uv run python -m training.analysis
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from environment import ACTION_NAMES, BatterySwapStationEnv

ROOT = Path(__file__).resolve().parent
BEST = ROOT / "logs" / "best_models.json"


def load_agent(algo: str | None):
    from training.common import load_policy_net
    info = json.loads(BEST.read_text())
    algo = algo or info["best_overall"]
    entry = info["models"][algo]
    net, _ = load_policy_net(ROOT / entry["model"])
    print(f"Loaded best {algo.upper()} agent (run {entry['run']}) from {entry['model']}")
    print(f"Evaluated reward per day: {entry['evaluation']['mean_reward']:.2f}\n")
    return net, algo


def run(args):
    if args.random or not BEST.exists():
        if not args.random:
            print("No trained models found yet (logs/best_models.json missing): showing a random agent.\n")
        net, name = None, "random"
    else:
        net, name = load_agent(args.algo)
    rng = np.random.default_rng(args.seed)

    env = BatterySwapStationEnv(render_mode=None if args.no_render else "human", randomize_demand=False)
    options = {"demand_scale": 1.3} if args.busy else {}
    for ep in range(args.episodes):
        obs, info = env.reset(seed=20_000 + args.seed + ep, options=options)
        done = False
        print(f"=== Day {ep + 1} | agent: {name} | {'busy day' if args.busy else 'normal day'} ===")
        while not done:
            if net is None:
                action = int(rng.integers(env.action_space.n))
            else:
                with torch.no_grad():
                    action = int(net(torch.as_tensor(obs)).argmax().item())
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            hh = int(info["hour"])
            mm = int(round((info["hour"] - hh) * 60)) % 60
            print(f"{hh:02d}:{mm:02d}  {ACTION_NAMES[action]:<12} reward {reward:+6.2f} | "
                  f"waiting {info['queue']:>2}  ready {info['ready']:>2}/10  grid {info['power_kw']:.1f} kW | "
                  f"served {info['served']:>3}  lost {info['lost']:>2}  total {info['reward']:+7.2f}")
        print(f"\nDay {ep + 1} done: reward {info['reward']:.2f} | served {info['served']} | lost {info['lost']} | "
              f"electricity {info['energy_cost']:,.0f} RWF | fast-charged {info['fast_kwh']:.1f} kWh"
              + ("  (station failed: too many riders lost)" if terminated else "") + "\n")
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Battery-swap station: watch an agent run the chargers")
    p.add_argument("--algo", choices=["dqn", "reinforce", "ppo", "a2c"], default=None)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--busy", action="store_true")
    p.add_argument("--random", action="store_true")
    p.add_argument("--no-render", action="store_true")
    run(p.parse_args())
