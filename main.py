"""Entry point: run the battery-swap station simulation.

    uv run main.py                     # watch a random (untrained) agent for one day
    uv run main.py --episodes 3        # several days
    uv run main.py --no-render         # terminal output only

Once training is done, this file will load the best trained model instead
(added in the next step of the project).
"""

import argparse

import numpy as np

from environment import ACTION_NAMES, BatterySwapStationEnv


def run(episodes: int, render: bool, seed: int):
    env = BatterySwapStationEnv(render_mode="human" if render else None)
    rng = np.random.default_rng(seed)
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = int(rng.integers(env.action_space.n))   # random agent
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if env.t % 6 == 0 or done:   # print once per simulated hour
                print(f"[day {ep + 1}] {info['hour']:05.2f}h  action={ACTION_NAMES[action]:<12} "
                      f"reward={reward:+.2f}  queue={info['queue']}  ready={info['ready']}  "
                      f"served={info['served']}  lost={info['lost']}")
        print(f"Day {ep + 1} finished: total reward {info['reward']:.1f}, served {info['served']}, "
              f"lost {info['lost']}, electricity cost {info['energy_cost']:,.0f} RWF"
              + ("  (station FAILED: too many lost riders)" if terminated else ""))
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    run(a.episodes, not a.no_render, a.seed)
