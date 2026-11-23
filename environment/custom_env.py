"""
BatterySwapStationEnv
=====================

Mission: reduce waiting time at electric motorcycle (e-moto taxi) battery-swap
stations in Kigali while keeping electricity costs and battery wear low.

The agent is the station's smart charging controller. Every 10 minutes it decides
how the station's chargers should run. Riders arrive at random (more during the
morning and evening rush), take a fully charged battery and leave their depleted
one behind. If no charged battery is ready, they wait; if they wait too long,
they leave (a lost customer).

Why this is an RL problem (and not one-step prediction):
  * Decisions have delayed consequences: a battery started now is only ready
    1-2 hours later, so the agent must charge *before* the rush.
  * Persistent state: charger modes and battery charge levels carry over
    between steps and depend on earlier actions.
  * Trade-offs over time: charging at night is cheap, charging at the evening
    peak is expensive; fast charging is quicker but costs more power and wears
    batteries; the station's grid connection limits how many batteries can
    charge at once.

All parameters below are simulation assumptions (no operator data was available).
In a real deployment they would come from the station's swap logs, the battery
management system (BMS), the charger controller API and the utility tariff.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ----------------------------------------------------------------------------
# Station and simulation parameters (assumptions, documented in the report)
# ----------------------------------------------------------------------------
STEP_MINUTES = 10                     # one decision every 10 minutes
STEPS_PER_DAY = 24 * 60 // STEP_MINUTES  # 144 steps = one full day (00:00-24:00)

N_SLOTS = 10                          # batteries held / charging bays at the station
BATTERY_KWH = 2.9                     # usable capacity of one e-moto battery
NORMAL_KW = 1.0                       # normal charging power per battery
FAST_KW = 2.5                         # fast charging power per battery
GRID_CAP_KW = 8.0                     # maximum total power of the station's grid connection
READY_SOC = 0.95                      # a battery at or above this can be given to a rider

MAX_QUEUE = 10                        # riders that can wait at once
PATIENCE_STEPS = 3                    # a rider leaves after waiting 30 minutes
LOST_LIMIT = 15                       # station "fails" the day after this many lost riders

RETURNED_SOC_RANGE = (0.10, 0.30)     # charge level of batteries riders bring back

# Expected riders arriving per hour, by hour of day (Kigali moto-taxi pattern:
# morning rush 06-09, steady day, evening rush 17-20, quiet night).
HOURLY_ARRIVALS = np.array(
    [0.2, 0.1, 0.1, 0.1, 0.3, 1.0,   # 00-05
     3.5, 4.5, 4.0, 2.5, 2.0, 2.0,   # 06-11
     2.5, 2.5, 2.0, 2.0, 2.5, 4.0,   # 12-17
     4.5, 4.0, 2.5, 1.5, 0.8, 0.4],  # 18-23
    dtype=np.float64,
)
MAX_HOURLY_ARRIVALS = 5.0             # used to normalise the demand forecast

# Time-of-use electricity price in RWF per kWh (illustrative three-band tariff).
TARIFF_OFF_PEAK = 120.0               # 22:00-06:00
TARIFF_SHOULDER = 180.0               # 06:00-17:00
TARIFF_PEAK = 260.0                   # 17:00-22:00

# Reward weights
R_SERVE = 1.0          # + per rider served
C_WAIT = 0.10          # - per rider waiting, per step
C_LOST = 2.0           # - per rider who leaves without a battery
C_ENERGY = 1 / 1000    # - per RWF spent on electricity (1 point = 1000 RWF)
C_FAST_WEAR = 0.05     # - per kWh delivered by fast charging (battery degradation)
C_INVALID = 0.05       # - for an action that cannot be carried out
C_FAIL = 10.0          # - one-off penalty when the station fails (too many lost riders)

# Charger modes
IDLE, NORMAL, FAST = 0, 1, 2

# Discrete actions
ACTION_NAMES = [
    "HOLD",          # 0: keep all chargers as they are
    "START_NORMAL",  # 1: start normal charging on the most depleted idle battery
    "START_FAST",    # 2: start fast charging on the most depleted idle battery
    "BOOST",         # 3: switch one normal-charging battery to fast
    "SLOW_DOWN",     # 4: switch one fast-charging battery back to normal
    "PAUSE",         # 5: pause the charging battery that is closest to full
]


def tariff_at(hour: float) -> float:
    """Electricity price (RWF/kWh) at a given hour of the day."""
    if 17 <= hour < 22:
        return TARIFF_PEAK
    if 6 <= hour < 17:
        return TARIFF_SHOULDER
    return TARIFF_OFF_PEAK


class BatterySwapStationEnv(gym.Env):
    """Smart charging control of a single e-moto battery-swap station."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    # Size of the observation vector (see _get_obs for the meaning of each entry)
    OBS_SIZE = N_SLOTS * 2 + 7

    def __init__(self, render_mode: str | None = None, demand_scale: float = 1.0,
                 randomize_demand: bool = True):
        super().__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.base_demand_scale = demand_scale
        self.randomize_demand = randomize_demand

        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        low = np.zeros(self.OBS_SIZE, dtype=np.float32)
        high = np.ones(self.OBS_SIZE, dtype=np.float32)
        low[N_SLOTS * 2 + 1] = -1.0   # sin(time)
        low[N_SLOTS * 2 + 2] = -1.0   # cos(time)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._renderer = None
        self._reset_state()

    # ------------------------------------------------------------------ state
    def _reset_state(self):
        self.t = 0
        self.soc = np.zeros(N_SLOTS, dtype=np.float64)
        self.mode = np.zeros(N_SLOTS, dtype=np.int64)
        self.queue: list[int] = []            # waiting time (steps) of each rider, FIFO
        self.demand_scale = self.base_demand_scale
        self.last_action = 0
        self.last_reward = 0.0
        self.last_action_valid = True
        self.totals = dict(served=0, lost=0, arrived=0, energy_kwh=0.0,
                           energy_cost=0.0, fast_kwh=0.0, wait_steps=0,
                           invalid_actions=0, reward=0.0)

    @property
    def hour(self) -> float:
        return (self.t * STEP_MINUTES) / 60.0

    def power_kw(self) -> float:
        return float(np.sum(self.mode == NORMAL) * NORMAL_KW + np.sum(self.mode == FAST) * FAST_KW)

    def ready_count(self) -> int:
        return int(np.sum(self.soc >= READY_SOC))

    def expected_arrivals_next_hour(self) -> float:
        return float(HOURLY_ARRIVALS[int(self.hour) % 24] * self.demand_scale)

    def _get_obs(self) -> np.ndarray:
        angle = 2 * math.pi * self.hour / 24.0
        obs = np.concatenate([
            self.soc,                                         # [0:10]  charge level of each battery (BMS)
            self.mode / 2.0,                                  # [10:20] charger mode 0 / 0.5 / 1 (charger API)
            [len(self.queue) / MAX_QUEUE,                     # [20]   riders waiting (camera / swap terminal)
             math.sin(angle), math.cos(angle),                # [21:23] time of day (station clock)
             (tariff_at(self.hour) - TARIFF_OFF_PEAK) / (TARIFF_PEAK - TARIFF_OFF_PEAK),  # [23] price band
             min(self.expected_arrivals_next_hour() / MAX_HOURLY_ARRIVALS, 1.0),  # [24] demand forecast
             self.power_kw() / GRID_CAP_KW,                   # [25]   grid load (smart meter)
             self.ready_count() / N_SLOTS],                   # [26]   batteries ready to swap
        ]).astype(np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _get_info(self) -> dict[str, Any]:
        return dict(hour=self.hour, queue=len(self.queue), ready=self.ready_count(),
                    power_kw=self.power_kw(), tariff=tariff_at(self.hour),
                    action_name=ACTION_NAMES[self.last_action],
                    action_valid=self.last_action_valid, **self.totals)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Start of a day at 00:00.

        options (all optional, used for generalisation tests):
            demand_scale: float  - multiply rider demand (e.g. 1.3 = busy day)
            init_soc: "random" | "low" | "full" - starting charge of the batteries
        """
        super().reset(seed=seed)
        options = options or {}
        self._reset_state()

        if "demand_scale" in options:
            self.demand_scale = float(options["demand_scale"])
        elif self.randomize_demand:
            self.demand_scale = self.base_demand_scale * self.np_random.uniform(0.85, 1.15)

        init = options.get("init_soc", "random")
        if init == "low":
            self.soc = self.np_random.uniform(0.10, 0.40, N_SLOTS)
        elif init == "full":
            self.soc = np.ones(N_SLOTS)
        else:  # leftover batteries from yesterday: a mix of empty and charged
            self.soc = self.np_random.uniform(0.15, 1.0, N_SLOTS)

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self._get_info()

    def _apply_action(self, action: int) -> bool:
        """Change charger modes. Returns False if the action could not be applied."""
        idle = np.where((self.mode == IDLE) & (self.soc < 1.0))[0]
        normal = np.where(self.mode == NORMAL)[0]
        fast = np.where(self.mode == FAST)[0]
        charging = np.where(self.mode != IDLE)[0]
        power = self.power_kw()

        if action == 0:  # HOLD
            return True
        if action == 1:  # START_NORMAL
            if len(idle) == 0 or power + NORMAL_KW > GRID_CAP_KW + 1e-9:
                return False
            self.mode[idle[np.argmin(self.soc[idle])]] = NORMAL
            return True
        if action == 2:  # START_FAST
            if len(idle) == 0 or power + FAST_KW > GRID_CAP_KW + 1e-9:
                return False
            self.mode[idle[np.argmin(self.soc[idle])]] = FAST
            return True
        if action == 3:  # BOOST
            if len(normal) == 0 or power - NORMAL_KW + FAST_KW > GRID_CAP_KW + 1e-9:
                return False
            self.mode[normal[np.argmin(self.soc[normal])]] = FAST
            return True
        if action == 4:  # SLOW_DOWN
            if len(fast) == 0:
                return False
            self.mode[fast[np.argmax(self.soc[fast])]] = NORMAL
            return True
        if action == 5:  # PAUSE
            if len(charging) == 0:
                return False
            self.mode[charging[np.argmax(self.soc[charging])]] = IDLE
            return True
        raise ValueError(f"Invalid action {action}")

    def step(self, action):
        action = int(action)
        assert self.action_space.contains(action), f"invalid action {action}"
        hours = STEP_MINUTES / 60.0
        price = tariff_at(self.hour)

        # 1. Agent changes charger modes
        valid = self._apply_action(action)
        self.last_action, self.last_action_valid = action, valid

        # 2. Charge for 10 minutes (energy limited by remaining capacity)
        kw = np.where(self.mode == FAST, FAST_KW, np.where(self.mode == NORMAL, NORMAL_KW, 0.0))
        room_kwh = (1.0 - self.soc) * BATTERY_KWH
        delivered = np.minimum(kw * hours, room_kwh)
        self.soc = np.minimum(self.soc + delivered / BATTERY_KWH, 1.0)
        energy_kwh = float(delivered.sum())
        fast_kwh = float(delivered[self.mode == FAST].sum())
        energy_cost = energy_kwh * price
        self.mode[self.soc >= 1.0 - 1e-9] = IDLE      # chargers stop when a battery is full

        # 3. Riders arrive (Poisson process, rate depends on time of day)
        lam = HOURLY_ARRIVALS[int(self.hour) % 24] * self.demand_scale * hours
        arrivals = int(self.np_random.poisson(lam))
        space = MAX_QUEUE - len(self.queue)
        lost = max(0, arrivals - space)                # queue full -> rider goes elsewhere
        self.queue.extend([0] * min(arrivals, space))

        # 4. Serve waiting riders (first come, first served) with ready batteries
        served = 0
        while self.queue:
            ready = np.where(self.soc >= READY_SOC)[0]
            if len(ready) == 0:
                break
            slot = ready[np.argmax(self.soc[ready])]
            self.soc[slot] = self.np_random.uniform(*RETURNED_SOC_RANGE)  # depleted battery returned
            self.mode[slot] = IDLE
            self.queue.pop(0)
            served += 1

        # 5. Remaining riders wait; impatient ones leave
        waiting = len(self.queue)
        self.queue = [w + 1 for w in self.queue]
        before = len(self.queue)
        self.queue = [w for w in self.queue if w < PATIENCE_STEPS]
        lost += before - len(self.queue)

        # 6. Reward
        reward = (R_SERVE * served
                  - C_WAIT * waiting
                  - C_LOST * lost
                  - C_ENERGY * energy_cost
                  - C_FAST_WEAR * fast_kwh
                  - (0.0 if valid else C_INVALID))

        # 7. Book-keeping and termination
        tt = self.totals
        tt["served"] += served
        tt["lost"] += lost
        tt["arrived"] += arrivals
        tt["energy_kwh"] += energy_kwh
        tt["energy_cost"] += energy_cost
        tt["fast_kwh"] += fast_kwh
        tt["wait_steps"] += waiting
        tt["invalid_actions"] += 0 if valid else 1

        self.t += 1
        terminated = tt["lost"] >= LOST_LIMIT          # station failed its riders today
        if terminated:
            reward -= C_FAIL
        truncated = (not terminated) and self.t >= STEPS_PER_DAY   # end of the day

        tt["reward"] += reward
        self.last_reward = reward

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    # ------------------------------------------------------------------ rendering
    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from environment.rendering import StationRenderer
            self._renderer = StationRenderer(self.render_mode, fps=self.metadata["render_fps"])
        return self._renderer.draw(self)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
