import numpy as np
from gymnasium.utils.env_checker import check_env

from environment import BatterySwapStationEnv
from environment import custom_env as E


def test_gymnasium_api():
    check_env(BatterySwapStationEnv(), skip_render_check=True)


def test_day_length_and_obs_bounds():
    env = BatterySwapStationEnv()
    obs, _ = env.reset(seed=0, options={"init_soc": "full"})
    steps, done = 0, False
    while not done:
        obs, _, term, trunc, _ = env.step(1)
        assert env.observation_space.contains(obs)
        steps += 1
        done = term or trunc
    assert steps <= E.STEPS_PER_DAY


def test_grid_cap_never_exceeded():
    env = BatterySwapStationEnv()
    env.reset(seed=1, options={"init_soc": "low"})
    for _ in range(20):
        env.step(2)   # keep asking for fast charging
        assert env.power_kw() <= E.GRID_CAP_KW + 1e-9


def test_invalid_action_is_penalised():
    env = BatterySwapStationEnv()
    env.reset(seed=2, options={"init_soc": "full"})
    env.step(5)       # nothing is charging -> PAUSE is invalid
    assert env.last_action_valid is False


def test_charging_is_persistent():
    env = BatterySwapStationEnv()
    env.reset(seed=3, options={"init_soc": "low"})
    env.step(1)
    soc_before = env.soc.copy()
    env.step(0)       # HOLD keeps the charger running
    assert np.any(env.soc > soc_before)
