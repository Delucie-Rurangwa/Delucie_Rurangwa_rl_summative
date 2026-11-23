from gymnasium.envs.registration import register

from environment.custom_env import ACTION_NAMES, BatterySwapStationEnv

register(id="BatterySwapStation-v0", entry_point="environment.custom_env:BatterySwapStationEnv")

__all__ = ["BatterySwapStationEnv", "ACTION_NAMES"]
