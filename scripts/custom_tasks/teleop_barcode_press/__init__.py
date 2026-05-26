# Copyright 2025 ROBOTIS / teleop custom task

import gymnasium as gym

from .barcode_press_env_cfg import BarcodePressFFWSG2EnvCfg

gym.register(
    id="Isaac-BarcodePress-FFW-SG2-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": BarcodePressFFWSG2EnvCfg,
    },
    disable_env_checker=True,
)
