# Copyright (c) 2026, The lame_dog Project Authors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2-SO101 flat-ground velocity task registration."""

import gymnasium as gym

from . import agents

gym.register(
    id="LameDog-Velocity-Flat-Go2So101",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:LameDogGo2So101FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LameDogFlatPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="LameDog-Velocity-Flat-Go2So101-LowSpeed",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:LameDogGo2So101LowSpeedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LameDogFlatPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
