# Copyright (c) 2026, The lame_dog Project Authors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for the Go2–SO-101 velocity task."""

from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2FlatPPORunnerCfg


@configclass
class LameDogFlatPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    """PPO configuration for the three-legged Go2-SO101 task."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "lame_dog_go2_so101_flat"
