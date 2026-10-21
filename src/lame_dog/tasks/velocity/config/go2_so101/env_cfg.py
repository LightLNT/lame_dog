# Copyright (c) 2026, The lame_dog Project Authors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based velocity task for the three-powered-leg Go2–SO-101 robot."""

from isaaclab.envs import mdp
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass, replace

from isaaclab_tasks.core.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg

from lame_dog.assets.go2_so101_cfg import GO2_SO101_CFG


@configclass
class LameDogGo2So101FlatEnvCfg(UnitreeGo2FlatEnvCfg):
    """Go2-SO101 velocity task with three powered legs."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot = replace(GO2_SO101_CFG, prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.joint_names = [
            "FR_.*_joint",
            "RL_.*_joint",
            "RR_.*_joint",
        ]

        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.2
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.2

        # Reward swing time only for the three powered feet.
        self.rewards.feet_air_time.weight = 0.05
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = "(FR|RL|RR)_foot"
        self.rewards.feet_air_time.params["threshold"] = 0.15
        self.rewards.undesired_contacts = RewardTermCfg(
            func=mdp.undesired_contacts,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"),
                "threshold": 1.0,
            },
        )
        # Penalize early failure once; timeouts are not failures.
        self.rewards.terminating = RewardTermCfg(func=mdp.terminated_penalty, weight=-2.0)
        # Include the passive FL thigh as well as the three powered legs.
        self.terminations.thigh_contact = TerminationTermCfg(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"),
                "threshold": 1.0,
            },
        )


@configclass
class LameDogGo2So101LowSpeedEnvCfg(LameDogGo2So101FlatEnvCfg):
    """Historical low-speed phase used before full-speed push training."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 0.3)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)
        self.commands.base_velocity.vel_xy_success_threshold = 0.1
        self.commands.base_velocity.vel_yaw_success_threshold = 0.1

        self.events.push_robot = None
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.params["threshold"] = 0.5
