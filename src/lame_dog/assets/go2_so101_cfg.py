# Copyright (c) 2026, The lame_dog Project Authors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 with a passive front-left leg and a mounted SO-101 arm."""

from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.utils import replace

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from lame_dog.assets import LAME_DOG_ASSETS_DIR

GO2_SO101_CFG = replace(
    UNITREE_GO2_CFG,
    spawn=replace(
        UNITREE_GO2_CFG.spawn,
        usd_path=str(LAME_DOG_ASSETS_DIR / "robots" / "go2_so101.usda"),
    ),
    actuators={
        "base_legs": replace(
            UNITREE_GO2_CFG.actuators["base_legs"],
            joint_names_expr=["FR_.*_joint", "RL_.*_joint", "RR_.*_joint"],
        ),
        # Keep the failed joints in the articulation, but remove all motor effort.
        "failed_fl": IdealPDActuatorCfg(
            joint_names_expr=["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"],
            stiffness=0.0,
            damping=0.0,
            actuator_effort_limit=0.0,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            ],
            effort_limit_sim=3.35,
            velocity_limit_sim=30.0,
            stiffness=40.0,
            damping=2.0,
        ),
    },
)
