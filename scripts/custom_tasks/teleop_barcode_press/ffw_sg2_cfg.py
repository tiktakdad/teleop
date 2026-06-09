# Copyright 2025 ROBOTIS / teleop custom task
"""FFW_SG2 (AI Worker) articulation configuration for Isaac Lab."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

CUSTOM_ASSETS_DIR = os.environ.get("CUSTOM_ASSETS_DIR", "/workspace/user/custom_assets")
FFW_SG2_USD = os.path.join(CUSTOM_ASSETS_DIR, "robot/ai_worker/FFW_SG2.usd")

# Reset/play 시 head_joint1 기본 피치(아래보기) 초기값.
FFW_HEAD_DOWN_INIT_POS = float(os.environ.get("FFW_HEAD_DOWN_INIT_POS", "0.55"))

# Match the authored articulation pose in scene/reference.usd.
_DEFAULT_JOINT_POS = {
    **{f"arm_l_joint{i}": 0.0 for i in range(1, 8)},
    **{f"arm_r_joint{i}": 0.0 for i in range(1, 8)},
    **{f"gripper_l_joint{i}": 0.0 for i in range(1, 5)},
    **{f"gripper_r_joint{i}": 0.0 for i in range(1, 5)},
    "head_joint1": FFW_HEAD_DOWN_INIT_POS,
    "head_joint2": 0.0,
    "lift_joint": 0.0,
    "left_wheel_drive": 0.0,
    "left_wheel_steer": 0.0,
    "right_wheel_drive": 0.0,
    "right_wheel_steer": 0.0,
    "rear_wheel_drive": 0.0,
    "rear_wheel_steer": 0.0,
}

FFW_SG2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=FFW_SG2_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        "lift": ImplicitActuatorCfg(
            joint_names_expr=["lift_joint"],
            velocity_limit_sim=0.2,
            effort_limit_sim=1_000_000.0,
            stiffness=10_000.0,
            damping=100.0,
        ),
        "arm_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["arm_[lr]_joint[1-2]"],
            velocity_limit_sim=15.0,
            effort_limit_sim=61.4,
            stiffness=600.0,
            damping=30.0,
        ),
        "arm_elbow": ImplicitActuatorCfg(
            joint_names_expr=["arm_[lr]_joint[3-6]"],
            velocity_limit_sim=15.0,
            effort_limit_sim=31.7,
            stiffness=600.0,
            damping=20.0,
        ),
        "arm_wrist": ImplicitActuatorCfg(
            joint_names_expr=["arm_[lr]_joint7"],
            velocity_limit_sim=6.0,
            effort_limit_sim=5.1,
            stiffness=200.0,
            damping=3.0,
        ),
        "gripper_master": ImplicitActuatorCfg(
            joint_names_expr=["gripper_[lr]_joint1"],
            velocity_limit_sim=2.2,
            effort_limit_sim=30.0,
            stiffness=100.0,
            damping=4.0,
        ),
        "gripper_slave": ImplicitActuatorCfg(
            joint_names_expr=["gripper_[lr]_joint[2-4]"],
            effort_limit_sim=20.0,
            stiffness=2.0,
            damping=0.5,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_joint[1-2]"],
            velocity_limit_sim=2.0,
            effort_limit_sim=30.0,
            stiffness=150.0,
            damping=3.0,
        ),
        "wheels_locked": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_(drive|steer)"],
            velocity_limit_sim=0.0,
            effort_limit_sim=200.0,
            stiffness=5_000.0,
            damping=500.0,
        ),
    },
)
