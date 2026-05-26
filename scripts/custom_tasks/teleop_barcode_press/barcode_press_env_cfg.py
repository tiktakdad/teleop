# Copyright 2025 ROBOTIS / teleop custom task
"""서버랙 바코드 프레스 텔레옵 환경 (FFW_SG2 + custom server rack, Isaac Lab 2.3)."""

from __future__ import annotations

import math
import os
import tempfile

import carb
import torch
from pink.tasks import DampingTask, FrameTask

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.pink_ik import NullSpacePostureTask, PinkIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass

from . import mdp
from .ffw_sg2_cfg import FFW_SG2_CFG
from .retargeters.ffw_sg2_retargeter import FfwSg2RetargeterCfg

CUSTOM_ASSETS_DIR = os.environ.get("CUSTOM_ASSETS_DIR", "/workspace/user/custom_assets")
SERVER_RACK_USD = os.path.join(CUSTOM_ASSETS_DIR, "env/server_rack_v6.1/server_rack_teleop.usd")
SERVER_RACK_SOURCE_USD = os.path.join(
    CUSTOM_ASSETS_DIR, "env/server_rack_v6.1/Server_Rack/server_rack_v5/configuration/server_rack_v5_base.usd"
)

FFW_LEFT_EE_LINK = "ffw_sg2_follower_arm_l_link7"
FFW_RIGHT_EE_LINK = "ffw_sg2_follower_arm_r_link7"
FFW_BASE_LINK = "world"

# 🔹 URDF에서 Pink IK 대상 외 관절 고정
FFW_FIXED_JOINTS = [
    "left_wheel_steer",
    "left_wheel_drive",
    "right_wheel_steer",
    "right_wheel_drive",
    "rear_wheel_steer",
    "rear_wheel_drive",
    "lift_joint",
    "head_joint1",
    "head_joint2",
]

# 🔹 기본 배치: 로봇 원점, 서버랙 +X (0.85m). Z lift 는 server_rack_teleop.usd 에 baked.
_BASE_RACK_POS = (0.85, 0.0, 0.0)
_BASE_RACK_ROT = (0.0, 0.0, 0.0, 1.0)  # Z 180° — 정면이 로봇 쪽
_BASE_ROBOT_ROT = (1.0, 0.0, 0.0, 0.0)
_BASE_BARCODE_POS = (0.50, 0.18, 1.18)

# 🔹 AR Start 시 정면(+Y)에 서버랙이 보이도록 씬 전체 Z 회전 (기본 +90°)
SCENE_YAW_DEG = float(os.environ.get("SCENE_YAW_DEG", "90"))


def _yaw_quat_wxyz(yaw_deg: float) -> tuple[float, float, float, float]:
    half = math.radians(yaw_deg) * 0.5
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _quat_mul(
    q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _rotate_xy_yaw(pos: tuple[float, float, float], yaw_deg: float) -> tuple[float, float, float]:
    x, y, z = pos
    rad = math.radians(yaw_deg)
    c, s = math.cos(rad), math.sin(rad)
    return (c * x - s * y, s * x + c * y, z)


_SCENE_YAW_QUAT = _yaw_quat_wxyz(SCENE_YAW_DEG)
SERVER_RACK_POS = _rotate_xy_yaw(_BASE_RACK_POS, SCENE_YAW_DEG)
SERVER_RACK_ROT = _quat_mul(_SCENE_YAW_QUAT, _BASE_RACK_ROT)
ROBOT_SPAWN_ROT = _quat_mul(_SCENE_YAW_QUAT, _BASE_ROBOT_ROT)
BARCODE_TARGET_POS = _rotate_xy_yaw(_BASE_BARCODE_POS, SCENE_YAW_DEG)

HAND_CAM_WIDTH = int(os.environ.get("HAND_CAM_WIDTH", os.environ.get("ROBOT_CAM_WIDTH", "256")))
HAND_CAM_HEIGHT = int(os.environ.get("HAND_CAM_HEIGHT", os.environ.get("ROBOT_CAM_HEIGHT", "160")))
HAND_CAM_FOCAL = float(os.environ.get("HAND_CAM_FOCAL_LENGTH", os.environ.get("ROBOT_CAM_FOCAL_LENGTH", "5.5")))

# 🔹 성공 조건: camera(오른손 cam 시야) | press(손가락 접촉, 레거시)
BARCODE_SUCCESS_MODE = os.environ.get("BARCODE_SUCCESS_MODE", "camera").strip().lower()
BARCODE_CAM_MARGIN = float(os.environ.get("BARCODE_CAM_MARGIN", "0.15"))
BARCODE_CAM_MIN_DEPTH = float(os.environ.get("BARCODE_CAM_MIN_DEPTH", "0.08"))
BARCODE_CAM_MAX_DEPTH = float(os.environ.get("BARCODE_CAM_MAX_DEPTH", "0.25"))
BARCODE_PRESS_DISTANCE = float(os.environ.get("BARCODE_PRESS_DISTANCE", "0.045"))
BARCODE_CAM_HOLD_TIME = float(os.environ.get("BARCODE_CAM_HOLD_TIME", "2.0"))
HAND_CAM_FRUSTUM_MAX_DEPTH = float(os.environ.get("HAND_CAM_FRUSTUM_MAX_DEPTH", "0.25"))


@configclass
class BarcodePressSceneCfg(InteractiveSceneCfg):
    """빈 환경 + 서버랙 + FFW_SG2."""

    server_rack = AssetBaseCfg(
        prim_path="/World/envs/env_.*/ServerRack",
        init_state=AssetBaseCfg.InitialStateCfg(pos=list(SERVER_RACK_POS), rot=list(SERVER_RACK_ROT)),
        spawn=UsdFileCfg(
            usd_path=SERVER_RACK_USD,
        ),
    )

    barcode_target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BarcodeTarget",
        init_state=RigidObjectCfg.InitialStateCfg(pos=list(BARCODE_TARGET_POS), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.SphereCfg(
            radius=0.025,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.2), opacity=0.15),
        ),
    )

    robot: ArticulationCfg = FFW_SG2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=FFW_SG2_CFG.init_state.replace(rot=ROBOT_SPAWN_ROT),
    )

    # 🔹 FFW_SG2 양손 D405 — USD camera_*_link 아래 hand_rgb 에 센서 spawn
    left_hand_cam = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/ffw_sg2_follower/arm_l_link7/camera_l_bottom_screw_frame/camera_l_link/hand_rgb",
        update_period=0.0,
        height=HAND_CAM_HEIGHT,
        width=HAND_CAM_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=HAND_CAM_FOCAL, clipping_range=(0.05, 10.0)),
        debug_vis=False,
    )
    right_hand_cam = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/ffw_sg2_follower/arm_r_link7/camera_r_bottom_screw_frame/camera_r_link/hand_rgb",
        update_period=0.0,
        height=HAND_CAM_HEIGHT,
        width=HAND_CAM_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=HAND_CAM_FOCAL, clipping_range=(0.05, 10.0)),
        debug_vis=False,
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    """Pink IK — 양팔 EE + 8-DOF 그리퍼."""

    upper_body_ik = PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=[f"arm_{side}_joint{i}" for side in ("l", "r") for i in range(1, 8)],
        hand_joint_names=[f"gripper_{side}_joint{i}" for side in ("l", "r") for i in range(1, 5)],
        target_eef_link_names={
            "left_wrist": FFW_LEFT_EE_LINK,
            "right_wrist": FFW_RIGHT_EE_LINK,
        },
        asset_name="robot",
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name=FFW_BASE_LINK,
            num_hand_joints=8,
            show_ik_warnings=False,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTask(
                    FFW_LEFT_EE_LINK,
                    position_cost=8.0,
                    orientation_cost=1.0,
                    lm_damping=12.0,
                    gain=0.5,
                ),
                FrameTask(
                    FFW_RIGHT_EE_LINK,
                    position_cost=8.0,
                    orientation_cost=1.0,
                    lm_damping=12.0,
                    gain=0.5,
                ),
                DampingTask(cost=0.5),
                NullSpacePostureTask(
                    cost=0.5,
                    lm_damping=1.0,
                    controlled_frames=[FFW_LEFT_EE_LINK, FFW_RIGHT_EE_LINK],
                    controlled_joints=[
                        "arm_l_joint1",
                        "arm_l_joint2",
                        "arm_l_joint4",
                        "arm_r_joint1",
                        "arm_r_joint2",
                        "arm_r_joint4",
                    ],
                ),
            ],
            fixed_input_tasks=[],
            xr_enabled=bool(carb.settings.get_settings().get("/app/xr/enabled")),
        ),
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(func=base_mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        left_hand_cam = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("left_hand_cam"), "data_type": "rgb", "normalize": False},
        )
        right_hand_cam = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("right_hand_cam"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


def _barcode_success_done_term() -> DoneTerm:
    """BARCODE_SUCCESS_MODE 에 따라 성공 종료 조건 선택."""
    if BARCODE_SUCCESS_MODE == "press":
        return DoneTerm(
            func=mdp.barcode_pressed,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "target_cfg": SceneEntityCfg("barcode_target"),
                "distance_threshold": BARCODE_PRESS_DISTANCE,
            },
        )
    return DoneTerm(
        func=mdp.barcode_in_hand_cam_hold,
        params={
            "camera_cfg": SceneEntityCfg("right_hand_cam"),
            "target_cfg": SceneEntityCfg("barcode_target"),
            "margin_frac": BARCODE_CAM_MARGIN,
            "min_depth": BARCODE_CAM_MIN_DEPTH,
            "max_depth": BARCODE_CAM_MAX_DEPTH,
            "hold_time_s": BARCODE_CAM_HOLD_TIME,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success: DoneTerm = _barcode_success_done_term()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_barcode_hold = EventTerm(func=mdp.reset_barcode_cam_hold, mode="reset")


@configclass
class BarcodePressFFWSG2EnvCfg(ManagerBasedRLEnvCfg):
    """FFW_SG2로 서버랙 바코드를 오른손 카메라로 맞추는 텔레옵 태스크."""

    scene: BarcodePressSceneCfg = BarcodePressSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    commands = None
    rewards = None
    curriculum = None

    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, 0.0),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    NUM_OPENXR_HAND_JOINTS = 26
    temp_urdf_dir = tempfile.gettempdir()
    idle_action = torch.zeros(22)

    def __post_init__(self):
        self.decimation = 6
        self.episode_length_s = 60.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = 2

        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
        )
        # URDF joint 이름은 Isaac Lab과 1:1 유지 (change_revolute_to_fixed 사용 시 Pink 매핑 깨짐)
        # 바퀴/헤드/리프트 잠금은 ffw_sg2_cfg.py actuator stiffness 로 처리

        self.actions.upper_body_ik.controller.urdf_path = temp_urdf_output_path
        self.actions.upper_body_ik.controller.mesh_path = temp_urdf_meshes_output_path

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        FfwSg2RetargeterCfg(
                            enable_visualization=True,
                            num_open_xr_hand_joints=2 * self.NUM_OPENXR_HAND_JOINTS,
                            sim_device=self.sim.device,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )
