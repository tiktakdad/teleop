# Copyright 2025 ROBOTIS / teleop custom task
"""서버랙 바코드 프레스 텔레옵 환경 (FFW_SG2 + custom server rack, Isaac Lab 2.3)."""

from __future__ import annotations

from functools import lru_cache
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
from .ffw_sg2_cfg import FFW_SG2_CFG, FFW_SG2_USD
from .retargeters.ffw_sg2_retargeter import FfwSg2RetargeterCfg

CUSTOM_ASSETS_DIR = os.environ.get("CUSTOM_ASSETS_DIR", "/workspace/user/custom_assets")
REFERENCE_SCENE_USD = os.path.join(CUSTOM_ASSETS_DIR, "scene/reference.usd")
REFERENCE_ROBOT_PRIM_PATH = "/World/FFW_SG2"
XR_FORWARD_YAW_OFFSET_DEG = -90.0

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

# Fallback for old/broken reference scenes. Normal operation reads this from reference.usd.
_FALLBACK_REFERENCE_ROBOT_POS = (1.4304832140901735, 0.0, 0.0)
_FALLBACK_REFERENCE_ROBOT_ROT = (0.0, 0.0, 0.0, 1.0)
BARCODE_TARGET_POS = (0.3516863028144265, 0.1996206657250342, 1.7259674316986728)

HAND_CAM_WIDTH = int(os.environ.get("HAND_CAM_WIDTH", "256"))
HAND_CAM_HEIGHT = int(os.environ.get("HAND_CAM_HEIGHT", "160"))
HAND_CAM_FOCAL = float(os.environ.get("HAND_CAM_FOCAL_LENGTH", "5.5"))

# 🔹 성공 조건: camera(오른손 cam 시야) | press(손가락 접촉, 레거시)
BARCODE_SUCCESS_MODE = os.environ.get("BARCODE_SUCCESS_MODE", "camera").strip().lower()
BARCODE_CAM_MARGIN = float(os.environ.get("BARCODE_CAM_MARGIN", "0.08"))
BARCODE_CAM_MIN_DEPTH = float(os.environ.get("BARCODE_CAM_MIN_DEPTH", "0.08"))
BARCODE_CAM_MAX_DEPTH = float(os.environ.get("BARCODE_CAM_MAX_DEPTH", "0.25"))
BARCODE_PRESS_DISTANCE = float(os.environ.get("BARCODE_PRESS_DISTANCE", "0.045"))
BARCODE_CAM_HOLD_TIME = float(os.environ.get("BARCODE_CAM_HOLD_TIME", "2.0"))
HAND_CAM_FRUSTUM_MAX_DEPTH = float(os.environ.get("HAND_CAM_FRUSTUM_MAX_DEPTH", "0.25"))


def _yaw_quat_wxyz(yaw_deg: float) -> tuple[float, float, float, float]:
    import math

    half = math.radians(yaw_deg) * 0.5
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _quat_mul(
    q1: tuple[float, float, float, float],
    q2: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _find_prim_by_name(stage, prim_name: str):
    for prim in stage.Traverse():
        if prim.GetName() == prim_name:
            return prim
    return None


@lru_cache(maxsize=8)
def _read_reference_prim_pose(
    usd_path: str,
    prim_path: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Read a prim's authored world pose from a USD file as Isaac Lab pos and wxyz quat."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Unable to open reference USD: {usd_path}")

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        prim = _find_prim_by_name(stage, prim_path.rsplit("/", 1)[-1])
    if prim is None or not prim.IsValid():
        raise RuntimeError(f"Unable to find robot prim '{prim_path}' in {usd_path}")

    xformable = UsdGeom.Xformable(prim)
    transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = transform.ExtractTranslation()
    rotation = transform.ExtractRotationQuat()
    rotation_imag = rotation.GetImaginary()

    pos = tuple(float(translation[i]) for i in range(3))
    rot = (
        float(rotation.GetReal()),
        float(rotation_imag[0]),
        float(rotation_imag[1]),
        float(rotation_imag[2]),
    )
    return pos, rot


def _reference_robot_pose() -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    try:
        pos, rot = _read_reference_prim_pose(REFERENCE_SCENE_USD, REFERENCE_ROBOT_PRIM_PATH)
        carb.log_info(
            f"Using robot pose from {REFERENCE_SCENE_USD}:{REFERENCE_ROBOT_PRIM_PATH} "
            f"pos={pos}, rot={rot}"
        )
        return pos, rot
    except Exception as exc:
        carb.log_warn(
            f"Falling back to hard-coded robot pose because reference pose extraction failed: {exc}"
        )
        return _FALLBACK_REFERENCE_ROBOT_POS, _FALLBACK_REFERENCE_ROBOT_ROT


REFERENCE_ROBOT_POS, REFERENCE_ROBOT_ROT = _reference_robot_pose()
XR_ANCHOR_ROT = _quat_mul(REFERENCE_ROBOT_ROT, _yaw_quat_wxyz(XR_FORWARD_YAW_OFFSET_DEG))
carb.log_info(
    f"Using XR anchor pose pos={REFERENCE_ROBOT_POS}, rot={XR_ANCHOR_ROT} "
    f"(robot_rot={REFERENCE_ROBOT_ROT}, yaw_offset_deg={XR_FORWARD_YAW_OFFSET_DEG})"
)


def spawn_reference_scene_for_robot(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the authored scene before the articulation resolves its nested prim."""
    scene_prim_path = prim_path.removesuffix("/FFW_SG2")
    return sim_utils.spawn_from_usd(
        scene_prim_path,
        cfg,
        translation=(0.0, 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
        **kwargs,
    )


@configclass
class BarcodePressSceneCfg(InteractiveSceneCfg):
    """Authored server-rack and robot scene with teleoperation entities attached."""

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
        prim_path="{ENV_REGEX_NS}/ReferenceScene/FFW_SG2",
        spawn=FFW_SG2_CFG.spawn.replace(
            usd_path=REFERENCE_SCENE_USD,
            func=spawn_reference_scene_for_robot,
        ),
        init_state=FFW_SG2_CFG.init_state.replace(pos=REFERENCE_ROBOT_POS, rot=REFERENCE_ROBOT_ROT),
    )

    # The USD authors each D405 camera_link as a physical frame with +X forward.
    # Create only the render sensor at that origin and preserve the camera_link forward axis.
    left_hand_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ReferenceScene/FFW_SG2/ffw_sg2_follower/arm_l_link7/camera_l_bottom_screw_frame/camera_l_link/hand_rgb",
        update_period=0.0,
        height=HAND_CAM_HEIGHT,
        width=HAND_CAM_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=HAND_CAM_FOCAL, clipping_range=(0.05, 10.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        update_latest_camera_pose=True,
        debug_vis=False,
    )
    right_hand_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ReferenceScene/FFW_SG2/ffw_sg2_follower/arm_r_link7/camera_r_bottom_screw_frame/camera_r_link/hand_rgb",
        update_period=0.0,
        height=HAND_CAM_HEIGHT,
        width=HAND_CAM_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=HAND_CAM_FOCAL, clipping_range=(0.05, 10.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        update_latest_camera_pose=True,
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


def safe_image(env, sensor_cfg, data_type="rgb", normalize=False):
    """헤드리스/XR 구동 조건에 따라 비어 있을 수 있는 카메라 RGB 버퍼를 안전하게 0-패딩 텐서로 대체해 시뮬레이터 크래시를 방지합니다."""
    try:
        val = base_mdp.image(env, sensor_cfg, data_type=data_type, normalize=normalize)
        if val is not None and val.numel() > 0:
            return val
    except Exception:
        pass
    sensor = env.scene[sensor_cfg.name]
    h, w = sensor.data.image_shape
    c = 3 if data_type == "rgb" else 1
    device = env.device
    # ObservationManager가 단일 env일 때 (H, W, C) 형태를 기대하는 케이스가 있어
    # env 개수에 따라 배치 차원을 조건부로 맞춥니다.
    if getattr(env, "num_envs", 1) == 1:
        return torch.zeros((h, w, c), dtype=torch.float32, device=device)
    return torch.zeros((env.num_envs, h, w, c), dtype=torch.float32, device=device)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(func=base_mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        left_hand_cam = ObsTerm(
            func=safe_image,
            params={"sensor_cfg": SceneEntityCfg("left_hand_cam"), "data_type": "rgb", "normalize": False},
        )
        right_hand_cam = ObsTerm(
            func=safe_image,
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
        anchor_pos=REFERENCE_ROBOT_POS,
        anchor_rot=XR_ANCHOR_ROT,
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
            FFW_SG2_USD, self.temp_urdf_dir, force_conversion=True
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
