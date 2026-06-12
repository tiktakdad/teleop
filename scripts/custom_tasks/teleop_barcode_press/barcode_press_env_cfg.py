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
# reference.usd 안에서 노란색 바코드 태그(prim 이름 = "location_barcode") Mesh 를 자동 탐지하지 못할 때 사용하는 폴백 좌표.
_FALLBACK_BARCODE_TARGET_POS = (0.3542, 0.1996, 1.5855)
# 🔹 성공 판정 대상이 되는 타겟 바코드 prim 이름 (서버랙 모델 변경 정합). location_barcode02 등과 구분된다.
_BARCODE_TARGET_PRIM_NAME = "location_barcode"
# 이름에 아래 키워드가 들어간 prim 을 바코드 태그로 간주 (서버랙 모델이 바뀌어도 자동 추적)
_BARCODE_PRIM_KEYWORDS = ("location_barcode", "barcodeplane", "barcode", "device_barc")

HAND_CAM_WIDTH = int(os.environ.get("HAND_CAM_WIDTH", "256"))
HAND_CAM_HEIGHT = int(os.environ.get("HAND_CAM_HEIGHT", "160"))
HAND_CAM_FOCAL = float(os.environ.get("HAND_CAM_FOCAL_LENGTH", "5.5"))
# 헤드(POV) 카메라는 데이터셋 영상용 고해상도(1280x720) 기본값을 사용.
# HEAD_CAM_WIDTH/HEIGHT > ROBOT_CAM_WIDTH/HEIGHT(.env 호환) > 1280x720 순으로 결정.
HEAD_CAM_WIDTH = int(os.environ.get("HEAD_CAM_WIDTH", os.environ.get("ROBOT_CAM_WIDTH", "1280")))
HEAD_CAM_HEIGHT = int(os.environ.get("HEAD_CAM_HEIGHT", os.environ.get("ROBOT_CAM_HEIGHT", "720")))
HEAD_CAM_FOCAL = float(os.environ.get("HEAD_CAM_FOCAL_LENGTH", "6.0"))

# 🔹 성공 조건: camera(오른손 cam 시야) | press(손가락 접촉, 레거시)
BARCODE_SUCCESS_MODE = os.environ.get("BARCODE_SUCCESS_MODE", "camera").strip().lower()
BARCODE_CAM_MARGIN = float(os.environ.get("BARCODE_CAM_MARGIN", "0.08"))
BARCODE_CAM_MIN_DEPTH = float(os.environ.get("BARCODE_CAM_MIN_DEPTH", "0.08"))
BARCODE_CAM_MAX_DEPTH = float(os.environ.get("BARCODE_CAM_MAX_DEPTH", "0.6"))
BARCODE_PRESS_DISTANCE = float(os.environ.get("BARCODE_PRESS_DISTANCE", "0.045"))
BARCODE_TARGET_RADIUS = float(os.environ.get("BARCODE_TARGET_RADIUS", "0.045"))
BARCODE_CAM_HOLD_TIME = float(os.environ.get("BARCODE_CAM_HOLD_TIME", "3.0"))
HAND_CAM_FRUSTUM_MAX_DEPTH = float(os.environ.get("HAND_CAM_FRUSTUM_MAX_DEPTH", "0.6"))

# 🔹 Pink IK FrameTask 가중치 (팔 도달 거리 튜닝용).
# orientation_cost 를 낮추면 IK 가 손목 자세보다 위치 추종을 우선해 팔을 더 멀리 뻗는다.
IK_POSITION_COST = float(os.environ.get("TELEOP_IK_POSITION_COST", "8.0"))
IK_ORIENTATION_COST = float(os.environ.get("TELEOP_IK_ORIENTATION_COST", "1.0"))
IK_LM_DAMPING = float(os.environ.get("TELEOP_IK_LM_DAMPING", "5.0"))
IK_GAIN = float(os.environ.get("TELEOP_IK_GAIN", "0.5"))


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


def _find_barcode_prim(stage):
    """타겟 바코드(location_barcode) prim 을 우선 탐색하고, 없으면 키워드 매칭으로 탐색."""
    from pxr import UsdGeom

    # 🔹 타겟 바코드는 이름이 정확히 일치하는 prim 을 우선한다 (location_barcode02 등과 구분).
    exact = _find_prim_by_name(stage, _BARCODE_TARGET_PRIM_NAME)
    if exact is not None and exact.IsValid():
        return exact

    fallback = None
    for prim in stage.Traverse():
        name = prim.GetName().lower()
        if not any(key in name for key in _BARCODE_PRIM_KEYWORDS):
            continue
        if prim.IsA(UsdGeom.Mesh):
            return prim
        if fallback is None and UsdGeom.Imageable(prim):
            fallback = prim
    return fallback


def _read_barcode_world_center(usd_path: str) -> tuple[float, float, float]:
    """reference.usd 에서 바코드 태그 Mesh 를 찾아 월드 중심 좌표를 계산."""
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Unable to open reference USD: {usd_path}")

    prim = _find_barcode_prim(stage)
    if prim is None or not prim.IsValid():
        raise RuntimeError(f"No barcode prim found in {usd_path}")

    xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get() if mesh else None
    if points:
        world_pts = [xform.Transform(Gf.Vec3d(p[0], p[1], p[2])) for p in points]
        cx = sum(w[0] for w in world_pts) / len(world_pts)
        cy = sum(w[1] for w in world_pts) / len(world_pts)
        cz = sum(w[2] for w in world_pts) / len(world_pts)
    else:
        t = xform.ExtractTranslation()
        cx, cy, cz = float(t[0]), float(t[1]), float(t[2])
    return (float(cx), float(cy), float(cz))


def _barcode_target_pos() -> tuple[float, float, float]:
    override = os.environ.get("BARCODE_TARGET_POS", "").strip()
    if override:
        try:
            parts = tuple(float(v) for v in override.replace(",", " ").split())
            if len(parts) == 3:
                carb.log_info(f"Using BARCODE_TARGET_POS override={parts}")
                return parts
        except ValueError:
            carb.log_warn(f"Invalid BARCODE_TARGET_POS override={override!r}; auto-detecting instead")
    try:
        pos = _read_barcode_world_center(REFERENCE_SCENE_USD)
        carb.log_info(f"Auto-detected barcode target pos={pos} from {REFERENCE_SCENE_USD}")
        return pos
    except Exception as exc:
        carb.log_warn(f"Barcode auto-detection failed ({exc}); using fallback {_FALLBACK_BARCODE_TARGET_POS}")
        return _FALLBACK_BARCODE_TARGET_POS


REFERENCE_ROBOT_POS, REFERENCE_ROBOT_ROT = _reference_robot_pose()
# 서버랙 모델 변경에 대응: reference.usd 에서 노란 바코드 태그 위치를 자동 탐지해 구를 배치
BARCODE_TARGET_POS = _barcode_target_pos()
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
            radius=BARCODE_TARGET_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.82, 0.05), opacity=1.0),
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
    head_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ReferenceScene/FFW_SG2/ffw_sg2_follower/head_link2/head_rgb",
        update_period=0.0,
        height=HEAD_CAM_HEIGHT,
        width=HEAD_CAM_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=HEAD_CAM_FOCAL, clipping_range=(0.05, 12.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.08, 0.0, 0.02), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
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
        # FFW gripper joints 2-4 mimic joint1 in USD, so teleop should only command each master joint.
        hand_joint_names=["gripper_l_joint1", "gripper_r_joint1"],
        target_eef_link_names={
            "left_wrist": FFW_LEFT_EE_LINK,
            "right_wrist": FFW_RIGHT_EE_LINK,
        },
        asset_name="robot",
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name=FFW_BASE_LINK,
            num_hand_joints=2,
            show_ik_warnings=False,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTask(
                    FFW_LEFT_EE_LINK,
                    position_cost=IK_POSITION_COST,
                    orientation_cost=IK_ORIENTATION_COST,
                    lm_damping=IK_LM_DAMPING,
                    gain=IK_GAIN,
                ),
                FrameTask(
                    FFW_RIGHT_EE_LINK,
                    position_cost=IK_POSITION_COST,
                    orientation_cost=IK_ORIENTATION_COST,
                    lm_damping=IK_LM_DAMPING,
                    gain=IK_GAIN,
                ),
                DampingTask(cost=0.5),
                NullSpacePostureTask(
                    cost=0.5,
                    lm_damping=1.0,
                    controlled_frames=[FFW_LEFT_EE_LINK, FFW_RIGHT_EE_LINK],
                    controlled_joints=[
                        "arm_l_joint1",
                        "arm_l_joint2",
                        "arm_l_joint3",
                        "arm_l_joint4",
                        "arm_l_joint5",
                        "arm_l_joint6",
                        "arm_l_joint7",
                        "arm_r_joint1",
                        "arm_r_joint2",
                        "arm_r_joint3",
                        "arm_r_joint4",
                        "arm_r_joint5",
                        "arm_r_joint6",
                        "arm_r_joint7",
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
    # 폴백: 렌더 버퍼가 비어 있을 때 0-패딩 텐서를 반환한다.
    # 주의 — sensor.data 접근은 동일한 버퍼 갱신(_update_buffers_impl) 크래시를
    # 재유발하므로(렌더 리소스가 비어 있을 때 [720,1280,3] expand 실패) 절대
    # 건드리지 않고, 해상도는 센서 cfg(고정 width/height)에서 직접 읽는다.
    sensor = env.scene[sensor_cfg.name]
    try:
        h = int(sensor.cfg.height)
        w = int(sensor.cfg.width)
    except Exception:
        h, w = 0, 0
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
        head_cam = ObsTerm(
            func=safe_image,
            params={"sensor_cfg": SceneEntityCfg("head_cam"), "data_type": "rgb", "normalize": False},
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
            # camera_cfg 미지정 → 양손(right/left hand cam) 모두 검사
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
    randomize_barcode_planes = EventTerm(func=mdp.randomize_barcode_planes_on_front_cover, mode="reset")
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
        # Robotis AI Worker: 100 Hz low-level joint loop, 15 Hz dataset/policy FPS.
        # Keep env control at 50 Hz for teleop responsiveness; RECORD_FPS throttles HDF5 samples to 15 Hz.
        self.decimation = 2
        self.episode_length_s = 60.0
        self.sim.dt = 1 / 100
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
