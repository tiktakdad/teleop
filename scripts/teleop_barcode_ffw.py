#!/usr/bin/env python3
# Copyright 2025 ROBOTIS / teleop custom task
"""FFW_SG2 서버랙 바코드 프레스 — Quest 핸드트래킹 텔레옵 (Isaac Lab 2.3)."""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="FFW_SG2 barcode press teleoperation.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--teleop_device", type=str, default="handtracking")
parser.add_argument("--task", type=str, default="Isaac-BarcodePress-FFW-SG2-Abs-v0")
parser.add_argument("--sensitivity", type=float, default=1.0)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio (Pink IK).",
)
parser.add_argument(
    "--record",
    action="store_true",
    default=False,
    help="데이터 수집 모드: 텔레옵과 동일하게 동작하되 성공(3초 유지) 시 에피소드를 HDF5로 저장하고 reset.",
)
parser.add_argument(
    "--dataset_file",
    type=str,
    default="/workspace/user/datasets/dataset.hdf5",
    help="record 모드에서 데모를 저장할 HDF5 경로.",
)
parser.add_argument("--num_demos", type=int, default=0, help="record 모드에서 수집할 데모 수 (0=무한).")
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="성공 종료로 간주하기 위한 연속 성공 스텝 수.",
)
parser.add_argument(
    "--no_tag",
    action="store_true",
    default=False,
    help="record 모드 종료 시 HDF5 관절/액션 메타데이터 태깅을 건너뜁니다 (기본: 태깅 수행).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)
# XR handtracking 경로에서는 PhysX UI overlay가 keyboard 인터페이스를 요구해
# startup 에러를 낼 수 있으므로 해당 extension을 비활성화한다.
kit_args = (app_launcher_args.get("kit_args") or "").strip()
_disable_physx_ui = "--/exts/omni.physx.ui/enabled=false"
if _disable_physx_ui not in kit_args:
    app_launcher_args["kit_args"] = f"{kit_args} {_disable_physx_ui}".strip()
if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import omni.log

from isaaclab.devices import OpenXRDevice, Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg

import teleop_barcode_press  # noqa: F401 — gym 등록
from isaaclab_tasks.utils import parse_env_cfg
from teleop_barcode_press.barcode_press_env_cfg import (
    BARCODE_TARGET_POS,
    BARCODE_CAM_HOLD_TIME,
    BARCODE_CAM_MARGIN,
    BARCODE_CAM_MAX_DEPTH,
    BARCODE_CAM_MIN_DEPTH,
    HAND_CAM_FRUSTUM_MAX_DEPTH,
    REFERENCE_ROBOT_POS,
)
from teleop_barcode_press.mdp.barcode_cam import (
    barcode_in_frame_mask,
    get_barcode_cam_debug,
    get_barcode_cam_hold_time,
    get_barcode_cam_in_frame,
    update_barcode_cam_hold,
    reset_barcode_cam_hold,
)
from teleop_barcode_press.retargeters import FfwSg2Retargeter
from teleop_barcode_press.utils import HotReloadProxy
from teleop_barcode_press.utils.camera_preview_displays import CameraPreviewDisplays
from teleop_barcode_press.utils.hand_cam_frustum_vis import HandCamFrustumVisualizer
from teleop_barcode_press.utils.xr_task_hud import BarcodeXrHud

FFW_EE_LINKS = ["arm_l_link7", "arm_r_link7"]
FFW_HAND_JOINTS = ["gripper_l_joint1", "gripper_r_joint1"]
FFW_LIFT_JOINT = "lift_joint"
LIFT_MIN = float(os.environ.get("TELEOP_LIFT_MIN", "-0.5"))
LIFT_MAX = float(os.environ.get("TELEOP_LIFT_MAX", "0.0"))
LIFT_STEP = float(os.environ.get("TELEOP_LIFT_STEP", "0.05"))
LIFT_HANDTRACKING = os.environ.get("TELEOP_LIFT_HANDTRACKING", "1").lower() not in ("0", "false", "no")
LIFT_HAND_SCALE = float(os.environ.get("TELEOP_LIFT_HAND_SCALE", "1.0"))
LIFT_TARGET_VEL = float(os.environ.get("TELEOP_LIFT_TARGET_VEL", "0.35"))
# 리프트 모드: auto=오른손 높이 추종, manual=양손 주먹 쑥기(왼손 ↓ / 오른손 ↑). 기본 manual.
LIFT_MODE = os.environ.get("TELEOP_LIFT_MODE", "manual").strip().lower()
# 그리퍼 쑥힘 임계값(0~π/4.6). 이 이상이면 주먹으로 간주.
LIFT_FIST_THRESH = float(os.environ.get("TELEOP_LIFT_FIST_THRESH", "0.45"))
LIFT_FIST_VEL = float(os.environ.get("TELEOP_LIFT_FIST_VEL", "0.3"))
LIFT_START_AT_MIN = os.environ.get("TELEOP_LIFT_START_AT_MIN", "1").lower() not in ("0", "false", "no")
TELEOP_PINCH_CONTROL = os.environ.get("TELEOP_PINCH_CONTROL", "lift").strip().lower()
FFW_HEAD_PITCH_JOINT = os.environ.get("TELEOP_HEAD_PITCH_JOINT", "head_joint1").strip()
HEAD_PITCH_VEL = float(os.environ.get("TELEOP_HEAD_PITCH_VEL", "0.45"))
HEAD_UP_SIGN = float(os.environ.get("TELEOP_HEAD_UP_SIGN", "-1.0"))
# START 직후 손 트래킹 노이즈/초기 핀치 상태로 헤드가 튀는 현상 방지용 지연.
HEAD_PINCH_ARM_DELAY = float(os.environ.get("TELEOP_HEAD_PINCH_ARM_DELAY", "0.6"))
# 텔레옵 시작 직후 급격한 점프를 줄이기 위한 램프업 구간(제조사 운용 가이드 정합).
TELEOP_START_SLOW_SECONDS = float(os.environ.get("TELEOP_START_SLOW_SECONDS", "5.0"))
TELEOP_START_MIN_BLEND = float(os.environ.get("TELEOP_START_MIN_BLEND", "0.4"))
# true면 START/RESET/성공 시 scene hard reset(env.reset), false면 현재 포즈 유지 소프트 리셋.
TELEOP_HARD_RESET = os.environ.get("TELEOP_HARD_RESET", "0").lower() in ("1", "true", "yes")
# record 모드에서 새 데모 녹화가 시작될 때 간단한 시작 사운드(터미널 벨)를 울린다.
TELEOP_START_SOUND = os.environ.get("TELEOP_START_SOUND", "1").lower() not in ("0", "false", "no")
# 시작 사운드로 울릴 터미널 벨(BEL) 횟수.
TELEOP_START_SOUND_BEEPS = max(1, int(float(os.environ.get("TELEOP_START_SOUND_BEEPS", "2"))))


def play_start_sound() -> None:
    """record 모드 데모 시작 시 간단한 시작 사운드(ASCII BEL)를 울린다.

    호스트 터미널 스피커로 비프음이 난다(Quest 헤드셋으로는 전달되지 않음).
    USD 스테이지/렌더 파이프라인을 건드리지 않아 시뮬에 영향이 없다.
    """
    if not TELEOP_START_SOUND:
        return
    try:
        print("\a" * TELEOP_START_SOUND_BEEPS, end="", flush=True)
    except Exception:
        pass


def set_barcode_target_color(env, state: str | bool) -> None:
    """바코드 타겟 구의 색상을 실시간으로 업데이트합니다 (인식 시 초록색, 미인식 시 회색)."""
    try:
        from pxr import Usd, Gf
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if isinstance(state, bool):
            state = "contact" if state else "idle"
        if state == "success":
            color = Gf.Vec3f(0.1, 0.9, 0.35)
            opacity = 1.0
        elif state == "contact":
            color = Gf.Vec3f(1.0, 0.58, 0.05)
            opacity = 1.0
        else:
            color = Gf.Vec3f(1.0, 0.82, 0.05)
            opacity = 1.0

        for i in range(env.num_envs):
            prim_path = f"/World/envs/env_{i}/BarcodeTarget"
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue

            for p in Usd.PrimRange(prim):
                if p.HasAttribute("inputs:diffuseColor"):
                    p.GetAttribute("inputs:diffuseColor").Set(color)
                if p.HasAttribute("inputs:opacity"):
                    p.GetAttribute("inputs:opacity").Set(opacity)
    except Exception:
        pass


def set_barcode_target_visible(env, visible: bool) -> None:
    """record 모드에서 노란색 구가 녹화 영상에 들어가지 않도록 BarcodeTarget prim 가시성을 토글."""
    try:
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        for i in range(env.num_envs):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/BarcodeTarget")
            if not prim.IsValid():
                continue
            imageable = UsdGeom.Imageable(prim)
            if visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()
    except Exception:
        pass


def current_robot_action(env) -> torch.Tensor:
    """Build a Pink target action that holds the robot at its current authored pose."""
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies(FFW_EE_LINKS, preserve_order=True)
    hand_ids, _ = robot.find_joints(FFW_HAND_JOINTS, preserve_order=True)
    if len(body_ids) != 2 or len(hand_ids) != len(FFW_HAND_JOINTS):
        raise RuntimeError("Unable to resolve FFW end effectors or gripper joints for pose hold.")

    env_origin = env.scene.env_origins[0]
    left_pose = torch.cat((robot.data.body_pos_w[0, body_ids[0]] - env_origin, robot.data.body_quat_w[0, body_ids[0]]))
    right_pose = torch.cat((robot.data.body_pos_w[0, body_ids[1]] - env_origin, robot.data.body_quat_w[0, body_ids[1]]))
    hand_pos = robot.data.joint_pos[0, hand_ids]
    return torch.cat((left_pose, right_pose, hand_pos))


def _apply_startup_blend(raw_action: torch.Tensor, hold_action: torch.Tensor, elapsed_s: float) -> torch.Tensor:
    """Blend hand-tracking target from hold pose during startup and re-normalize wrist quaternions."""
    if TELEOP_START_SLOW_SECONDS <= 0.0:
        return raw_action
    alpha = max(TELEOP_START_MIN_BLEND, min(1.0, elapsed_s / TELEOP_START_SLOW_SECONDS))
    # Ensure device alignment (hold_action may be CPU if built before sim device init)
    hold_action = hold_action.to(device=raw_action.device, dtype=raw_action.dtype)
    blended = hold_action + alpha * (raw_action - hold_action)
    # Re-normalize quaternion slices without in-place assignment (safe inside inference_mode)
    # [L xyz(0:3), L quat(3:7), R xyz(7:10), R quat(10:14), gripper(14:16)]
    parts: list[torch.Tensor] = []
    cursor = 0
    for q_start in (3, 10):
        parts.append(blended[cursor:q_start])
        q = blended[q_start : q_start + 4]
        q_norm = torch.linalg.vector_norm(q)
        parts.append(q / q_norm if float(q_norm.item()) > 1.0e-8 else q)
        cursor = q_start + 4
    parts.append(blended[cursor:])
    return torch.cat(parts)


class LiftJointController:
    def __init__(self, env):
        self._env = env
        self._robot = env.scene["robot"]
        joint_ids, _ = self._robot.find_joints([FFW_LIFT_JOINT], preserve_order=True)
        if len(joint_ids) != 1:
            raise RuntimeError("Unable to resolve lift_joint for teleop lift control.")
        self._joint_id = joint_ids[0]
        self._hand_base_z: float | None = None
        self._lift_base_target: float | None = None
        limits_msg = ""
        try:
            limits = self._robot.data.soft_joint_pos_limits[0, self._joint_id]
            limits_msg = f" limits=[{limits[0].item():.3f}, {limits[1].item():.3f}]"
        except Exception:
            pass
        print(f"[teleop] resolved lift control joint '{FFW_LIFT_JOINT}' id={self._joint_id}{limits_msg}", flush=True)
        self.refresh_from_robot()

    def refresh_from_robot(self) -> None:
        if LIFT_START_AT_MIN:
            lower = None
            try:
                lower = float(self._robot.data.soft_joint_pos_limits[0, self._joint_id, 0].item())
            except Exception:
                pass
            # Use the safer of configured min and robot soft lower limit to avoid ground penetration.
            self._target = max(float(LIFT_MIN), lower) if lower is not None else float(LIFT_MIN)
        else:
            pos = self._robot.data.joint_pos[:, self._joint_id]
            self._target = float(pos[0].item())
        self.reset_hand_tracking()
        self.apply()

    def reset_hand_tracking(self) -> None:
        self._hand_base_z = None
        self._lift_base_target = None

    def current_position(self) -> float:
        return float(self._robot.data.joint_pos[0, self._joint_id].item())

    def move(self, delta: float) -> None:
        self._target = max(LIFT_MIN, min(LIFT_MAX, self._target + delta))
        self._hand_base_z = None
        self._lift_base_target = None
        self.apply()
        print(f"[teleop] lift_joint target={self._target:.3f} m", flush=True)

    def nudge(self, delta: float) -> None:
        """Continuous manual move (no log spam, resets hand-tracking baseline)."""
        self._target = max(LIFT_MIN, min(LIFT_MAX, self._target + delta))
        self._hand_base_z = None
        self._lift_base_target = None
        self.apply()

    def sync_to_hand_height(self, wrist_z: float, dt: float) -> None:
        if self._hand_base_z is None or self._lift_base_target is None:
            self._hand_base_z = wrist_z
            self._lift_base_target = self._target
            return
        desired = max(LIFT_MIN, min(LIFT_MAX, self._lift_base_target + (wrist_z - self._hand_base_z) * LIFT_HAND_SCALE))
        max_delta = max(0.0, LIFT_TARGET_VEL * dt)
        self._target += max(-max_delta, min(max_delta, desired - self._target))

    def apply(self) -> None:
        target = torch.full((self._env.num_envs, 1), self._target, dtype=torch.float32, device=self._env.device)
        self._robot.set_joint_position_target(target, joint_ids=[self._joint_id])
        try:
            self._robot.write_data_to_sim()
        except Exception:
            pass


class HeadPitchController:
    def __init__(self, env):
        self._env = env
        self._robot = env.scene["robot"]
        joint_ids, _ = self._robot.find_joints([FFW_HEAD_PITCH_JOINT], preserve_order=True)
        if len(joint_ids) != 1:
            raise RuntimeError(f"Unable to resolve head pitch joint: {FFW_HEAD_PITCH_JOINT}")
        self._joint_id = joint_ids[0]
        self._lower = None
        self._upper = None
        try:
            limits = self._robot.data.soft_joint_pos_limits[0, self._joint_id]
            self._lower = float(limits[0].item())
            self._upper = float(limits[1].item())
        except Exception:
            pass
        lim_msg = ""
        if self._lower is not None and self._upper is not None:
            lim_msg = f" limits=[{self._lower:.3f}, {self._upper:.3f}]"
        print(f"[teleop] resolved head control joint '{FFW_HEAD_PITCH_JOINT}' id={self._joint_id}{lim_msg}", flush=True)
        # Ensure first apply() uses a valid target and avoids 1-frame startup jump.
        self.refresh_from_robot()

    def _clamp(self, value: float) -> float:
        if self._lower is not None:
            value = max(self._lower, value)
        if self._upper is not None:
            value = min(self._upper, value)
        return value

    def refresh_from_robot(self) -> None:
        # Keep target aligned with current robot state.
        # Reset/play default head pose is defined in ffw_sg2_cfg.py init_state.
        pos = self._robot.data.joint_pos[:, self._joint_id]
        self._target = self._clamp(float(pos[0].item()))
        self.apply()

    def nudge(self, delta: float) -> None:
        self._target = self._clamp(self._target + delta)
        self.apply()

    def current_position(self) -> float:
        return float(self._robot.data.joint_pos[0, self._joint_id].item())

    def apply(self) -> None:
        target = torch.full((self._env.num_envs, 1), self._target, dtype=torch.float32, device=self._env.device)
        self._robot.set_joint_position_target(target, joint_ids=[self._joint_id])
        try:
            self._robot.write_data_to_sim()
        except Exception:
            pass


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    env_cfg.terminations.time_out = None

    record_mode = bool(args_cli.record)
    success_term = None
    if record_mode:
        # record 모드: 텔레옵과 동일하게 동작하되 성공을 메인 루프에서 수동 판정해 에피소드를 저장한다.
        # (record_demos.py 와 동일한 패턴) 성공 종료조건은 비활성화하고 success_term 으로 직접 확인.
        if hasattr(env_cfg.terminations, "success"):
            success_term = env_cfg.terminations.success
            env_cfg.terminations.success = None
        env_cfg.observations.policy.concatenate_terms = False
        try:
            from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
            from isaaclab.managers import DatasetExportMode

            output_dir = os.path.dirname(args_cli.dataset_file)
            output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            env_cfg.recorders = ActionStateRecorderManagerCfg()
            env_cfg.recorders.dataset_export_dir_path = output_dir
            env_cfg.recorders.dataset_filename = output_file_name
            env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
            print(f"[record] dataset → {args_cli.dataset_file} (num_demos={args_cli.num_demos})", flush=True)
        except Exception as exc:
            omni.log.error(f"Failed to configure recorders: {exc}")
            simulation_app.close()
            return
    # 핸드 트래킹(텔레옥) 모드에서는 성공 시 자동 reset이 핸드 트래킹을 끊으므로 종료조건을 비활성화한다.
    # (인디케이터 초록색 전환은 메인 루프에서 수동 타이머로 유지)
    elif "handtracking" in args_cli.teleop_device.lower():
        env_cfg.terminations.success = None

    if args_cli.xr:
        # FFW 양손 hand cam 은 유지 (debug_vis + record 연동)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    except Exception as exc:
        omni.log.error(f"Failed to create environment: {exc}")
        simulation_app.close()
        return

    should_reset = False
    is_handtracking = "handtracking" in args_cli.teleop_device.lower()
    # record 모드에서 START로 인한 reset은 텔레옵을 활성 상태로 유지해야 한다.
    pending_start_reset = False    # XR keeps stepping for camera/HUD updates, but robot targets remain authored until START.
    teleoperation_active = not is_handtracking
    env.teleoperation_active = teleoperation_active
    _debug_barcode = os.environ.get("BARCODE_DEBUG", "1").lower() not in ("0", "false", "no")
    _debug_interval = max(1, int(float(os.environ.get("BARCODE_DEBUG_INTERVAL", "60"))))
    _step_i = 0

    def reset_recording_instance() -> None:
        nonlocal should_reset
        should_reset = True
        print("Reset triggered - Environment will reset on next step")

    hold_action = None
    lift_controller: LiftJointController | None = None
    head_controller: HeadPitchController | None = None
    teleop_started_at: float | None = None
    head_pinch_ready = False

    def start_teleoperation() -> None:
        nonlocal teleoperation_active, should_reset, pending_start_reset, teleop_started_at, head_pinch_ready
        teleoperation_active = True
        env.teleoperation_active = True
        teleop_started_at = time.monotonic()
        head_pinch_ready = False
        if lift_controller is not None:
            lift_controller.reset_hand_tracking()
        # 🔹 텔레옵 시작 시점의 손 위치를 clutch 기준점으로 재설정 (hand scale 기준 원점)
        try:
            if hasattr(ffw_retargeter, "reset_clutch"):
                ffw_retargeter.reset_clutch()
                print("Clutch reference reset to current hand pose")
        except Exception as exc:
            print(f"[teleop] clutch reset skipped: {exc}", flush=True)
        # record 모드: START 시 환경/레코더 버퍼를 리셋해 초기상태부터 깨끗하게 녹화 시작 (활성 유지)
        if record_mode:
            if TELEOP_HARD_RESET:
                should_reset = True
                pending_start_reset = True
            else:
                pending_start_reset = False
        print("Teleoperation activated - following raw hand tracking targets")

    def stop_teleoperation() -> None:
        nonlocal teleoperation_active, hold_action, teleop_started_at, head_pinch_ready
        teleoperation_active = False
        env.teleoperation_active = False
        teleop_started_at = None
        head_pinch_ready = False
        hold_action = current_robot_action(env)
        if lift_controller is not None:
            lift_controller.reset_hand_tracking()
        print("Teleoperation deactivated")

    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_recording_instance,
    }

    sensitivity = args_cli.sensitivity
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(
            Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
        )
        for key, callback in teleoperation_callbacks.items():
            teleop_interface.add_callback(key, callback)
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(
            Se3SpaceMouseCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
        )
        for key, callback in teleoperation_callbacks.items():
            teleop_interface.add_callback(key, callback)
    elif "handtracking" in args_cli.teleop_device.lower():
        hand_cfg = env_cfg.teleop_devices.devices["handtracking"]
        # 🔹 record 모드: 손 트래킹 마커(붉은 점)가 녹화 영상/데이터셋에 찍히지 않도록 시각화 비활성.
        #    (트래킹·리타게팅은 그대로 동작)
        if record_mode:
            for _rt in hand_cfg.retargeters:
                if hasattr(_rt, "enable_visualization"):
                    _rt.enable_visualization = False
        # 🔹 실시간 코드 반영을 위한 HotReloadProxy 적용 (ffw_sg2_retargeter.py 수정 시 자동 리로드)
        ffw_retargeter = HotReloadProxy(
            module_name="teleop_barcode_press.retargeters.ffw_sg2_retargeter",
            class_name="FfwSg2Retargeter",
            cfg=hand_cfg.retargeters[0]
        )
        teleop_interface = OpenXRDevice(cfg=hand_cfg, retargeters=[ffw_retargeter])
        for key, callback in teleoperation_callbacks.items():
            teleop_interface.add_callback(key, callback)
    else:
        omni.log.error(f"Unsupported teleop device: {args_cli.teleop_device}")
        env.close()
        simulation_app.close()
        return

    # 리프트는 이제 키보드/메타퀄스트 컴트롤러가 아닌 씬 내 컨트롤 패널(가리키기/핀치)로 제어한다.

    print(f"Using teleop device: {teleop_interface}")

    last_countdown_sec = -1
    success_flash_steps = 0
    success_step_count = 0
    recorded_demo_count = 0

    # ── 증분 HDF5 태깅 셋업 ──
    # 컨테이너가 SIGKILL(exit 137) 등으로 강제 종료돼도 이미 export 된 데모의
    # 메타데이터가 보존되도록, 각 에피소드 export 직후 즉시 태깅한다.
    # 라이브 env가 필요한 관절/액션 이름 태깅은 최초 1회만 수행하고,
    # action→관절 공간 변환은 멱등 함수로 매번 호출한다(이미 변환된 데모는 건너뜀).
    _tagger = None
    _metadata_tagged = False
    if record_mode and not args_cli.no_tag:
        try:
            import sys as _sys

            _tagger_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "joint_targeter"
            )
            if _tagger_dir not in _sys.path:
                _sys.path.insert(0, _tagger_dir)
            from isaaclab_hdf5_tagger import IsaacLabHdf5Tagger as _tagger
        except Exception as exc:
            omni.log.error(f"[record] HDF5 태거 로드 실패 (태깅 비활성): {exc}")
            _tagger = None

    def _tag_after_export() -> None:
        """에피소드 export 직후 호출: 메타데이터 태깅 + action 관절 공간 변환(멱등)."""
        nonlocal _metadata_tagged
        if _tagger is None:
            return
        try:
            if not os.path.exists(args_cli.dataset_file):
                return
            if not _metadata_tagged:
                # 라이브 env가 있어야 가능한 관절/액션 이름 태깅 (최초 1회)
                _tagger.tag_joint_names(env, args_cli.dataset_file, robot_name="robot")
                _tagger.tag_action_info(env, args_cli.dataset_file)
                _metadata_tagged = True
            # action → 관절 공간 변환 (이미 변환된 데모는 건너뜀 → 멱등)
            _tagger.convert_actions_to_joint_space(
                args_cli.dataset_file, robot_name="robot", verbose=False
            )
        except Exception as exc:
            omni.log.error(f"[record] 증분 태깅 실패 (데이터는 보존됨): {exc}")

    env.reset()
    teleop_interface.reset()
    hold_action = current_robot_action(env)

    robot = env.scene["robot"]
    _ee_body_ids, _ = robot.find_bodies(FFW_EE_LINKS, preserve_order=True)
    _ee_cam_by_side = {
        "left": "left_hand_cam",
        "right": "right_hand_cam",
    }
    lift_controller = LiftJointController(env)
    if TELEOP_PINCH_CONTROL == "head":
        head_controller = HeadPitchController(env)
    # record 모드에서는 노란색 구·파란 레이저(시각화 인디케이터)를 제거해 녹화 영상에 들어가지 않게 한다.
    # (상위 카메라 디스플레이/네비게이션 패널은 유지)
    if record_mode:
        set_barcode_target_visible(env, False)
        frustum_vis = None
        frustum_vis_left = None
    else:
        set_barcode_target_color(env, False)
        # 🔹 양손 모두 파란 레이저/성공 시각화 (어느 손이든 바코드 인식 가능)
        frustum_vis = HandCamFrustumVisualizer(
            env,
            camera_name="right_hand_cam",
            margin_frac=BARCODE_CAM_MARGIN,
            min_depth=BARCODE_CAM_MIN_DEPTH,
            max_depth=HAND_CAM_FRUSTUM_MAX_DEPTH,
        )
        frustum_vis_left = HandCamFrustumVisualizer(
            env,
            camera_name="left_hand_cam",
            margin_frac=BARCODE_CAM_MARGIN,
            min_depth=BARCODE_CAM_MIN_DEPTH,
            max_depth=HAND_CAM_FRUSTUM_MAX_DEPTH,
        )
    xr_hud = BarcodeXrHud(BARCODE_CAM_HOLD_TIME, device=env.device) if args_cli.xr else None
    camera_previews = CameraPreviewDisplays(
        env,
        robot_pos=REFERENCE_ROBOT_POS,
        target_pos=BARCODE_TARGET_POS,
    )
    print(
        f"Teleoperation started. 바코드 {BARCODE_CAM_HOLD_TIME:.1f}s 연속 인식 시 성공. "
        "파란 선=양손 D405 정면. START=현재 손 위치에서 조작 시작. "
        f"리프트 모드={LIFT_MODE}, 핀치 제어={TELEOP_PINCH_CONTROL or 'none'}. "
        "Press 'R' to reset, STOP=일시정지."
    )

    # 시뮬레이션 스텝 강건성: 카메라 렌더 일시 글리치(env.reset 직후 빈 버퍼 등)
    # 같은 복구 가능한 예외는 해당 스텝만 건너뛰고 계속 진행한다. 강제 종료하지
    # 않으므로 카메라가 다시 정상 렌더되면 자동으로 복구된다.
    step_error_count = 0
    STEP_ERROR_LOG_INTERVAL = 30  # 에러 로그 폭주 방지용 출력 간격(스텝)

    while simulation_app.is_running():
        try:
            with torch.inference_mode():
                raw_action = teleop_interface.advance()

                # 핀치(집게) 기반 보조 제어: none="미사용", lift="리프트", head="헤드 피치"
                if teleoperation_active and is_handtracking and "ffw_retargeter" in locals():
                    left_pinch = float(getattr(ffw_retargeter, "latest_left_gripper", 0.0)) >= LIFT_FIST_THRESH
                    right_pinch = float(getattr(ffw_retargeter, "latest_right_gripper", 0.0)) >= LIFT_FIST_THRESH

                    if TELEOP_PINCH_CONTROL == "lift" and LIFT_HANDTRACKING and lift_controller is not None:
                        if LIFT_MODE == "auto" and hasattr(ffw_retargeter, "latest_right_wrist"):
                            lift_controller.sync_to_hand_height(float(ffw_retargeter.latest_right_wrist[2]), env.step_dt)
                        else:
                            delta = LIFT_FIST_VEL * env.step_dt
                            if right_pinch and not left_pinch:
                                lift_controller.nudge(delta)
                            elif left_pinch and not right_pinch:
                                lift_controller.nudge(-delta)
                    elif TELEOP_PINCH_CONTROL == "head" and head_controller is not None:
                        # START 직후에는 트래킹 안정화 + 중립(양손 비핀치) 확인 전까지 헤드 제어를 보류.
                        if not head_pinch_ready:
                            elapsed = (time.monotonic() - teleop_started_at) if teleop_started_at is not None else 0.0
                            if elapsed < HEAD_PINCH_ARM_DELAY:
                                pass
                            elif not left_pinch and not right_pinch:
                                head_pinch_ready = True
                            else:
                                pass
                        elif right_pinch ^ left_pinch:
                            delta = HEAD_PITCH_VEL * env.step_dt * max(0.1, abs(HEAD_UP_SIGN))
                            if right_pinch:
                                head_controller.nudge(delta * (1.0 if HEAD_UP_SIGN >= 0.0 else -1.0))
                            else:
                                head_controller.nudge(-delta * (1.0 if HEAD_UP_SIGN >= 0.0 else -1.0))

                if success_flash_steps > 0:
                    success_flash_steps -= 1

                selected_action = raw_action if teleoperation_active else hold_action
                if teleoperation_active and is_handtracking and hold_action is not None and teleop_started_at is not None:
                    elapsed_s = time.monotonic() - teleop_started_at
                    if elapsed_s < TELEOP_START_SLOW_SECONDS:
                        selected_action = _apply_startup_blend(selected_action, hold_action, elapsed_s)
                    else:
                        teleop_started_at = None

                actions = selected_action.unsqueeze(0) if selected_action.dim() == 1 else selected_action
                if actions.shape[0] != env.num_envs:
                    actions = actions.repeat(env.num_envs, 1)

                if lift_controller is not None:
                    lift_controller.apply()
                if head_controller is not None:
                    head_controller.apply()
                step_out = env.step(actions)
                if lift_controller is not None:
                    lift_controller.apply()
                if head_controller is not None:
                    head_controller.apply()
                terminated = step_out[2] if teleoperation_active else None

                # 🔹 termination 과 동일 로직으로 매 스텝 갱신 (로그/HUD용)
                in_frame = barcode_in_frame_mask(
                    env,
                    margin_frac=BARCODE_CAM_MARGIN,
                    min_depth=BARCODE_CAM_MIN_DEPTH,
                    max_depth=BARCODE_CAM_MAX_DEPTH,
                )
                _dbg_nav = get_barcode_cam_debug(env) or {}
                _per_cam = _dbg_nav.get("per_camera", {})

                # per-camera ray_hit OR 를 in_frame 에 다시 반영해 인디케이터/타이머 판정을 일치시킨다.
                _in_frame_from_cam = None
                for _info in _per_cam.values():
                    _ray_hit = _info.get("ray_hit")
                    if _ray_hit is None:
                        continue
                    _ray_hit_b = _ray_hit.to(dtype=torch.bool)
                    _in_frame_from_cam = _ray_hit_b if _in_frame_from_cam is None else (_in_frame_from_cam | _ray_hit_b)
                if _in_frame_from_cam is not None:
                    in_frame = _in_frame_from_cam

                # 성공 종료조건을 비활성화한 핸드 트래킹 모드에서는 홀드 타이머를 수동 누적
                # (3초 유지 시 인디케이터만 초록색으로 바뀌고 env.reset()은 하지 않음)
                if is_handtracking:
                    update_barcode_cam_hold(env, in_frame, BARCODE_CAM_HOLD_TIME, env.step_dt)
                hold_s = float(get_barcode_cam_hold_time(env)[0].item())
                in_frame_b = bool(in_frame[0].item())

                hold_done = hold_s >= BARCODE_CAM_HOLD_TIME
                laser_state = "success" if hold_done else "contact" if in_frame_b else "idle"
                # 🔹 바코드 타겟 구의 색상 업데이트 (record 모드에서는 인디케이터를 숨기므로 생략)
                if not record_mode:
                    set_barcode_target_color(env, laser_state)
                # 🔹 양손 레이저를 각 손의 인식 상태에 따라 개별 갱신

                def _hand_in(cam_name):
                    info = _per_cam.get(cam_name)
                    if info is None:
                        return in_frame_b
                    return bool(info["ray_hit"][0].item())

                def _hand_state(cam_name):
                    hin = _hand_in(cam_name)
                    if hold_done and hin:
                        return "success"
                    return "contact" if hin else "idle"

                if frustum_vis is not None:
                    frustum_vis.update(
                        tracking_active=_hand_in("right_hand_cam") and teleoperation_active,
                        laser_state=_hand_state("right_hand_cam"),
                    )
                if frustum_vis_left is not None:
                    frustum_vis_left.update(
                        tracking_active=_hand_in("left_hand_cam") and teleoperation_active,
                        laser_state=_hand_state("left_hand_cam"),
                    )
                if xr_hud is not None:
                    xr_hud.update(
                        teleop_active=teleoperation_active,
                        in_frame=in_frame_b,
                        hold_s=hold_s,
                        success_flash=success_flash_steps > 0,
                    )
                camera_previews.update()

                # 🔹 네비게이션 기준 손 선택: 로봇 양손 EE 중 barcode target 과 실제 거리(월드 좌표)가 더 가까운 손 우선
                _offset_cam = None
                _dist_m = 0.0
                if _dbg_nav and len(_ee_body_ids) == 2:
                    _target_w = env.scene["barcode_target"].data.root_pos_w[0]
                    _l_ee_w = robot.data.body_pos_w[0, _ee_body_ids[0]]
                    _r_ee_w = robot.data.body_pos_w[0, _ee_body_ids[1]]
                    _l_dist = float(torch.linalg.norm(_target_w - _l_ee_w).item())
                    _r_dist = float(torch.linalg.norm(_target_w - _r_ee_w).item())
                    _primary_cam = _ee_cam_by_side["left"] if _l_dist <= _r_dist else _ee_cam_by_side["right"]
                    _secondary_cam = _ee_cam_by_side["right"] if _primary_cam == _ee_cam_by_side["left"] else _ee_cam_by_side["left"]

                    for _cam_name in (_primary_cam, _secondary_cam):
                        _info = _per_cam.get(_cam_name)
                        if _info is None or "p_cam" not in _info or "dist" not in _info:
                            continue
                        _offset_cam = _info["p_cam"][0].detach().cpu().numpy()
                        _dist_m = float(_info["dist"][0].item())
                        break

                    if _offset_cam is None and "p_cam" in _dbg_nav:
                        # per_camera 누락 시(구버전/예외) 대표 metric 으로 폴백
                        _offset_cam = _dbg_nav["p_cam"][0].detach().cpu().numpy()
                        _dist_m = float(_dbg_nav["dist"][0].item())
                camera_previews.update_nav_panel(
                    offset_cam=_offset_cam,
                    distance_m=_dist_m,
                    hold_s=hold_s,
                    hold_time_s=BARCODE_CAM_HOLD_TIME,
                    in_frame=in_frame_b,
                    success=(laser_state == "success"),
                )

                _step_i += 1
                if _debug_barcode and _step_i % _debug_interval == 0:
                    dbg = get_barcode_cam_debug(env)
                    if dbg is not None:
                        msg = (
                            f"[teleop][dbg] active={teleoperation_active} in_frame={in_frame_b} "
                            f"hold={hold_s:.2f}s dist={dbg['dist'][0].item():.3f} depth={dbg['depth'][0].item():.3f} "
                            f"ray_hit={bool(dbg['ray_hit'][0].item())} ray_dist={dbg['ray_dist'][0].item():.3f} "
                            f"u={dbg['u'][0].item():.0f} v={dbg['v'][0].item():.0f} "
                            f"fov={bool(dbg['fov_in'][0].item())} pixel={bool(dbg['pixel_in'][0].item())} "
                            f"cone={bool(dbg['cone_in'][0].item())}"
                        )
                        # 🔹 양손 카메라 각각의 버퍼 유효성(valid)·ray_hit·dist 를 표시하여
                        # 어느 카메라가 비어 있는지(empty dict 반환) 진단한다.
                        for _cname, _cm in (dbg.get("per_camera", {}) or {}).items():
                            try:
                                msg += (
                                    f"\n    └ {_cname}: valid={bool(_cm['valid'][0].item())} "
                                    f"ray_hit={bool(_cm['ray_hit'][0].item())} dist={_cm['dist'][0].item():.3f} "
                                    f"depth={_cm['depth'][0].item():.3f}"
                                )
                            except Exception:
                                pass
                        
                        # 🔹 [비교 디버그 로그] 실시간 손 추적 좌표 vs 로봇 손 카메라(레이저 가이드) 좌표 비교 출력
                        comp_msg = ""
                        if "ffw_retargeter" in locals() and hasattr(ffw_retargeter, "latest_right_wrist"):
                            comp_msg += f"\n  👉 [VR Hand Target] L_Wrist: {list(ffw_retargeter.latest_left_wrist)}, R_Wrist: {list(ffw_retargeter.latest_right_wrist)}"
                        if "frustum_vis" in locals() and hasattr(frustum_vis, "latest_cam_pos"):
                            cam_pos_xyz = [round(x, 3) for x in frustum_vis.latest_cam_pos.tolist()]
                            forward_xyz = [round(x, 3) for x in frustum_vis.latest_forward_w.tolist()]
                            comp_msg += f"\n  👉 [Robot Hand Cam] Position: {cam_pos_xyz}, Direction: {forward_xyz}"
                        if lift_controller is not None:
                            comp_msg += f"\n  👉 [Lift] joint={FFW_LIFT_JOINT}, target={lift_controller._target:.3f}, actual={lift_controller.current_position():.3f}"
                        
                        if comp_msg:
                            msg += comp_msg
                            
                        print(msg, flush=True)
                        omni.log.info(msg)

                if teleoperation_active and in_frame_b and hold_s < BARCODE_CAM_HOLD_TIME:
                    sec = int(hold_s)
                    if sec != last_countdown_sec:
                        last_countdown_sec = sec
                        msg = f"[teleop] 바코드 인식 중… {hold_s:.1f}s / {BARCODE_CAM_HOLD_TIME:.1f}s"
                        print(msg, flush=True)
                        omni.log.info(msg)
                elif not in_frame_b:
                    last_countdown_sec = -1

                # 🔹 record 모드: 3초 연속 유지(성공) 시 에피소드 저장 후 reset → 다음 에피소드 녹화
                if record_mode and teleoperation_active:
                    if hold_s >= BARCODE_CAM_HOLD_TIME:
                        success_step_count += 1
                        if success_step_count >= args_cli.num_success_steps and not should_reset:
                            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                            env.recorder_manager.set_success_to_episodes(
                                [0],
                                torch.tensor([[True]], dtype=torch.bool, device=env.device),
                            )
                            env.recorder_manager.export_episodes([0])
                            # 강제 종료에도 보존되도록 export 직후 즉시 태깅한다.
                            _tag_after_export()
                            success_flash_steps = 30
                            should_reset = True
                            print(
                                f"[record] ✓ 바코드 {BARCODE_CAM_HOLD_TIME:.1f}s 유지 — 에피소드 저장",
                                flush=True,
                            )
                    else:
                        success_step_count = 0

                    exported = int(env.recorder_manager.exported_successful_episode_count)
                    if exported > recorded_demo_count:
                        recorded_demo_count = exported
                        print(f"[record] 저장된 데모: {recorded_demo_count}", flush=True)
                    if args_cli.num_demos > 0 and exported >= args_cli.num_demos:
                        print(f"[record] 목표 {args_cli.num_demos}개 수집 완료 — 종료", flush=True)
                        break

                if (
                    teleoperation_active
                    and terminated is not None
                    and hasattr(terminated, "any")
                    and terminated.any()
                ):
                    msg = f"[teleop] ✓ 양손 카메라 {BARCODE_CAM_HOLD_TIME:.1f}s 연속 인식 — 성공!"
                    print(msg, flush=True)
                    omni.log.info(msg)
                    success_flash_steps = 30
                    if xr_hud is not None:
                        xr_hud.update(
                            teleop_active=True,
                            in_frame=True,
                            hold_s=BARCODE_CAM_HOLD_TIME,
                            success_flash=True,
                        )
                    if TELEOP_HARD_RESET:
                        env.reset()
                    else:
                        reset_barcode_cam_hold(env)
                    hold_action = current_robot_action(env)
                    if lift_controller is not None:
                        lift_controller.refresh_from_robot()
                    if head_controller is not None:
                        head_controller.refresh_from_robot()
                    head_pinch_ready = False
                    teleoperation_active = False if is_handtracking else teleoperation_active
                    env.teleoperation_active = teleoperation_active
                    set_barcode_target_color(env, False)
                    last_countdown_sec = -1

                if should_reset:
                    if record_mode:
                        # 비-성공(수동 R/START) 시 진행 중 에피소드 버퍼를 버리고 새로 시작
                        env.recorder_manager.reset()
                        success_step_count = 0
                    if TELEOP_HARD_RESET:
                        env.reset()
                        if record_mode:
                            # reset 후에도 노란색 구가 녹화에 들어가지 않도록 다시 숨김
                            set_barcode_target_visible(env, False)
                    else:
                        reset_barcode_cam_hold(env)
                    hold_action = current_robot_action(env)
                    if lift_controller is not None:
                        lift_controller.refresh_from_robot()
                    if head_controller is not None:
                        head_controller.refresh_from_robot()
                    head_pinch_ready = False
                    if record_mode:
                        # record 모드: reset 후에도 텔레옵 핸드포인트를 계속 따라가도록 활성 유지
                        # (성공/START 어느 경로든 다음 에피소드를 바로 녹화)
                        teleoperation_active = True
                    else:
                        teleoperation_active = False if is_handtracking else teleoperation_active
                    env.teleoperation_active = teleoperation_active
                    pending_start_reset = False
                    if not record_mode:
                        set_barcode_target_color(env, False)
                    should_reset = False
                    last_countdown_sec = -1
                    print("Environment reset complete")
                    if record_mode:
                        # 새 데모 녹화가 시작됨을 알리는 시작 사운드
                        play_start_sound()
                        print("[record] ▶ 다음 데모 녹화 시작 — 시작 사운드", flush=True)

            # 정상 스텝 완료 → 에러 카운터 리셋
            step_error_count = 0
        except Exception as exc:
            step_error_count += 1
            if step_error_count == 1:
                # 첫 실패는 근본 원인 추적을 위해 전체 traceback을 남긴다.
                import traceback as _tb
                omni.log.error(
                    f"Error during simulation step: {exc}\n{_tb.format_exc()}"
                )
            elif step_error_count % STEP_ERROR_LOG_INTERVAL == 0:
                # 동일 글리치가 지속되면 일정 간격으로만 요약 로그를 남긴다.
                omni.log.warn(
                    f"Recoverable simulation step error x{step_error_count} (계속 진행): {exc}"
                )
            # 복구 가능한 글리치로 보고 이 스텝만 건너뛴 뒤 계속 진행한다.
            # (강제 종료하지 않음 → 카메라 렌더가 회복되면 자동으로 정상화)
            continue

    # ── HDF5 관절/액션 메타데이터 태깅 ──
    # record 모드로 수집한 데이터셋에 로봇 관절 이름과 state/action 메타데이터를
    # 기록한다. FFW 태스크의 action은 Pink IK(EEF pose)이고 state는 관절각이라
    # 표현이 다르므로, auto_align=True로 action을 obs/robot_joint_pos[t+1] 기반
    # 관절 공간으로 변환(원본은 actions_original 백업)해 state/action을 동일한
    # 관절 이름 체계로 정렬한다. 태깅 실패가 수집 데이터를 깨지 않도록 감싼다.
    if record_mode and not args_cli.no_tag:
        print(
            f"[record] HDF5 메타데이터 최종 태깅 → {args_cli.dataset_file}",
            flush=True,
        )
        _tag_after_export()
        print("[record] 최종 태깅 완료", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
