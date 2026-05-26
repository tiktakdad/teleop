#!/usr/bin/env python3
# Copyright 2025 ROBOTIS / teleop custom task
"""FFW_SG2 서버랙 바코드 프레스 — Quest 핸드트래킹 텔레옵 (Isaac Lab 2.3)."""

from __future__ import annotations

import argparse
import os
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
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)
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
    BARCODE_CAM_HOLD_TIME,
    BARCODE_CAM_MARGIN,
    BARCODE_CAM_MAX_DEPTH,
    BARCODE_CAM_MIN_DEPTH,
    HAND_CAM_FRUSTUM_MAX_DEPTH,
)
from teleop_barcode_press.mdp.barcode_cam import (
    barcode_in_frame_mask,
    get_barcode_cam_debug,
    get_barcode_cam_hold_time,
    get_barcode_cam_in_frame,
)
from teleop_barcode_press.retargeters import FfwSg2Retargeter
from teleop_barcode_press.utils.hand_cam_frustum_vis import HandCamFrustumVisualizer
from teleop_barcode_press.utils.xr_task_hud import BarcodeXrHud


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    env_cfg.terminations.time_out = None

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
    # 🔹 XR 도 Connect 직후 step 이 돌아야 카메라/frustum/인식이 갱신됨 (START 대기 시 정지 문제)
    teleoperation_active = "handtracking" not in args_cli.teleop_device.lower()
    env.teleoperation_active = teleoperation_active
    _debug_barcode = os.environ.get("BARCODE_DEBUG", "1").lower() not in ("0", "false", "no")
    _debug_interval = max(1, int(float(os.environ.get("BARCODE_DEBUG_INTERVAL", "60"))))
    _step_i = 0

    def reset_recording_instance() -> None:
        nonlocal should_reset
        should_reset = True
        print("Reset triggered - Environment will reset on next step")

    def start_teleoperation() -> None:
        nonlocal teleoperation_active
        teleoperation_active = True
        env.teleoperation_active = True
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        nonlocal teleoperation_active
        teleoperation_active = False
        env.teleoperation_active = False
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
        ffw_retargeter = FfwSg2Retargeter(hand_cfg.retargeters[0])
        teleop_interface = OpenXRDevice(cfg=hand_cfg, retargeters=[ffw_retargeter])
        for key, callback in teleoperation_callbacks.items():
            teleop_interface.add_callback(key, callback)
    else:
        omni.log.error(f"Unsupported teleop device: {args_cli.teleop_device}")
        env.close()
        simulation_app.close()
        return

    print(f"Using teleop device: {teleop_interface}")

    last_countdown_sec = -1
    success_flash_steps = 0

    env.reset()
    teleop_interface.reset()
    frustum_vis = HandCamFrustumVisualizer(
        env,
        camera_name="right_hand_cam",
        margin_frac=BARCODE_CAM_MARGIN,
        min_depth=BARCODE_CAM_MIN_DEPTH,
        max_depth=HAND_CAM_FRUSTUM_MAX_DEPTH,
    )
    xr_hud = BarcodeXrHud(BARCODE_CAM_HOLD_TIME, device=env.device) if args_cli.xr else None
    idle_action = env_cfg.idle_action.to(env.device)
    print(
        f"Teleoperation started. 바코드 {BARCODE_CAM_HOLD_TIME:.1f}s 연속 인식 시 성공. "
        "파란 frustum=오른손 D405 시야. Press 'R' to reset, STOP=일시정지."
    )

    while simulation_app.is_running():
        try:
            with torch.inference_mode():
                action = teleop_interface.advance()

                if success_flash_steps > 0:
                    success_flash_steps -= 1

                # 🔹 teleoperation_active 와 무관하게 항상 advance() 가 반환한 유효한 action 을 사용 (zeros 대입 시 OSQP LDL 에러 방지)
                actions = action.unsqueeze(0) if action.dim() == 1 else action
                if actions.shape[0] != env.num_envs:
                    actions = actions.repeat(env.num_envs, 1)

                step_out = env.step(actions)
                terminated = step_out[2] if teleoperation_active else None

                # 🔹 termination 과 동일 로직으로 매 스텝 갱신 (로그/HUD용)
                in_frame = barcode_in_frame_mask(
                    env,
                    margin_frac=BARCODE_CAM_MARGIN,
                    min_depth=BARCODE_CAM_MIN_DEPTH,
                    max_depth=BARCODE_CAM_MAX_DEPTH,
                )
                hold_s = float(get_barcode_cam_hold_time(env)[0].item())
                in_frame_b = bool(in_frame[0].item())

                frustum_vis.update(tracking_active=in_frame_b and teleoperation_active)
                if xr_hud is not None:
                    xr_hud.update(
                        teleop_active=teleoperation_active,
                        in_frame=in_frame_b,
                        hold_s=hold_s,
                        success_flash=success_flash_steps > 0,
                    )

                _step_i += 1
                if _debug_barcode and _step_i % _debug_interval == 0:
                    dbg = get_barcode_cam_debug(env)
                    if dbg is not None:
                        msg = (
                            f"[teleop][dbg] active={teleoperation_active} in_frame={in_frame_b} "
                            f"hold={hold_s:.2f}s dist={dbg['dist'][0].item():.3f} depth={dbg['depth'][0].item():.3f} "
                            f"u={dbg['u'][0].item():.0f} v={dbg['v'][0].item():.0f} "
                            f"fov={bool(dbg['fov_in'][0].item())} pixel={bool(dbg['pixel_in'][0].item())} "
                            f"cone={bool(dbg['cone_in'][0].item())}"
                        )
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

                if (
                    teleoperation_active
                    and terminated is not None
                    and hasattr(terminated, "any")
                    and terminated.any()
                ):
                    msg = f"[teleop] ✓ 오른손 카메라 {BARCODE_CAM_HOLD_TIME:.1f}s 연속 인식 — 성공!"
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
                    env.reset()
                    last_countdown_sec = -1

                if should_reset:
                    env.reset()
                    should_reset = False
                    last_countdown_sec = -1
                    print("Environment reset complete")
        except Exception as exc:
            omni.log.error(f"Error during simulation step: {exc}")
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
