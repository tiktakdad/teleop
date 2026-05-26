# Copyright 2025 ROBOTIS / teleop custom task
"""손 카메라 바코드 시야 판정·홀드 타이머 공용 로직."""

from __future__ import annotations

import math
import os

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_HOLD_TIME_KEY = "_barcode_cam_hold_time"
_IN_FRAME_KEY = "_barcode_cam_in_frame"
_DEBUG_KEY = "_barcode_cam_debug"

_USE_CONE = os.environ.get("BARCODE_USE_CONE", "1").lower() not in ("0", "false", "no")
_CONE_HALF_ANGLE_DEG = float(os.environ.get("BARCODE_CONE_HALF_ANGLE_DEG", "35"))


def _get_hold_tensor(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, _HOLD_TIME_KEY) or getattr(env, _HOLD_TIME_KEY) is None:
        setattr(env, _HOLD_TIME_KEY, torch.zeros(env.num_envs, device=env.device))
    return getattr(env, _HOLD_TIME_KEY)


def _compute_view_metrics(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    margin_frac: float,
    min_depth: float,
    max_depth: float,
) -> dict[str, torch.Tensor]:
    """픽셀·콘 각도 판정용 중간값."""
    camera = env.scene[camera_cfg.name]
    target = env.scene[target_cfg.name]

    empty = {
        "valid": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        "pixel_in": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        "cone_in": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        "depth": torch.zeros(env.num_envs, device=env.device),
        "u": torch.zeros(env.num_envs, device=env.device),
        "v": torch.zeros(env.num_envs, device=env.device),
    }

    cam_pos = camera.data.pos_w
    cam_quat = camera.data.quat_w_ros
    intrinsics = camera.data.intrinsic_matrices
    if cam_pos is None or cam_quat is None or intrinsics is None:
        return empty

    # 🔹 target 과 동일하게 env origin 기준으로 맞춤
    cam_pos = cam_pos - env.scene.env_origins
    target_pos = target.data.root_pos_w - env.scene.env_origins
    rel_w = target_pos - cam_pos
    p_cam = quat_apply_inverse(cam_quat, rel_w)

    depth = p_cam[:, 2]
    dist = torch.linalg.norm(rel_w, dim=-1)
    valid = (
        torch.isfinite(depth)
        & torch.isfinite(p_cam[:, 0])
        & torch.isfinite(p_cam[:, 1])
        & (depth > min_depth)
        & (depth < max_depth)
    )

    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    u = fx * p_cam[:, 0] / depth.clamp(min=1e-6) + cx
    v = fy * p_cam[:, 1] / depth.clamp(min=1e-6) + cy

    height, width = camera.data.image_shape
    # 🔹 픽셀 경계(margin) 대신 카메라 프레임 각도(FOV) — u=45,v=18 이 20% margin 밖이면 pixel=False 가 되는 문제 방지
    tan_lim_x = (width * (0.5 - margin_frac)) / fx.clamp(min=1e-6)
    tan_lim_y = (height * (0.5 - margin_frac)) / fy.clamp(min=1e-6)
    fov_in = valid & (torch.abs(p_cam[:, 0]) <= depth * tan_lim_x) & (torch.abs(p_cam[:, 1]) <= depth * tan_lim_y)

    # 레거시 픽셀 박스 (디버그용)
    margin_w = margin_frac * width
    margin_h = margin_frac * height
    pixel_in = valid & (u >= margin_w) & (u <= width - margin_w) & (v >= margin_h) & (v <= height - margin_h)

    cone_in = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if _USE_CONE:
        half = math.radians(_CONE_HALF_ANGLE_DEG)
        tan_cone = math.tan(half)
        cone_in = valid & (torch.abs(p_cam[:, 0]) <= depth * tan_cone) & (torch.abs(p_cam[:, 1]) <= depth * tan_cone)

    return {
        "valid": valid,
        "fov_in": fov_in,
        "pixel_in": pixel_in,
        "cone_in": cone_in,
        "depth": depth,
        "u": u,
        "v": v,
        "dist": dist,
    }


def barcode_in_frame_mask(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("right_hand_cam"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("barcode_target"),
    margin_frac: float = 0.15,
    min_depth: float = 0.08,
    max_depth: float = 2.5,
) -> torch.Tensor:
    """바코드 타겟이 핸드 카메라 시야 안 (픽셀 또는 전방 콘)."""
    # 🔹 텔레옵 비활성화 상태에서는 바코드 인식을 원천적으로 False 처리하여 오동작 방지
    if not getattr(env, "teleoperation_active", False):
        empty = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        setattr(env, _IN_FRAME_KEY, empty)
        m = {
            "valid": empty,
            "fov_in": empty,
            "pixel_in": empty,
            "cone_in": empty,
            "depth": torch.zeros(env.num_envs, device=env.device),
            "u": torch.zeros(env.num_envs, device=env.device),
            "v": torch.zeros(env.num_envs, device=env.device),
            "dist": torch.zeros(env.num_envs, device=env.device),
        }
        setattr(env, _DEBUG_KEY, m)
        hold_time = _get_hold_tensor(env)
        hold_time.zero_()
        return empty

    m = _compute_view_metrics(env, camera_cfg, target_cfg, margin_frac, min_depth, max_depth)
    in_frame = m["fov_in"]
    if _USE_CONE:
        in_frame = in_frame | m["cone_in"]
    setattr(env, _IN_FRAME_KEY, in_frame)
    setattr(env, _DEBUG_KEY, m)
    return in_frame


def update_barcode_cam_hold(
    env: ManagerBasedRLEnv,
    in_frame: torch.Tensor,
    hold_time_s: float,
    step_dt: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """연속 인식 시간 누적. 반환: (성공 여부, 현재 홀드 시간 [s])."""
    dt = step_dt if step_dt is not None else env.step_dt
    hold_time = _get_hold_tensor(env)
    hold_time = torch.where(in_frame, hold_time + dt, torch.zeros_like(hold_time))
    setattr(env, _HOLD_TIME_KEY, hold_time)
    return hold_time >= hold_time_s, hold_time


def reset_barcode_cam_hold(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None) -> None:
    """환경 리셋 시 홀드 타이머 초기화."""
    if not hasattr(env, _HOLD_TIME_KEY) or getattr(env, _HOLD_TIME_KEY) is None:
        return
    hold_time = getattr(env, _HOLD_TIME_KEY)
    if env_ids is None:
        hold_time.zero_()
    else:
        hold_time[env_ids] = 0.0
    if hasattr(env, _IN_FRAME_KEY):
        setattr(env, _IN_FRAME_KEY, torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))


def get_barcode_cam_hold_time(env: ManagerBasedRLEnv) -> torch.Tensor:
    """현재 홀드 누적 시간 [s] (N,)."""
    if not hasattr(env, _HOLD_TIME_KEY) or getattr(env, _HOLD_TIME_KEY) is None:
        return torch.zeros(env.num_envs, device=env.device)
    return getattr(env, _HOLD_TIME_KEY)


def get_barcode_cam_in_frame(env: ManagerBasedRLEnv) -> torch.Tensor:
    """마지막 스텝의 in-frame 마스크 (N,) bool."""
    if not hasattr(env, _IN_FRAME_KEY) or getattr(env, _IN_FRAME_KEY) is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return getattr(env, _IN_FRAME_KEY)


def get_barcode_cam_debug(env: ManagerBasedRLEnv) -> dict[str, torch.Tensor] | None:
    """마지막 판정 디버그 수치 (depth, u, v, pixel_in, cone_in)."""
    if not hasattr(env, _DEBUG_KEY):
        return None
    return getattr(env, _DEBUG_KEY)
