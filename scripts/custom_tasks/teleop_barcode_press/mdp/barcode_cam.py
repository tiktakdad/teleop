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
_CONE_HALF_ANGLE_DEG = float(os.environ.get("BARCODE_CONE_HALF_ANGLE_DEG", "45"))
_TARGET_RADIUS = float(os.environ.get("BARCODE_TARGET_RADIUS", "0.045"))
_CAMERA_OFFSET_KEY = "_barcode_cam_body_offsets"


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    out = quat.clone()
    out[..., 1:] = -out[..., 1:]
    return out


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
        "ray_hit": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        "ray_dist": torch.zeros(env.num_envs, device=env.device),
        "u": torch.zeros(env.num_envs, device=env.device),
        "v": torch.zeros(env.num_envs, device=env.device),
    }

    cam_pos, cam_quat = _resolve_camera_pose(env, camera_cfg.name, camera)
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

    forward_w = quat_apply(cam_quat, torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1))
    ray_depth = torch.sum(rel_w * forward_w, dim=-1)
    closest = rel_w - ray_depth.unsqueeze(-1) * forward_w
    ray_dist = torch.linalg.norm(closest, dim=-1)
    ray_hit = valid & (ray_depth > min_depth) & (ray_depth < max_depth) & (ray_dist <= _TARGET_RADIUS)

    return {
        "valid": valid,
        "fov_in": fov_in,
        "pixel_in": pixel_in,
        "cone_in": cone_in,
        "ray_hit": ray_hit,
        "ray_dist": ray_dist,
        "depth": depth,
        "u": u,
        "v": v,
        "dist": dist,
    }


def _resolve_camera_pose(env: ManagerBasedRLEnv, camera_name: str, camera) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    cam_pos = camera.data.pos_w
    cam_quat = camera.data.quat_w_ros
    if cam_pos is None or cam_quat is None:
        return cam_pos, cam_quat

    side = "r" if "right" in camera_name or "_r" in camera_name else "l" if "left" in camera_name or "_l" in camera_name else ""
    if not side:
        return cam_pos, cam_quat

    offsets = getattr(env, _CAMERA_OFFSET_KEY, None)
    if offsets is None:
        offsets = {}
        setattr(env, _CAMERA_OFFSET_KEY, offsets)
    if camera_name not in offsets:
        try:
            robot = env.scene["robot"]
            body_ids, _ = robot.find_bodies([f"arm_{side}_link7"], preserve_order=True)
            if len(body_ids) != 1:
                return cam_pos, cam_quat
            body_id = body_ids[0]
            body_pos = robot.data.body_pos_w[:, body_id]
            body_quat = robot.data.body_quat_w[:, body_id]
            inv_body_quat = _quat_conjugate(body_quat)
            pos_offset = quat_apply(inv_body_quat, cam_pos - body_pos)
            quat_offset = _quat_mul(inv_body_quat, cam_quat)
            offsets[camera_name] = (body_id, pos_offset, quat_offset)
        except Exception:
            return cam_pos, cam_quat

    body_id, pos_offset, quat_offset = offsets[camera_name]
    robot = env.scene["robot"]
    body_pos = robot.data.body_pos_w[:, body_id]
    body_quat = robot.data.body_quat_w[:, body_id]
    return body_pos + quat_apply(body_quat, pos_offset), _quat_mul(body_quat, quat_offset)


def barcode_in_frame_mask(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("right_hand_cam"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("barcode_target"),
    margin_frac: float = 0.15,
    min_depth: float = 0.08,
    max_depth: float = 2.5,
) -> torch.Tensor:
    """바코드 타겟 구가 핸드 카메라 중앙 ray 와 접촉."""
    # 🔹 텔레옵 비활성화 상태에서는 바코드 인식을 원천적으로 False 처리하여 오동작 방지
    # 🔹 텔레옵 비활성화 상태에서는 바코드 인식을 정상적으로 계산하여 디버깅/시각화에 반영하되,
    # 타이머 누적만 0으로 리셋하여 오동작을 방지합니다.
    m = _compute_view_metrics(env, camera_cfg, target_cfg, margin_frac, min_depth, max_depth)
    in_frame = m["ray_hit"]
    setattr(env, _IN_FRAME_KEY, in_frame)
    setattr(env, _DEBUG_KEY, m)

    if not getattr(env, "teleoperation_active", False):
        hold_time = _get_hold_tensor(env)
        hold_time.zero_()

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
    
    # 🔹 텔레옵이 활성화되지 않은 상태라면 홀드 타임을 누적하지 않고 0으로 리셋합니다.
    if not getattr(env, "teleoperation_active", False):
        hold_time.zero_()
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device), hold_time

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
