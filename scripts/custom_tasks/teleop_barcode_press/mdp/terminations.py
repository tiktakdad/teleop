# Copyright 2025 ROBOTIS / teleop custom task
"""바코드 프레스 태스크 종료 조건."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

from .barcode_cam import barcode_in_frame_mask, reset_barcode_cam_hold, update_barcode_cam_hold

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def barcode_in_hand_cam(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("right_hand_cam"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("barcode_target"),
    margin_frac: float = 0.15,
    min_depth: float = 0.08,
    max_depth: float = 2.5,
) -> torch.Tensor:
    """오른손 핸드 카메라 화면에 바코드가 들어오면 즉시 성공 (홀드 없음)."""
    return barcode_in_frame_mask(
        env,
        camera_cfg=camera_cfg,
        target_cfg=target_cfg,
        margin_frac=margin_frac,
        min_depth=min_depth,
        max_depth=max_depth,
    )


def barcode_in_hand_cam_hold(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("right_hand_cam"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("barcode_target"),
    margin_frac: float = 0.15,
    min_depth: float = 0.08,
    max_depth: float = 2.5,
    hold_time_s: float = 2.0,
) -> torch.Tensor:
    """바코드가 연속으로 hold_time_s 초 동안 카메라 안에 있으면 성공."""
    in_frame = barcode_in_frame_mask(
        env,
        camera_cfg=camera_cfg,
        target_cfg=target_cfg,
        margin_frac=margin_frac,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    success, _ = update_barcode_cam_hold(env, in_frame, hold_time_s=hold_time_s)
    return success


def barcode_pressed(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("barcode_target"),
    finger_body_names: tuple[str, ...] = ("gripper_r_rh_p12_rn_l1", "gripper_l_rh_p12_rn_l1"),
    distance_threshold: float = 0.045,
) -> torch.Tensor:
    """로봇 손가락이 바코드 타겟에 닿으면 성공.

    Args:
        env: 환경 인스턴스.
        robot_cfg: 로봇 entity 설정.
        target_cfg: 바코드 타겟(보이지 않는 구) entity 설정.
        finger_body_names: 접촉 판정에 사용할 gripper finger body 이름.
        distance_threshold: 성공 거리 임계값 [m].
    """
    robot = env.scene[robot_cfg.name]
    target = env.scene[target_cfg.name]

    target_pos = target.data.root_pos_w - env.scene.env_origins
    body_names = robot.data.body_names

    min_dist = torch.full((env.num_envs,), float("inf"), device=env.device)
    for body_name in finger_body_names:
        if body_name not in body_names:
            continue
        body_idx = body_names.index(body_name)
        finger_pos = robot.data.body_pos_w[:, body_idx, :] - env.scene.env_origins
        dist = torch.linalg.norm(finger_pos - target_pos, dim=-1)
        min_dist = torch.minimum(min_dist, dist)

    return min_dist < distance_threshold
