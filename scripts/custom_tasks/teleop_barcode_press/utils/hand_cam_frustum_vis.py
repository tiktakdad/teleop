# Copyright 2025 ROBOTIS / teleop custom task
"""손 카메라 FOV frustum — 텔레옵 뷰포트/XR 전용 (HDF5·hand_cam RGB 미포함).

VisualizationMarkers 는 mesh 에 invisibleToSecondaryRays 를 설정해
Replicator/hand_rgb 캡처에는 나타나지 않습니다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply, quat_from_angle_axis

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _teleop_show_hand_cam_frustum() -> bool:
    # 실시간 디버깅 및 사용자 레이저 가이드를 위해 항상 활성화합니다.
    return True


def _frustum_corners_cam(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: float,
    height: float,
    depth: float,
    margin_frac: float,
    device: torch.device,
) -> torch.Tensor:
    """ROS 카메라 프레임에서 near/far 사각형 8코너 (8, 3)."""
    mw, mh = margin_frac * width, margin_frac * height
    us = torch.tensor([mw, width - mw, width - mw, mw], device=device, dtype=torch.float32)
    vs = torch.tensor([mh, mh, height - mh, height - mh], device=device, dtype=torch.float32)
    x = (us - cx) * depth / fx
    y = (vs - cy) * depth / fy
    z = torch.full_like(x, depth)
    return torch.stack([x, y, z], dim=-1)


def _quat_from_direction(direction: torch.Tensor) -> torch.Tensor:
    """로컬 +Z 축을 direction 으로 맞추는 quaternion (w, x, y, z)."""
    device = direction.device
    z = torch.tensor([0.0, 0.0, 1.0], device=device)
    d = direction / (torch.linalg.norm(direction) + 1e-8)
    dot = torch.clamp(torch.dot(z, d), -1.0, 1.0)
    if dot > 0.9999:
        return torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    if dot < -0.9999:
        return torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)
    axis = torch.cross(z, d, dim=-1)
    axis = axis / (torch.linalg.norm(axis) + 1e-8)
    angle = torch.acos(dot)
    return quat_from_angle_axis(angle.unsqueeze(0), axis.unsqueeze(0)).squeeze(0)


def _edge_poses(origin: torch.Tensor, corners_near: torch.Tensor, corners_far: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """frustum 모서리용 (중점, quat wxyz, scale xyz)."""
    device = origin.device
    edges: list[tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(4):
        edges.append((origin, corners_far[i]))
        edges.append((corners_near[i], corners_near[(i + 1) % 4]))
        edges.append((corners_far[i], corners_far[(i + 1) % 4]))
        edges.append((corners_near[i], corners_far[i]))

    translations = []
    orientations = []
    scales = []
    thickness = 0.004
    for a, b in edges:
        mid = 0.5 * (a + b)
        delta = b - a
        length = torch.linalg.norm(delta)
        if length < 1e-5:
            continue
        direction = delta / length
        quat = _quat_from_direction(direction)
        translations.append(mid)
        orientations.append(quat)
        scales.append(torch.tensor([thickness, thickness, length.item()], device=device))

    if not translations:
        empty = torch.zeros(0, 3, device=device)
        return empty, torch.zeros(0, 4, device=device), empty
    return torch.stack(translations), torch.stack(orientations), torch.stack(scales)


class HandCamFrustumVisualizer:
    """단일 env 기준 오른손(또는 지정) 카메라 frustum 와이어프레임."""

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        camera_name: str = "right_hand_cam",
        margin_frac: float = 0.15,
        min_depth: float = 0.08,
        max_depth: float = 0.25,
        opacity: float = 0.28,
    ):
        self._env = env
        self._camera_name = camera_name
        self._margin_frac = margin_frac
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._enabled = _teleop_show_hand_cam_frustum()
        print(f"🔥 [HandCamFrustumVisualizer] 초기화됨. Enabled = {self._enabled}", flush=True)
        self._markers: VisualizationMarkers | None = None

        if not self._enabled:
            return

        edge_color = (0.15, 0.55, 1.0)
        track_color = (0.1, 0.9, 0.35)
        face_color = (0.15, 0.55, 1.0)
        self._markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/hand_cam_frustum",
                markers={
                    "edge": sim_utils.CuboidCfg(
                        size=(0.015, 0.015, 0.1),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=edge_color,
                            opacity=opacity,
                        ),
                    ),
                    "edge_track": sim_utils.CuboidCfg(
                        size=(0.015, 0.015, 0.1),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=track_color,
                            opacity=min(1.0, opacity + 0.15),
                        ),
                    ),
                    "face": sim_utils.CuboidCfg(
                        size=(0.1, 0.1, 0.002),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=face_color,
                            opacity=opacity * 0.55,
                        ),
                    ),
                    "laser": sim_utils.CuboidCfg(
                        size=(0.015, 0.015, 1.0),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=edge_color,
                            opacity=0.8,
                        ),
                    ),
                    "laser_track": sim_utils.CuboidCfg(
                        size=(0.015, 0.015, 1.0),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=track_color,
                            opacity=0.9,
                        ),
                    ),
                },
            )
        )
        # PointInstancer prototype 개수 고정 (첫 visualize 전 PhysX 경고 방지)
        dev = env.device
        self._markers.visualize(
            translations=torch.zeros(1, 3, device=dev),
            orientations=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev),
            scales=torch.tensor([[0.001, 0.001, 0.001]], device=dev),
            marker_indices=[0],
        )
        self._pose_source = "sensor"
        self._markers.set_visibility(False)

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled and _teleop_show_hand_cam_frustum()
        if self._markers is not None:
            self._markers.set_visibility(self._enabled)

    def _resolve_camera_pose(self, camera) -> tuple[torch.Tensor, torch.Tensor]:
        """Read the pose reported by the authored USD camera sensor."""
        try:
            camera.update(self._env.step_dt, force_recompute=True)
        except Exception:
            pass

        cam_pos = camera.data.pos_w[0]
        cam_quat = camera.data.quat_w_ros[0]
        return cam_pos, cam_quat

    def update(self, tracking_active: bool = False) -> None:
        if not self._enabled or self._markers is None:
            return

        camera = self._env.scene[self._camera_name]
        device = self._env.device
        cam_pos, cam_quat = self._resolve_camera_pose(camera)

        K = camera.data.intrinsic_matrices[0]
        height, width = camera.data.image_shape

        # 🔹 외부 디버그 출력용 데이터 보관
        self.latest_cam_pos = cam_pos.clone()

        if not hasattr(self, "_step_i"):
            self._step_i = 0
        self._step_i += 1

        if not torch.isfinite(cam_pos).all() or not torch.isfinite(K).all():
            self._markers.set_visibility(False)
            return

        self._markers.set_visibility(True)
        fx, fy, cx, cy = K[0, 0].item(), K[1, 1].item(), K[0, 2].item(), K[1, 2].item()
        near = _frustum_corners_cam(fx, fy, cx, cy, width, height, self._min_depth, self._margin_frac, device)
        far = _frustum_corners_cam(fx, fy, cx, cy, width, height, self._max_depth, self._margin_frac, device)

        cam_quat_b = cam_quat.unsqueeze(0).expand(near.shape[0], -1)
        near_w = quat_apply(cam_quat_b, near) + cam_pos
        far_w = quat_apply(cam_quat_b, far) + cam_pos
        origin_w = cam_pos

        edge_t, edge_q, edge_s = _edge_poses(origin_w, near_w, far_w)
        # 🔹 측면 4개 — 얇은 반투명 패널
        face_t = []
        face_q = []
        face_s = []
        for i in range(4):
            p0, p1 = far_w[i], far_w[(i + 1) % 4]
            mid = (origin_w + p0 + p1) / 3.0
            w_edge = torch.linalg.norm(p1 - p0)
            h_edge = torch.linalg.norm(0.5 * (p0 + p1) - origin_w)
            normal = torch.cross(p1 - p0, origin_w - p0, dim=-1)
            face_t.append(mid)
            face_q.append(_quat_from_direction(normal))
            face_s.append(torch.tensor([w_edge.item(), h_edge.item(), 0.002], device=device))

        if edge_t.shape[0] == 0:
            return

        translations = torch.cat([edge_t, torch.stack(face_t)], dim=0)
        orientations = torch.cat([edge_q, torch.stack(face_q)], dim=0)
        scales = torch.cat([edge_s, torch.stack(face_s)], dim=0)

        # 🔹 레이저 포인터: 카메라 원점(cam_pos)에서 ROS +Z(전방)로 1.5m
        forward_w = quat_apply(
            cam_quat.unsqueeze(0),
            torch.tensor([[0.0, 0.0, 1.0]], device=device),
        ).squeeze(0)
        self.latest_forward_w = forward_w.clone()

        if self._step_i % 60 == 0:
            print(
                f"🔥 [HandCamFrustumVisualizer] update - source={self._pose_source}, "
                f"Tracking Active: {tracking_active}, cam_pos: {cam_pos.tolist()}, "
                f"forward: {forward_w.tolist()}",
                flush=True,
            )

        laser_len = 1.5
        laser_start = cam_pos
        laser_end = cam_pos + forward_w * laser_len
        laser_mid = 0.5 * (laser_start + laser_end)
        laser_q = _quat_from_direction(forward_w)
        laser_s = torch.tensor([1.0, 1.0, laser_len], device=device)

        translations = torch.cat([translations, laser_mid.unsqueeze(0)], dim=0)
        orientations = torch.cat([orientations, laser_q.unsqueeze(0)], dim=0)
        scales = torch.cat([scales, laser_s.unsqueeze(0)], dim=0)

        n_edge = edge_t.shape[0]
        edge_idx = 1 if tracking_active else 0  # 0=edge, 1=edge_track, 2=face
        indices = [edge_idx] * n_edge + [2] * (translations.shape[0] - 1 - n_edge)
        laser_idx = 4 if tracking_active else 3
        indices.append(laser_idx)

        # 🔹 디버그용: 실제 시각화 전달 데이터 주기적 출력
        if self._step_i % 60 == 0:
            print(f"🔥 [HandCamFrustumVisualizer] visualize - translations (first 3 & last 1): {translations[:3].tolist()} ... {translations[-1:].tolist()}, indices (first 3 & last 1): {indices[:3]} ... {indices[-1:]}", flush=True)

        self._markers.visualize(
            translations=translations,
            orientations=orientations,
            scales=scales,
            marker_indices=indices,
        )

    def hide(self) -> None:
        if self._markers is not None:
            self._markers.set_visibility(False)
