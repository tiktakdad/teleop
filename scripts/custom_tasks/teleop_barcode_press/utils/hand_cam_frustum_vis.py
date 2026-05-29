# Copyright 2025 ROBOTIS / teleop custom task
"""손 카메라 FOV frustum — 텔레옵 뷰포트/XR 전용 (HDF5·hand_cam RGB 미포함).

VisualizationMarkers 는 mesh 에 invisibleToSecondaryRays 를 설정해
Replicator/hand_rgb 캡처에는 나타나지 않습니다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

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


LaserState = Literal["idle", "contact", "success"]


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    out = quat.clone()
    out[..., 1:] = -out[..., 1:]
    return out


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


class HandCamFrustumVisualizer:
    """단일 env 기준 오른손(또는 지정) 카메라 정면 레이저 인디케이터."""

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

        idle_color = (0.15, 0.55, 1.0)
        contact_color = (1.0, 0.82, 0.08)
        success_color = (0.1, 0.9, 0.35)
        self._markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/hand_cam_frustum",
                markers={
                    "laser_idle": sim_utils.CuboidCfg(
                        size=(0.006, 0.006, 1.0),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=idle_color,
                            opacity=1.0,
                        ),
                    ),
                    "laser_contact": sim_utils.CuboidCfg(
                        size=(0.006, 0.006, 1.0),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=contact_color,
                            opacity=1.0,
                        ),
                    ),
                    "laser_success": sim_utils.CuboidCfg(
                        size=(0.006, 0.006, 1.0),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=success_color,
                            opacity=1.0,
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
        self._body_id: int | None = None
        self._camera_pos_in_body: torch.Tensor | None = None
        self._camera_quat_in_body: torch.Tensor | None = None
        self._init_body_pose_source(camera_name)
        self._markers.set_visibility(False)

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled and _teleop_show_hand_cam_frustum()
        if self._markers is not None:
            self._markers.set_visibility(self._enabled)

    def _init_body_pose_source(self, camera_name: str) -> None:
        side = "r" if "right" in camera_name or "_r" in camera_name else "l"
        body_name = f"arm_{side}_link7"
        try:
            robot = self._env.scene["robot"]
            body_ids, _ = robot.find_bodies([body_name], preserve_order=True)
            if len(body_ids) != 1:
                return
            camera = self._env.scene[camera_name]
            camera.update(self._env.step_dt, force_recompute=True)
            body_id = body_ids[0]
            body_pos = robot.data.body_pos_w[0, body_id]
            body_quat = robot.data.body_quat_w[0, body_id]
            cam_pos = camera.data.pos_w[0]
            cam_quat = camera.data.quat_w_ros[0]
            inv_body_quat = _quat_conjugate(body_quat)
            self._body_id = body_id
            self._camera_pos_in_body = quat_apply(inv_body_quat.unsqueeze(0), (cam_pos - body_pos).unsqueeze(0)).squeeze(0)
            self._camera_quat_in_body = _quat_mul(inv_body_quat, cam_quat)
            self._pose_source = body_name
        except Exception as exc:
            print(f"[HandCamFrustumVisualizer] using sensor pose fallback: {exc}", flush=True)

    def _resolve_camera_pose(self, camera) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve hand camera pose from the moving wrist link when the USD sensor pose is static."""
        try:
            camera.update(self._env.step_dt, force_recompute=True)
        except Exception:
            pass

        if self._body_id is not None and self._camera_pos_in_body is not None and self._camera_quat_in_body is not None:
            robot = self._env.scene["robot"]
            body_pos = robot.data.body_pos_w[0, self._body_id]
            body_quat = robot.data.body_quat_w[0, self._body_id]
            cam_pos = body_pos + quat_apply(body_quat.unsqueeze(0), self._camera_pos_in_body.unsqueeze(0)).squeeze(0)
            cam_quat = _quat_mul(body_quat, self._camera_quat_in_body)
            return cam_pos, cam_quat

        cam_pos = camera.data.pos_w[0]
        cam_quat = camera.data.quat_w_ros[0]
        return cam_pos, cam_quat

    def update(self, tracking_active: bool = False, laser_state: LaserState | None = None) -> None:
        if not self._enabled or self._markers is None:
            return

        camera = self._env.scene[self._camera_name]
        device = self._env.device
        cam_pos, cam_quat = self._resolve_camera_pose(camera)

        K = camera.data.intrinsic_matrices[0]

        # 🔹 외부 디버그 출력용 데이터 보관
        self.latest_cam_pos = cam_pos.clone()

        if not hasattr(self, "_step_i"):
            self._step_i = 0
        self._step_i += 1

        if not torch.isfinite(cam_pos).all() or not torch.isfinite(K).all():
            self._markers.set_visibility(False)
            return

        self._markers.set_visibility(True)
        # 🔹 레이저 포인터: 카메라 원점(cam_pos)에서 ROS +Z(전방)로 0.5m
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

        laser_len = float(self._max_depth)
        laser_start = cam_pos
        laser_end = cam_pos + forward_w * laser_len
        laser_mid = 0.5 * (laser_start + laser_end)
        laser_q = _quat_from_direction(forward_w)
        laser_s = torch.tensor([1.0, 1.0, laser_len], device=device)

        translations = laser_mid.unsqueeze(0)
        orientations = laser_q.unsqueeze(0)
        scales = laser_s.unsqueeze(0)
        if laser_state is None:
            laser_state = "contact" if tracking_active else "idle"
        laser_idx = {"idle": 0, "contact": 1, "success": 2}.get(laser_state, 0)
        indices = [laser_idx]

        # 🔹 디버그용: 실제 시각화 전달 데이터 주기적 출력
        if self._step_i % 60 == 0:
            print(f"🔥 [HandCamFrustumVisualizer] visualize - laser: {translations.tolist()}, state={laser_state}, indices={indices}", flush=True)

        self._markers.visualize(
            translations=translations,
            orientations=orientations,
            scales=scales,
            marker_indices=indices,
        )

    def hide(self) -> None:
        if self._markers is not None:
            self._markers.set_visibility(False)
