# Copyright 2025 ROBOTIS / teleop custom task
"""Quest/CloudXR 시야용 바코드 태스크 HUD (3D 월드 공간, 헤드 추적).

Isaac Teleop WebXR HTML 오버레이와 별도로, 시뮬 3D 마커를 HMD 앞에 배치합니다.
VisualizationMarkers → invisibleToSecondaryRays → hand_rgb/HDF5 미포함, VR 메인 뷰에는 표시.
"""

from __future__ import annotations

import os

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply


def _xr_hud_enabled() -> bool:
    if os.environ.get("RUN_MODE", "teleop") == "record":
        return False
    return os.environ.get("BARCODE_XR_HUD", "1").lower() not in ("0", "false", "no")


def _get_head_pose_world() -> tuple[np.ndarray, np.ndarray] | None:
    """OpenXR 헤드 (pos xyz, quat wxyz) 월드 좌표."""
    try:
        from omni.kit.xr.core import XRCore

        head = XRCore.get_singleton().get_input_device("/user/head")
        if head is None:
            return None
        mat = head.get_virtual_world_pose("")
        pos = np.array(mat.ExtractTranslation(), dtype=np.float32)
        q = mat.ExtractRotationQuat()
        quat = np.array([q.GetReal(), q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2]], dtype=np.float32)
        return pos, quat
    except Exception:
        return None


class BarcodeXrHud:
    """HMD 앞 진행률 바 + 상태 표시 (Quest VR 스트림에 포함)."""

    NUM_SEGMENTS = 14
    HUD_DISTANCE = float(os.environ.get("BARCODE_XR_HUD_DISTANCE", "0.55"))
    HUD_DOWN = float(os.environ.get("BARCODE_XR_HUD_DOWN", "0.10"))
    HUD_SPREAD = float(os.environ.get("BARCODE_XR_HUD_SPREAD", "0.035"))

    def __init__(self, hold_time_s: float, device: str | torch.device = "cpu"):
        self._hold_time_s = hold_time_s
        self._device = torch.device(device)
        self._enabled = _xr_hud_enabled()
        self._markers: VisualizationMarkers | None = None

        if not self._enabled:
            return

        self._markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/barcode_xr_hud",
                markers={
                    "idle": sim_utils.SphereCfg(
                        radius=0.012,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.45, 0.45, 0.5),
                            opacity=0.85,
                        ),
                    ),
                    "active": sim_utils.SphereCfg(
                        radius=0.014,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.85, 0.1),
                            opacity=0.95,
                        ),
                    ),
                    "done": sim_utils.SphereCfg(
                        radius=0.014,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.15, 0.95, 0.35),
                            opacity=0.95,
                        ),
                    ),
                    "wait": sim_utils.SphereCfg(
                        radius=0.018,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.9, 0.25, 0.2),
                            opacity=0.9,
                        ),
                    ),
                    "ok": sim_utils.SphereCfg(
                        radius=0.022,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.1, 1.0, 0.4),
                            opacity=1.0,
                        ),
                    ),
                },
            )
        )
        dev = self._device
        self._markers.visualize(
            translations=torch.zeros(1, 3, device=dev),
            orientations=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev),
            scales=torch.tensor([[0.001, 0.001, 0.001]], device=dev),
            marker_indices=[0],
        )
        self._markers.set_visibility(False)

    def hide(self) -> None:
        if self._markers is not None:
            self._markers.set_visibility(False)

    def update(
        self,
        *,
        teleop_active: bool,
        in_frame: bool,
        hold_s: float,
        success_flash: bool = False,
    ) -> None:
        if not self._enabled or self._markers is None:
            return

        head = _get_head_pose_world()
        if head is None:
            self._markers.set_visibility(False)
            return

        pos, quat_np = head
        quat = torch.tensor(quat_np, dtype=torch.float32, device=self._device)
        pos_t = torch.tensor(pos, dtype=torch.float32, device=self._device)

        # 🔹 HMD 로컬: 앞(-Z), 오른쪽(+X), 위(+Y) — OpenXR/Isaac 관례
        forward = quat_apply(quat, torch.tensor([0.0, 0.0, -1.0], device=self._device))
        right = quat_apply(quat, torch.tensor([1.0, 0.0, 0.0], device=self._device))
        up = quat_apply(quat, torch.tensor([0.0, 1.0, 0.0], device=self._device))
        center = pos_t + self.HUD_DISTANCE * forward - self.HUD_DOWN * up

        translations: list[torch.Tensor] = []
        indices: list[int] = []

        if success_flash:
            translations.append(center)
            indices.append(4)  # ok
        elif not teleop_active:
            translations.append(center + 0.5 * self.HUD_SPREAD * right)
            indices.append(3)  # wait — START 대기
            for i in range(self.NUM_SEGMENTS):
                off = (i - (self.NUM_SEGMENTS - 1) / 2.0) * self.HUD_SPREAD
                translations.append(center + off * right - 0.02 * up)
                indices.append(0)
        else:
            progress = min(1.0, hold_s / self._hold_time_s) if in_frame else 0.0
            n_fill = int(progress * self.NUM_SEGMENTS)
            status_idx = 1 if in_frame else 3
            translations.append(center - 0.06 * up)
            indices.append(status_idx)

            for i in range(self.NUM_SEGMENTS):
                off = (i - (self.NUM_SEGMENTS - 1) / 2.0) * self.HUD_SPREAD
                translations.append(center + off * right - 0.02 * up)
                if not in_frame:
                    indices.append(0)
                elif i < n_fill:
                    indices.append(2)
                else:
                    indices.append(1)

        if not translations:
            return

        n = len(translations)
        dev = self._device
        # 🔹 scale/orientation 미지정 시 PointInstancer 가 inf scale → RTX 경고·VR 화면 깨짐
        orientations = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n, device=dev)
        scales = torch.ones(n, 3, device=dev)

        self._markers.set_visibility(True)
        self._markers.visualize(
            translations=torch.stack(translations),
            orientations=orientations,
            scales=scales,
            marker_indices=indices,
        )
