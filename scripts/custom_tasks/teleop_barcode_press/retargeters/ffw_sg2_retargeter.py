# Copyright 2025 ROBOTIS / teleop custom task
"""OpenXR → FFW_SG2 Pink IK action 리타게팅 (Isaac Lab 2.3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as PoseUtils
from isaaclab.devices import OpenXRDevice
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

_GRIPPER_CLOSED_RAD = float(np.pi / 4.6)  # 🔹 핫 리로드 실시간 테스트용 (2차 수정)


@dataclass
class FfwSg2RetargeterCfg(RetargeterCfg):
    """FFW_SG2 리타게터 설정."""

    enable_visualization: bool = False
    num_open_xr_hand_joints: int = 52


class FfwSg2Retargeter(RetargeterBase):
    """OpenXR 양손 트래킹 → FFW_SG2 양팔 EE + 그리퍼 joint (22D action)."""

    def __init__(self, cfg: FfwSg2RetargeterCfg):
        self._enable_visualization = cfg.enable_visualization
        self._num_open_xr_hand_joints = cfg.num_open_xr_hand_joints
        self._sim_device = cfg.sim_device

        if self._enable_visualization:
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/ffw_hand_markers",
                markers={
                    "joint": sim_utils.SphereCfg(
                        radius=0.005,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.0)),
                    ),
                },
            )
            self._markers = VisualizationMarkers(marker_cfg)

    def retarget(self, data: dict) -> torch.Tensor:
        left_hand_poses = data[OpenXRDevice.TrackingTarget.HAND_LEFT]
        right_hand_poses = data[OpenXRDevice.TrackingTarget.HAND_RIGHT]

        # 🔹 실시간 핫리로드 정보 취득용: 최신 손목 트래킹 좌표 보관
        self.latest_left_wrist = np.round(left_hand_poses.get("wrist", np.zeros(7, dtype=np.float32))[:3], 3)
        self.latest_right_wrist = np.round(right_hand_poses.get("wrist", np.zeros(7, dtype=np.float32))[:3], 3)

        # 🔹 디버그용: 입력받은 손 실시간 트래킹 데이터 값 변화 확인 (너무 빈번하지 않게 60스텝마다 출력)
        self._step_i = getattr(self, "_step_i", 0) + 1
        if left_hand_poses or right_hand_poses:
            if self._step_i % 60 == 0:
                l_wrist_pos = left_hand_poses.get("wrist", [0.0, 0.0, 0.0])[:3]
                r_wrist_pos = right_hand_poses.get("wrist", [0.0, 0.0, 0.0])[:3]
                print(f"[ffw_retargeter][wrist] L: {list(np.round(l_wrist_pos, 3))}, R: {list(np.round(r_wrist_pos, 3))}", flush=True)
        else:
            if self._step_i % 60 == 0:
                print("[ffw_retargeter] ⚠️ 양손 트래킹 데이터 비어있음 (HMD 연결 안됨 또는 핸드 트래킹 비활성)", flush=True)

        left_wrist = left_hand_poses.get("wrist", np.zeros(7, dtype=np.float32))
        right_wrist = right_hand_poses.get("wrist", np.zeros(7, dtype=np.float32))

        if self._enable_visualization:
            joints_position = np.zeros((self._num_open_xr_hand_joints, 3))
            if left_hand_poses:
                left_pts = np.array([pose[:3] for pose in left_hand_poses.values()])
                n = min(left_pts.shape[0], self._num_open_xr_hand_joints // 2)
                joints_position[0 : n * 2 : 2, :3] = left_pts[:n]
            if right_hand_poses:
                right_pts = np.array([pose[:3] for pose in right_hand_poses.values()])
                n = min(right_pts.shape[0], self._num_open_xr_hand_joints // 2)
                joints_position[1 : n * 2 : 2, :3] = right_pts[:n]
            
            dev = self._sim_device
            num_pts = self._num_open_xr_hand_joints
            orientations = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_pts, device=dev)
            scales = torch.ones(num_pts, 3, device=dev)
            self._markers.visualize(
                translations=torch.tensor(joints_position, device=dev),
                orientations=orientations,
                scales=scales
            )

        hand_joints = np.zeros(8, dtype=np.float32)
        hand_joints[:4] = self._compute_gripper_joints(left_hand_poses)
        hand_joints[4:8] = self._compute_gripper_joints(right_hand_poses)

        left_wrist_tensor = torch.tensor(self._retarget_abs(left_wrist, side="left"), dtype=torch.float32, device=self._sim_device)
        right_wrist_tensor = torch.tensor(
            self._retarget_abs(right_wrist, side="right"), dtype=torch.float32, device=self._sim_device
        )
        hand_joints_tensor = torch.tensor(hand_joints, dtype=torch.float32, device=self._sim_device)

        return torch.cat([left_wrist_tensor, right_wrist_tensor, hand_joints_tensor])

    def _compute_gripper_joints(self, hand_poses: dict) -> np.ndarray:
        """엄지-검지 거리로 그리퍼 개폐 근사."""
        thumb = hand_poses.get("thumb_tip", hand_poses.get("thumb"))
        index = hand_poses.get("index_tip", hand_poses.get("index"))
        if thumb is None or index is None:
            return np.zeros(4, dtype=np.float32)

        pinch_dist = float(np.linalg.norm(np.array(thumb[:3]) - np.array(index[:3])))
        closed = np.clip(1.0 - pinch_dist / 0.08, 0.0, 1.0) * _GRIPPER_CLOSED_RAD
        return np.full(4, closed, dtype=np.float32)

    def _retarget_abs(self, wrist: np.ndarray, side: str) -> np.ndarray:
        """OpenXR wrist → FFW_SG2 wrist control frame.

        The FFW left/right wrist link frames are mirrored. Applying the same local
        orientation correction to both hands flips the left palm/back direction,
        so only the right wrist uses the GR1T2-style 180 degree Z correction.
        """
        wrist_pos = torch.tensor(wrist[:3], dtype=torch.float32)
        wrist_quat = torch.tensor(wrist[3:], dtype=torch.float32)
        openxr_wrist_in_world = PoseUtils.make_pose(wrist_pos, PoseUtils.matrix_from_quat(wrist_quat))

        zero_pos = torch.zeros(3, dtype=torch.float32)
        if side == "right":
            correction_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        else:
            correction_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        usd_link_in_openxr_wrist = PoseUtils.make_pose(zero_pos, PoseUtils.matrix_from_quat(correction_quat))

        usd_link_in_world = PoseUtils.pose_in_A_to_pose_in_B(usd_link_in_openxr_wrist, openxr_wrist_in_world)
        pos, mat = PoseUtils.unmake_pose(usd_link_in_world)
        quat = PoseUtils.quat_from_matrix(mat)
        return np.concatenate([pos.numpy(), quat.numpy()])
