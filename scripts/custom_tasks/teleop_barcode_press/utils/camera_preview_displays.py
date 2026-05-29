"""World-space camera preview displays for the teleop task."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _normalize_xy(vec: np.ndarray) -> np.ndarray:
    out = np.array([vec[0], vec[1], 0.0], dtype=np.float32)
    norm = float(np.linalg.norm(out))
    if norm < 1e-6:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return out / norm


class CameraPreviewDisplays:
    """Three in-scene display planes fed by left, head, and right camera RGB."""

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        *,
        robot_pos: tuple[float, float, float],
        target_pos: tuple[float, float, float],
        camera_names: tuple[str, str, str] = ("left_hand_cam", "head_cam", "right_hand_cam"),
        update_period_steps: int | None = None,
    ):
        self._env = env
        self._camera_names = camera_names
        self._enabled = os.environ.get("TELEOP_CAMERA_PREVIEW_DISPLAYS", "1").lower() not in ("0", "false", "no")
        self._period = max(1, update_period_steps or int(os.environ.get("TELEOP_CAMERA_PREVIEW_PERIOD_STEPS", "6")))
        self._step_i = 0
        self._texture_inputs = {}
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="teleop_camera_previews_"))
        # 중앙(head_cam) 디스플레이 월드 정보 (컨트롤 패널 배치용)
        self.center_display_info: dict | None = None

        if not self._enabled:
            return

        try:
            import omni.usd

            self._stage = omni.usd.get_context().get_stage()
            self._define_displays(robot_pos, target_pos)
        except Exception as exc:
            print(f"[CameraPreviewDisplays] disabled: {exc}", flush=True)
            self._enabled = False

    def _define_displays(self, robot_pos: tuple[float, float, float], target_pos: tuple[float, float, float]) -> None:
        width = float(os.environ.get("TELEOP_CAMERA_PREVIEW_WIDTH", "1.05"))
        height = float(os.environ.get("TELEOP_CAMERA_PREVIEW_HEIGHT", "0.66"))
        gap = float(os.environ.get("TELEOP_CAMERA_PREVIEW_GAP", "0.16"))
        behind = float(os.environ.get("TELEOP_CAMERA_PREVIEW_BEHIND", "0.35"))
        up_offset = float(os.environ.get("TELEOP_CAMERA_PREVIEW_UP", "0.55"))
        anchor = os.environ.get("TELEOP_CAMERA_PREVIEW_ANCHOR", "cabinet").strip().lower()

        robot = np.array(robot_pos, dtype=np.float32)
        target = np.array(target_pos, dtype=np.float32)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        cabinet_bounds = self._cabinet_bounds()
        if anchor == "robot":
            forward = _normalize_xy(target - robot)
            right = np.cross(up, forward)
            right = right / max(float(np.linalg.norm(right)), 1e-6)
            center = robot - behind * forward + up_offset * up
        elif cabinet_bounds is not None:
            cabinet_center, cabinet_top_z = cabinet_bounds
            # 컴비닛(cavinet_v2) 중심 기준으로 로봇을 바라보게 정렬
            # (바코드 Y 오프셋으로 인한 전체 기울어짐 제거, 높이는 현재 높이 유지)
            forward = _normalize_xy(robot - cabinet_center)
            right = np.cross(up, forward)
            right = right / max(float(np.linalg.norm(right)), 1e-6)
            center = np.array(
                [
                    cabinet_center[0] + behind * forward[0],
                    cabinet_center[1] + behind * forward[1],
                    cabinet_top_z + up_offset,
                ],
                dtype=np.float32,
            )
        else:
            forward = _normalize_xy(target - robot)
            right = np.cross(up, forward)
            right = right / max(float(np.linalg.norm(right)), 1e-6)
            center = target + behind * forward + up_offset * up

        root = UsdGeom.Xform.Define(self._stage, "/World/CameraPreviewDisplays")
        root.GetPrim().SetMetadata("hide_in_stage_window", False)

        hand_width = height
        hand_height = width
        offsets = [-(0.5 * hand_width + gap + 0.5 * width), 0.0, 0.5 * width + gap + 0.5 * hand_width]
        for camera_name, offset in zip(self._camera_names, offsets, strict=True):
            display_center = center + offset * right
            display_width, display_height = (hand_width, hand_height) if "hand" in camera_name else (width, height)
            mesh_path = f"/World/CameraPreviewDisplays/{camera_name}"
            mesh = UsdGeom.Mesh.Define(self._stage, mesh_path)
            points = [
                display_center - 0.5 * display_width * right - 0.5 * display_height * up,
                display_center + 0.5 * display_width * right - 0.5 * display_height * up,
                display_center + 0.5 * display_width * right + 0.5 * display_height * up,
                display_center - 0.5 * display_width * right + 0.5 * display_height * up,
            ]
            mesh.CreatePointsAttr([Gf.Vec3f(*p.tolist()) for p in points])
            mesh.CreateFaceVertexCountsAttr([4])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            mesh.CreateDoubleSidedAttr(True)
            st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                "st",
                Sdf.ValueTypeNames.TexCoord2fArray,
                UsdGeom.Tokens.varying,
            )
            st.Set([Gf.Vec2f(0.0, 1.0), Gf.Vec2f(1.0, 1.0), Gf.Vec2f(1.0, 0.0), Gf.Vec2f(0.0, 0.0)])
            self._bind_camera_material(mesh, camera_name)
            if abs(offset) < 1e-6 or camera_name == self._camera_names[1]:
                self.center_display_info = {
                    "center": np.array(display_center, dtype=np.float32),
                    "right": np.array(right, dtype=np.float32),
                    "up": np.array(up, dtype=np.float32),
                    "forward": np.array(forward, dtype=np.float32),
                    "width": float(display_width),
                    "height": float(display_height),
                }

    def _cabinet_bounds(self) -> tuple[np.ndarray, float] | None:
        candidates = ("cavinet_v2", "cabinet_v2", "cabinet_link/visuals", "cabinet_link")
        for suffix in candidates:
            prim = self._find_prim_by_suffix(suffix)
            if prim is None:
                continue
            try:
                bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(prim)
                bbox_range = bbox.GetRange()
                if bbox_range.IsEmpty():
                    continue
                center = bbox_range.GetMidpoint()
                top_z = float(bbox_range.GetMax()[2])
                return np.array([center[0], center[1], center[2]], dtype=np.float32), top_z
            except Exception:
                continue
        return None

    def _find_prim_by_suffix(self, suffix: str):
        suffix = suffix.strip("/")
        for prim in self._stage.Traverse():
            path = str(prim.GetPath()).strip("/")
            if path.endswith(suffix):
                return prim
        return None

    def _bind_camera_material(self, mesh: UsdGeom.Mesh, camera_name: str) -> None:
        material_path = f"/World/CameraPreviewDisplays/Looks/{camera_name}_mat"
        material = UsdShade.Material.Define(self._stage, material_path)
        shader = UsdShade.Shader.Define(self._stage, f"{material_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.02, 0.02, 0.02))

        reader = UsdShade.Shader.Define(self._stage, f"{material_path}/PrimvarReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

        texture = UsdShade.Shader.Define(self._stage, f"{material_path}/CameraTexture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        file_input = texture.CreateInput("file", Sdf.ValueTypeNames.Asset)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(mesh).Bind(material)
        self._texture_inputs[camera_name] = file_input

    def _camera_rgb_array(self, camera_name: str) -> np.ndarray | None:
        camera = self._env.scene[camera_name]
        try:
            camera.update(self._env.step_dt, force_recompute=True)
        except Exception:
            pass

        output = getattr(camera.data, "output", None)
        image = output.get("rgb") if isinstance(output, dict) else None
        if image is None:
            return None
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image[0]
            image = image[..., :3].detach().cpu()
            if image.dtype.is_floating_point:
                max_value = float(image.max().item()) if image.numel() else 0.0
                if max_value <= 1.0:
                    image = image * 255.0
                image = image.clamp(0, 255).to(torch.uint8)
            else:
                image = image.clamp(0, 255).to(torch.uint8)
            return image.numpy()

        arr = np.asarray(image)
        if arr.ndim == 4:
            arr = arr[0]
        arr = arr[..., :3]
        if np.issubdtype(arr.dtype, np.floating):
            if arr.size and float(arr.max()) <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def update(self) -> None:
        if not self._enabled:
            return
        self._step_i += 1
        if self._step_i % self._period != 0:
            return

        try:
            from PIL import Image
        except Exception:
            self._enabled = False
            print("[CameraPreviewDisplays] disabled: Pillow is unavailable", flush=True)
            return

        for camera_name in self._camera_names:
            arr = self._camera_rgb_array(camera_name)
            if arr is None or arr.size == 0:
                continue
            # 핸드 카메라 영상은 시계방향으로 90도 회전 (디스플레이 평면은 세로형 유지)
            if "hand" in camera_name:
                arr = np.rot90(arr, k=-1)
            elif "head" in camera_name:
                # 중앙(헤드) 카메라 영상이 상하 반전되어 있어 180도 회전
                arr = np.rot90(arr, k=2)
            # 디스플레이에 비친 좌우 반전 보정 (텍스처 U축 = 디스플레이 가로 방향)
            arr = np.ascontiguousarray(np.fliplr(arr))
            path = self._tmp_dir / f"{camera_name}_{self._step_i:08d}.png"
            Image.fromarray(arr).save(path)
            texture_input = self._texture_inputs.get(camera_name)
            if texture_input is not None:
                texture_input.Set(Sdf.AssetPath(str(path)))