# Copyright 2025 ROBOTIS / teleop custom task
"""손 카메라 바코드 시야 판정·홀드 타이머 공용 로직."""

from __future__ import annotations

import math
import os

import carb
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_HOLD_TIME_KEY = "_barcode_cam_hold_time"
_MISS_TIME_KEY = "_barcode_cam_miss_time"
_IN_FRAME_KEY = "_barcode_cam_in_frame"
_DEBUG_KEY = "_barcode_cam_debug"

# Hold timer flicker grace (s). If ray_hit briefly drops to False
# (hand-tracking jitter / camera render glitch) but recovers within this
# window, the accumulated hold time is preserved instead of reset to 0.
# Without this the 3s countdown keeps falling back to 0 and record
# success/export is never triggered.
_HOLD_MISS_GRACE_S = float(os.environ.get("BARCODE_HOLD_MISS_GRACE_S", "0.4"))

_USE_CONE = os.environ.get("BARCODE_USE_CONE", "1").lower() not in ("0", "false", "no")
_CONE_HALF_ANGLE_DEG = float(os.environ.get("BARCODE_CONE_HALF_ANGLE_DEG", "45"))
_TARGET_RADIUS = float(os.environ.get("BARCODE_TARGET_RADIUS", "0.045"))
_CAMERA_OFFSET_KEY = "_barcode_cam_body_offsets"
_BARCODE_PLANE_CACHE_KEY = "_barcode_plane_randomizer_cache"

# 🔹 reference.usd 구조: .../ServerRack/barcodes/{location_plane, location_barcode, location_barcode_02 ...}
# scene 모델이 바뀌어도 prim 이름 기준으로 추적하도록 환경변수로 오버라이드 가능.
_BARCODE_SCOPE_NAME = os.environ.get("BARCODE_SCOPE_NAME", "barcodes")
_BARCODE_PLANE_NAME = os.environ.get("BARCODE_PLANE_NAME", "location_plane")
# 랜덤 배치 대상 바코드 prim 이름 접두사 (location_barcode, location_barcode_02, location_barcode02 모두 포함)
_BARCODE_RANDOM_PREFIX = os.environ.get("BARCODE_RANDOM_PREFIX", "location_barcode")
# 성공 판정용 BarcodeTarget 구가 따라갈 주(主) 바코드 prim 이름
_BARCODE_TARGET_NAME = os.environ.get("BARCODE_TARGET_NAME", "location_barcode")
# 성공 판정 대상 RigidObject(scene entity) 이름
_BARCODE_TARGET_ENTITY = "barcode_target"
# 바코드끼리 겹치지 않게 할 때의 추가 여유 간격(m)과 재시도 횟수
_BARCODE_OVERLAP_MARGIN = float(os.environ.get("BARCODE_OVERLAP_MARGIN", "0.01"))
_BARCODE_OVERLAP_MAX_TRIES = int(os.environ.get("BARCODE_OVERLAP_MAX_TRIES", "30"))


def _find_descendant_by_name(root_prim, name: str):
    from pxr import Usd

    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() == name:
            return prim
    return None


def _find_descendants_by_prefix(root_prim, prefix: str):
    from pxr import Usd

    out = []
    for prim in Usd.PrimRange(root_prim):
        if prim.GetName().startswith(prefix):
            out.append(prim)
    return out


def _get_or_add_translate_op(prim):
    from pxr import UsdGeom

    xformable = UsdGeom.Xformable(prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op
    return xformable.AddTranslateOp()


def _resolve_rack_prims_for_env(env, env_idx: int):
    """env_{idx} 안에서 barcodes scope · location_plane · location_barcode* prim 들을 해석."""
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None

    env_root = stage.GetPrimAtPath(f"/World/envs/env_{env_idx}")
    if not env_root.IsValid():
        return None

    # barcodes scope 를 우선 탐색하고, 없으면 env 전체에서 탐색한다.
    scope = _find_descendant_by_name(env_root, _BARCODE_SCOPE_NAME)
    search_root = scope if (scope is not None and scope.IsValid()) else env_root

    plane_prim = _find_descendant_by_name(search_root, _BARCODE_PLANE_NAME)
    if plane_prim is None or not plane_prim.IsValid():
        return None

    barcode_prims = [
        p for p in _find_descendants_by_prefix(search_root, _BARCODE_RANDOM_PREFIX)
        if p.GetName() != _BARCODE_PLANE_NAME
    ]
    if not barcode_prims:
        return None

    time_code = Usd.TimeCode.Default()
    barcodes = []
    for prim in barcode_prims:
        world_t = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time_code).ExtractTranslation()
        barcodes.append(
            {
                "path": str(prim.GetPath()),
                "name": prim.GetName(),
                "fixed_world_x": float(world_t[0]),
                "is_target": prim.GetName() == _BARCODE_TARGET_NAME,
            }
        )

    # 정확히 일치하는 타겟이 없으면 첫 번째 바코드를 타겟으로 사용.
    if not any(b["is_target"] for b in barcodes):
        barcodes[0]["is_target"] = True

    return {
        "plane_path": str(plane_prim.GetPath()),
        "barcodes": barcodes,
    }


def _move_barcode_target(env: ManagerBasedRLEnv, target_world_by_env: dict[int, tuple[float, float, float]]) -> None:
    """성공 판정용 BarcodeTarget 구를 주 바코드의 새 월드 좌표로 이동 (teleop/record 공통)."""
    if not target_world_by_env:
        return
    try:
        target = env.scene[_BARCODE_TARGET_ENTITY]
    except Exception:
        return

    root_pose = target.data.root_state_w[:, :7].clone()
    env_idx_list = sorted(target_world_by_env.keys())
    for env_idx in env_idx_list:
        x, y, z = target_world_by_env[env_idx]
        root_pose[env_idx, 0] = x
        root_pose[env_idx, 1] = y
        root_pose[env_idx, 2] = z

    ids = torch.tensor(env_idx_list, dtype=torch.long, device=env.device)
    try:
        target.write_root_pose_to_sim(root_pose[ids], env_ids=ids)
    except Exception as exc:
        carb.log_warn(f"[barcode_random] failed to move {_BARCODE_TARGET_ENTITY}: {exc}")


def randomize_barcode_planes_on_front_cover(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None = None,
) -> None:
    """Reset 시 location_plane 면 위에서 location_barcode 들의 Y/Z 좌표를 무작위 배치하고,
    성공 판정용 BarcodeTarget 구를 주 바코드(location_barcode) 위치로 이동시킨다."""
    import omni.usd
    from pxr import Gf, Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    if env_ids is None:
        target_env_ids = list(range(env.num_envs))
    else:
        target_env_ids = [int(i) for i in env_ids.detach().cpu().tolist()]

    cache = getattr(env, _BARCODE_PLANE_CACHE_KEY, None)
    if cache is None:
        cache = {}
        setattr(env, _BARCODE_PLANE_CACHE_KEY, cache)

    time_code = Usd.TimeCode.Default()
    bbox_cache = UsdGeom.BBoxCache(time_code, ["default", "render"])

    target_world_by_env: dict[int, tuple[float, float, float]] = {}

    for env_idx in target_env_ids:
        prim_info = cache.get(env_idx)
        if prim_info is None:
            prim_info = _resolve_rack_prims_for_env(env, env_idx)
            if prim_info is None:
                carb.log_warn(
                    f"[barcode_random] env_{env_idx}: failed to resolve "
                    f"{_BARCODE_SCOPE_NAME}/{_BARCODE_PLANE_NAME}/{_BARCODE_RANDOM_PREFIX}*"
                )
                continue
            cache[env_idx] = prim_info

        plane_prim = stage.GetPrimAtPath(prim_info["plane_path"])
        if not plane_prim.IsValid():
            continue
        plane_bbox = bbox_cache.ComputeWorldBound(plane_prim).ComputeAlignedBox()
        y_min = float(plane_bbox.GetMin()[1])
        y_max = float(plane_bbox.GetMax()[1])
        z_min = float(plane_bbox.GetMin()[2])
        z_max = float(plane_bbox.GetMax()[2])

        if y_max <= y_min or z_max <= z_min:
            continue

        # 이미 배치된 바코드들의 (중심Y, 중심Z, 절반크기Y, 절반크기Z) → 겹침 방지용
        placed: list[tuple[float, float, float, float]] = []

        for bc in prim_info["barcodes"]:
            bc_prim = stage.GetPrimAtPath(bc["path"])
            if not bc_prim.IsValid():
                continue

            # 🔹 바코드 자체 크기를 계산해 평면(location_plane) 안에서만 이동하도록 샘플 범위를 좁힌다.
            bc_bbox = bbox_cache.ComputeWorldBound(bc_prim).ComputeAlignedBox()
            half_y = max(0.0, float(bc_bbox.GetMax()[1] - bc_bbox.GetMin()[1]) * 0.5)
            half_z = max(0.0, float(bc_bbox.GetMax()[2] - bc_bbox.GetMin()[2]) * 0.5)

            lo_y, hi_y = y_min + half_y, y_max - half_y
            lo_z, hi_z = z_min + half_z, z_max - half_z
            # 바코드가 평면보다 큰 경우엔 중앙으로.
            if hi_y < lo_y:
                lo_y = hi_y = 0.5 * (y_min + y_max)
            if hi_z < lo_z:
                lo_z = hi_z = 0.5 * (z_min + z_max)

            # 🔹 다른 바코드와 겹치지 않는 위치를 찾을 때까지 재시도(rejection sampling).
            sampled_y = torch.empty(1).uniform_(lo_y, hi_y).item()
            sampled_z = torch.empty(1).uniform_(lo_z, hi_z).item()
            for _ in range(_BARCODE_OVERLAP_MAX_TRIES):
                if not any(
                    abs(sampled_y - py) < (half_y + phy + _BARCODE_OVERLAP_MARGIN)
                    and abs(sampled_z - pz) < (half_z + phz + _BARCODE_OVERLAP_MARGIN)
                    for (py, pz, phy, phz) in placed
                ):
                    break
                sampled_y = torch.empty(1).uniform_(lo_y, hi_y).item()
                sampled_z = torch.empty(1).uniform_(lo_z, hi_z).item()

            placed.append((sampled_y, sampled_z, half_y, half_z))
            target_world = Gf.Vec3d(bc["fixed_world_x"], sampled_y, sampled_z)

            parent_prim = bc_prim.GetParent()
            if not parent_prim.IsValid():
                continue
            parent_world = UsdGeom.Xformable(parent_prim).ComputeLocalToWorldTransform(time_code)
            target_local = parent_world.GetInverse().Transform(target_world)

            translate_op = _get_or_add_translate_op(bc_prim)
            translate_op.Set(Gf.Vec3d(float(target_local[0]), float(target_local[1]), float(target_local[2])))

            if bc["is_target"]:
                target_world_by_env[env_idx] = (
                    float(target_world[0]),
                    float(target_world[1]),
                    float(target_world[2]),
                )

    # 🔹 성공 판정 구(BarcodeTarget)를 주 바코드 위치로 동기화 → teleop/record 성공조건이 함께 따라감
    _move_barcode_target(env, target_world_by_env)


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
        "fov_in": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        "depth": torch.zeros(env.num_envs, device=env.device),
        "ray_hit": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        "ray_dist": torch.zeros(env.num_envs, device=env.device),
        "u": torch.zeros(env.num_envs, device=env.device),
        "v": torch.zeros(env.num_envs, device=env.device),
        "dist": torch.zeros(env.num_envs, device=env.device),
        "p_cam": torch.zeros(env.num_envs, 3, device=env.device),
    }
    cam_pos, cam_quat = _resolve_camera_pose(env, camera_cfg.name, camera)
    try:
        intrinsics = camera.data.intrinsic_matrices
    except Exception:
        intrinsics = None
    if cam_pos is None or cam_quat is None:
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

    if intrinsics is not None:
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]
        u = fx * p_cam[:, 0] / depth.clamp(min=1e-6) + cx
        v = fy * p_cam[:, 1] / depth.clamp(min=1e-6) + cy
    else:
        fx = fy = torch.ones_like(depth)
        u = torch.zeros_like(depth)
        v = torch.zeros_like(depth)

    fov_in = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    pixel_in = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    try:
        height, width = camera.data.image_shape
        if intrinsics is not None:
            # 🔹 픽셀 경계(margin) 대신 카메라 프레임 각도(FOV) — u=45,v=18 이 20% margin 밖이면 pixel=False 가 되는 문제 방지
            tan_lim_x = (width * (0.5 - margin_frac)) / fx.clamp(min=1e-6)
            tan_lim_y = (height * (0.5 - margin_frac)) / fy.clamp(min=1e-6)
            fov_in = valid & (torch.abs(p_cam[:, 0]) <= depth * tan_lim_x) & (torch.abs(p_cam[:, 1]) <= depth * tan_lim_y)

            # 레거시 픽셀 박스 (디버그용)
            margin_w = margin_frac * width
            margin_h = margin_frac * height
            pixel_in = valid & (u >= margin_w) & (u <= width - margin_w) & (v >= margin_h) & (v <= height - margin_h)
    except Exception:
        pass

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
        "p_cam": p_cam,
    }


def _resolve_camera_pose(env: ManagerBasedRLEnv, camera_name: str, camera) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    # 렌더 버퍼 글리치로 camera.data 접근이 실패해도, 한 번 계산된 body-offset 이 있으면
    # 손목 링크 기준 포즈를 계속 복원해 ray 판정/타이머가 끊기지 않게 한다.
    side = "r" if "right" in camera_name or "_r" in camera_name else "l" if "left" in camera_name or "_l" in camera_name else ""
    offsets = getattr(env, _CAMERA_OFFSET_KEY, None)
    if offsets is None:
        offsets = {}
        setattr(env, _CAMERA_OFFSET_KEY, offsets)

    if side and camera_name in offsets:
        try:
            body_id, pos_offset, quat_offset = offsets[camera_name]
            robot = env.scene["robot"]
            body_pos = robot.data.body_pos_w[:, body_id]
            body_quat = robot.data.body_quat_w[:, body_id]
            return body_pos + quat_apply(body_quat, pos_offset), _quat_mul(body_quat, quat_offset)
        except Exception:
            pass

    try:
        cam_pos = camera.data.pos_w
        cam_quat = camera.data.quat_w_ros
    except Exception:
        return None, None
    if cam_pos is None or cam_quat is None:
        return cam_pos, cam_quat

    if not side:
        return cam_pos, cam_quat

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


_DEFAULT_HAND_CAMERA_NAMES = ("right_hand_cam", "left_hand_cam")


def _normalize_camera_cfgs(camera_cfg) -> tuple[SceneEntityCfg, ...]:
    """단일/복수/None 입력을 SceneEntityCfg 튜플로 정규화. None 이면 양손 카메라."""
    if camera_cfg is None:
        return tuple(SceneEntityCfg(name) for name in _DEFAULT_HAND_CAMERA_NAMES)
    if isinstance(camera_cfg, (tuple, list)):
        return tuple(camera_cfg)
    return (camera_cfg,)


def _where_metric(mask: torch.Tensor, a: dict, b: dict) -> dict:
    """env 별 mask 에 따라 두 metric dict 를 원소 단위로 선택."""
    out = {}
    for key, av in a.items():
        bv = b[key]
        if not torch.is_tensor(av):
            out[key] = av
            continue
        m = mask
        if av.dim() > 1:
            m = mask.view(mask.shape[0], *([1] * (av.dim() - 1)))
        out[key] = torch.where(m, av, bv)
    return out


def _select_primary_metrics(per_cam_metrics: dict[str, dict]) -> dict:
    """nav/laser 표시용 대표 카메라 metric: ray_hit 우선, 아니면 (유효한) 더 가까운 카메라.

    🔹 카메라 버퍼가 비어 있으면 _compute_view_metrics 가 dist=0.0, valid=False 인
    empty dict 을 반환한다. dist 0.0 은 어떤 실제 거리보다도 작아서 그대로 비교하면
    빈 카메라가 항상 "더 가까운" 대표로 잘못 선택되어 nav/laser 가 0 벡터로 깨진다.
    유효하지 않은 카메라의 거리는 +inf 로 취급해 절대 선택되지 않게 한다.
    """
    metrics = list(per_cam_metrics.values())
    primary = metrics[0]
    for m in metrics[1:]:
        inf = float("inf")
        p_dist = torch.where(primary["valid"], primary["dist"], torch.full_like(primary["dist"], inf))
        m_dist = torch.where(m["valid"], m["dist"], torch.full_like(m["dist"], inf))
        choose = (~primary["ray_hit"]) & (m["ray_hit"] | (m_dist < p_dist))
        primary = _where_metric(choose, m, primary)
    return primary


def barcode_in_frame_mask(
    env: ManagerBasedRLEnv,
    camera_cfg=None,
    target_cfg: SceneEntityCfg = SceneEntityCfg("barcode_target"),
    margin_frac: float = 0.15,
    min_depth: float = 0.08,
    max_depth: float = 2.5,
) -> torch.Tensor:
    """바코드 타겟 구가 (양손 중 어느 한쪽) 핸드 카메라 중앙 ray 와 접촉하면 True.

    camera_cfg 가 None 이면 양손(right_hand_cam, left_hand_cam)을 모두 검사하고 OR 한다.
    """
    # 🔹 텔레옵 비활성화 상태에서도 인식은 정상 계산하여 디버깅/시각화에 반영하되,
    # 타이머 누적만 0으로 리셋하여 오동작을 방지한다.
    camera_cfgs = _normalize_camera_cfgs(camera_cfg)
    per_cam: dict[str, dict] = {}
    in_frame = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for cam_cfg in camera_cfgs:
        m = _compute_view_metrics(env, cam_cfg, target_cfg, margin_frac, min_depth, max_depth)
        per_cam[cam_cfg.name] = m
        in_frame = in_frame | m["ray_hit"]

    primary = dict(_select_primary_metrics(per_cam))
    # 🔹 네비게이션이 "더 가까운 손"을 동적으로 고를 수 있도록 방향(p_cam)·유효성(valid)도 보관
    primary["per_camera"] = {
        name: {
            "ray_hit": m["ray_hit"],
            "dist": m["dist"],
            "depth": m["depth"],
            "p_cam": m["p_cam"],
            "valid": m["valid"],
        }
        for name, m in per_cam.items()
    }

    setattr(env, _IN_FRAME_KEY, in_frame)
    setattr(env, _DEBUG_KEY, primary)

    if not getattr(env, "teleoperation_active", False):
        hold_time = _get_hold_tensor(env)
        hold_time.zero_()
        _get_miss_tensor(env).zero_()

    return in_frame


def _get_miss_tensor(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, _MISS_TIME_KEY) or getattr(env, _MISS_TIME_KEY) is None:
        setattr(env, _MISS_TIME_KEY, torch.zeros(env.num_envs, device=env.device))
    return getattr(env, _MISS_TIME_KEY)


def update_barcode_cam_hold(
    env: ManagerBasedRLEnv,
    in_frame: torch.Tensor,
    hold_time_s: float,
    step_dt: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """연속 인식 시간 누적. 반환: (성공 여부, 현재 홀드 시간 [s]).

    🔹 짧은 ray_hit flicker(False 한두 프레임)에는 누적 시간을 0으로 리셋하지 않고
    유지한다. miss(연속 미인식) 시간이 _HOLD_MISS_GRACE_S 를 초과할 때만 리셋한다.
    """
    dt = step_dt if step_dt is not None else env.step_dt
    hold_time = _get_hold_tensor(env)
    miss_time = _get_miss_tensor(env)

    # 🔹 텔레옵이 활성화되지 않은 상태라면 홀드/miss 타임을 누적하지 않고 0으로 리셋합니다.
    if not getattr(env, "teleoperation_active", False):
        hold_time.zero_()
        miss_time.zero_()
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device), hold_time

    in_frame = in_frame.to(dtype=torch.bool)
    # 인식된 env: hold += dt, miss = 0
    # 미인식 env: hold 유지(추가 없음), miss += dt → grace 초과 시 hold = 0
    miss_time = torch.where(in_frame, torch.zeros_like(miss_time), miss_time + dt)
    grace_expired = miss_time > _HOLD_MISS_GRACE_S
    hold_time = torch.where(in_frame, hold_time + dt, hold_time)
    hold_time = torch.where(grace_expired, torch.zeros_like(hold_time), hold_time)

    setattr(env, _HOLD_TIME_KEY, hold_time)
    setattr(env, _MISS_TIME_KEY, miss_time)
    return hold_time >= hold_time_s, hold_time


def reset_barcode_cam_hold(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None) -> None:
    """환경 리셋 시 홀드 타이머 초기화."""
    if not hasattr(env, _HOLD_TIME_KEY) or getattr(env, _HOLD_TIME_KEY) is None:
        return
    hold_time = getattr(env, _HOLD_TIME_KEY)
    miss_time = getattr(env, _MISS_TIME_KEY, None)
    if env_ids is None:
        hold_time.zero_()
        if miss_time is not None:
            miss_time.zero_()
    else:
        hold_time[env_ids] = 0.0
        if miss_time is not None:
            miss_time[env_ids] = 0.0
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
