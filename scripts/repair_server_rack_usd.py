#!/usr/bin/env python3
"""server_rack USD 내부 reference 수리.

Isaac Lab 이 defaultPrim(/network_rack)만 instancing 할 때
stage root 의 /visuals, /meshes 참조가 깨져 서버랙 메시가 보이지 않습니다.
teleop용 USD 에 geometry 를 /network_rack 아래로 inline 복사합니다.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdUtils

_VISUAL_LINKS = [
    "cabinet_link",
    "glass_door_link",
    *[f"empty_{i:02d}_link" for i in range(21)],
    "srv_00_link",
]

_MESH_NAMES = ["device_barcode", "door_perforated"]

SOURCE_BASE = "Server_Rack/server_rack_v5/configuration/server_rack_v5_base.usd"
OUTPUT_NAME = "server_rack_teleop.usd"


def _ensure_prim(dst_stage: Usd.Stage, path: str) -> None:
    if dst_stage.GetPrimAtPath(path).IsValid():
        return
    parent = str(Sdf.Path(path).GetParentPath())
    if parent != "/":
        _ensure_prim(dst_stage, parent)
    dst_stage.DefinePrim(path)


def _replace_with_copy(flat_layer: Sdf.Layer, src_path: str, dst_stage: Usd.Stage, dst_path: str) -> None:
    dst_layer = dst_stage.GetRootLayer()
    if dst_stage.GetPrimAtPath(dst_path).IsValid():
        dst_stage.RemovePrim(dst_path)
    _ensure_prim(dst_stage, dst_path)
    Sdf.CopySpec(flat_layer, src_path, dst_layer, dst_path)
    prim_spec = dst_layer.GetPrimAtPath(dst_path)
    if prim_spec is not None and prim_spec.referenceList:
        prim_spec.referenceList.ClearEdits()


def _fix_texture_paths(layer: Sdf.Layer, out_path: Path) -> None:
    """flatten 시 박힌 절대 texture 경로를 teleop USD 기준 상대 경로로 교체."""

    textures_dir = out_path.parent / "Server_Rack/server_rack_v5/configuration/materials/textures"
    rel_prefix = textures_dir.relative_to(out_path.parent).as_posix()

    def visit(path: Sdf.Path) -> None:
        spec = layer.GetPrimAtPath(path)
        if spec is None:
            return
        for prop in spec.properties:
            if not isinstance(prop, Sdf.AttributeSpec):
                continue
            if prop.typeName != "asset":
                continue
            value = prop.default
            if value is None:
                continue
            asset_path = str(value.path if hasattr(value, "path") else value)
            if "/materials/textures/" not in asset_path:
                continue
            texture_name = Path(asset_path).name
            prop.default = Sdf.AssetPath(f"./{rel_prefix}/{texture_name}")

        for child_spec in spec.nameChildren:
            visit(path.AppendChild(child_spec.name))

    visit(Sdf.Path("/network_rack"))


def _lift_root_to_ground(dst_stage: Usd.Stage) -> float:
    """network_rack bbox 하단이 z=0 이 되도록 root translate 적용."""

    root = dst_stage.GetPrimAtPath("/network_rack")
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(root)
    min_z = float(bbox.GetRange().GetMin()[2])
    lift = max(0.0, -min_z)
    if lift <= 1e-6:
        return 0.0

    xform = UsdGeom.Xformable(root)
    translate_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op is None:
        translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "teleopLift")
    current = translate_op.Get() or Gf.Vec3d(0.0, 0.0, 0.0)
    translate_op.Set(Gf.Vec3d(current[0], current[1], current[2] + lift))
    return lift


DOOR_OPEN_ANGLE_DEG = 150.0
DOOR_OPEN_ORIENT_WXYZ = (0.2588207011044941, 0.0, 0.0, 0.9659253825631459)


def _open_glass_door(dst_stage: Usd.Stage, angle_deg: float = DOOR_OPEN_ANGLE_DEG) -> None:
    """glass_door_link 를 door_hinge upper limit(150°) 만큼 연 상태로 설정."""

    door = dst_stage.GetPrimAtPath("/network_rack/glass_door_link")
    if not door.IsValid():
        return

    xform = UsdGeom.Xformable(door)
    orient_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            orient_op = op
            break
    if orient_op is None:
        orient_op = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, "doorOpen")

    w, x, y, z = DOOR_OPEN_ORIENT_WXYZ
    orient_op.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))


def _fix_external_references(layer: Sdf.Layer, root_path: str = "/network_rack") -> None:
    """남은 stage-root(/visuals, /meshes) 참조를 /network_rack 하위로 재지정."""

    def visit(path: Sdf.Path) -> None:
        spec = layer.GetPrimAtPath(path)
        if spec is None:
            return

        ref_list = spec.referenceList
        items = ref_list.GetAddedOrExplicitItems() if ref_list else []
        if items:
            new_items: list[Sdf.Reference] = []
            changed = False
            for ref in items:
                prim_path = str(ref.primPath)
                if prim_path.startswith("/visuals/"):
                    link = prim_path.split("/")[2]
                    new_path = f"/network_rack/{link}/visuals"
                    if str(path) == new_path:
                        changed = True
                        continue
                    new_items.append(Sdf.Reference(assetPath="", primPath=new_path))
                    changed = True
                elif prim_path.startswith("/meshes/"):
                    mesh = prim_path.split("/")[2]
                    new_items.append(Sdf.Reference(assetPath="", primPath=f"/network_rack/meshes/{mesh}"))
                    changed = True
                else:
                    new_items.append(ref)
            if changed:
                ref_list.ClearEdits()
                for item in new_items:
                    ref_list.Add(item)

        for child_spec in spec.nameChildren:
            visit(path.AppendChild(child_spec.name))

    visit(Sdf.Path(root_path))


def repair_server_rack(custom_assets_dir: Path) -> Path:
    src_path = custom_assets_dir / "env/server_rack_v6.1" / SOURCE_BASE
    out_path = custom_assets_dir / "env/server_rack_v6.1" / OUTPUT_NAME

    if not src_path.is_file():
        raise FileNotFoundError(f"server rack base USD not found: {src_path}")

    flat_layer = UsdUtils.FlattenLayerStack(Usd.Stage.Open(str(src_path)))
    flat_stage = Usd.Stage.Open(flat_layer.identifier)

    if out_path.exists():
        out_path.unlink()

    dst_stage = Usd.Stage.CreateNew(str(out_path))
    dst_layer = dst_stage.GetRootLayer()

    # 🔹 articulation 골격
    Sdf.CopySpec(flat_layer, "/network_rack", dst_layer, "/network_rack")

    # 🔹 mesh 를 defaultPrim subtree 로 이동
    for mesh_name in _MESH_NAMES:
        src_mesh = f"/meshes/{mesh_name}"
        dst_mesh = f"/network_rack/meshes/{mesh_name}"
        if flat_stage.GetPrimAtPath(src_mesh).IsValid():
            _replace_with_copy(flat_layer, src_mesh, dst_stage, dst_mesh)

    # 🔹 reference visuals → inline geometry
    for link_name in _VISUAL_LINKS:
        src_visual = f"/visuals/{link_name}"
        dst_visual = f"/network_rack/{link_name}/visuals"
        if flat_stage.GetPrimAtPath(src_visual).IsValid():
            _replace_with_copy(flat_layer, src_visual, dst_stage, dst_visual)

    _fix_external_references(dst_layer)

    # 🔹 존재하지 않는 visuals reference 제거 (world/scene_link)
    for link_name in ("world", "scene_link"):
        visuals_path = f"/network_rack/{link_name}/visuals"
        if dst_stage.GetPrimAtPath(visuals_path).IsValid():
            dst_stage.RemovePrim(visuals_path)

    lift = _lift_root_to_ground(dst_stage)
    _open_glass_door(dst_stage)
    _fix_texture_paths(dst_layer, out_path)

    root = dst_stage.GetPrimAtPath("/network_rack")
    dst_stage.SetDefaultPrim(root)
    dst_layer.Save()

    verify = Usd.Stage.Open(str(out_path))
    cabinet_vis = verify.GetPrimAtPath("/network_rack/cabinet_link/visuals")
    cabinet_children = len(list(cabinet_vis.GetChildren())) if cabinet_vis.IsValid() else 0
    geom_count = sum(1 for p in verify.Traverse() if p.GetTypeName() in ("Mesh", "Cube", "Sphere"))
    bbox = UsdGeom.BBoxCache(0, ["default"]).ComputeWorldBound(verify.GetPrimAtPath("/network_rack"))
    world_range = bbox.GetRange()

    print(f"[repair_server_rack] ✓ {out_path}")
    print(f"[repair_server_rack]   bbox min={world_range.GetMin()} max={world_range.GetMax()}")
    print(f"[repair_server_rack]   geom={geom_count}, cabinet_visual_children={cabinet_children}, z_lift={lift:.3f}, door_open={DOOR_OPEN_ANGLE_DEG}deg")

    if cabinet_children == 0:
        raise RuntimeError("cabinet visuals still empty after repair")

    return out_path


def main() -> int:
    assets_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/workspace/user/custom_assets")
    try:
        repair_server_rack(assets_dir)
    except Exception as exc:
        print(f"[repair_server_rack] ! {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
