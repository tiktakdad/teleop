#!/usr/bin/env python3
"""G1 Locomanipulation env cfg에 robot_pov_cam (torso d435 RGB) 추가."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Isaac Lab Locomanipulation SDG 기본(속도·용량). D435 실기 640×480 등이 아님 → .env로 상향 가능
CAM_WIDTH = int(os.environ.get("ROBOT_CAM_WIDTH", "256"))
CAM_HEIGHT = int(os.environ.get("ROBOT_CAM_HEIGHT", "160"))

MARKER = "# [teleop] robot_pov_cam"
ISAACLAB_TASKS = Path("/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks")

TARGETS = [
    ISAACLAB_TASKS
    / "manager_based/locomanipulation/pick_place/locomanipulation_g1_env_cfg.py",
    ISAACLAB_TASKS
    / "manager_based/locomanipulation/pick_place/fixed_base_upper_body_ik_g1_env_cfg.py",
]

CAMERA_IMPORT = "from isaaclab.sensors import CameraCfg\n"

SCENE_BLOCK = f"""
    # [teleop] robot_pov_cam — G1 몸통(d435) 로봇 시점 (Locomanipulation SDG 동일)
    robot_pov_cam = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/torso_link/d435_link/camera",
        update_period=0.0,
        height={CAM_HEIGHT},
        width={CAM_WIDTH},
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=8.0, clipping_range=(0.1, 20.0)),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0), rot=(0.9848078, 0.0, -0.1736482, 0.0), convention="world"
        ),
    )

"""

OBS_BLOCK = """
        robot_pov_cam = ObsTerm(
            func=manip_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("robot_pov_cam"), "data_type": "rgb", "normalize": False},
        )

"""


def _replace_robot_pov_block(text: str) -> str:
    """이미 패치된 cfg에서 robot_pov_cam 블록만 갱신 (해상도 변경 등)."""
    pat = re.compile(
        r"\n    # \[teleop\] robot_pov_cam.*?robot_pov_cam = CameraCfg\([\s\S]*?\)\n\n",
        re.MULTILINE,
    )
    if pat.search(text):
        return pat.sub(SCENE_BLOCK, text, count=1)
    return text


def patch_file(path: Path) -> bool:
    if not path.is_file():
        print(f"[patch_g1_camera] skip (없음): {path}", file=sys.stderr)
        return False

    text = path.read_text(encoding="utf-8")

    if MARKER in text:
        updated = _replace_robot_pov_block(text)
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            print(f"[patch_g1_camera] updated: {path.name} ({CAM_WIDTH}x{CAM_HEIGHT})")
            return True
        print(f"[patch_g1_camera] already patched: {path.name}")
        return True

    changed = False

    if "from isaaclab.sensors import CameraCfg" not in text:
        m = re.search(r"(from isaaclab\.assets import ArticulationCfg.*\n)", text)
        if not m:
            print(f"[patch_g1_camera] fail import: {path.name}", file=sys.stderr)
            return False
        text = text[: m.end()] + CAMERA_IMPORT + text[m.end() :]
        changed = True

    if "robot_pov_cam = CameraCfg" not in text:
        if re.search(r"(\n    # Ground plane\n)", text):
            text = re.sub(r"(\n    # Ground plane\n)", SCENE_BLOCK + r"\1", text, count=1)
            changed = True
        elif re.search(r"(\n    # Lights\n)", text):
            text = re.sub(r"(\n    # Lights\n)", SCENE_BLOCK + r"\1", text, count=1)
            changed = True
        else:
            print(f"[patch_g1_camera] fail scene insert: {path.name}", file=sys.stderr)
            return False

    if "robot_pov_cam = ObsTerm" not in text or "sensor_cfg" not in text:
        pat = re.compile(
            r"(        object = ObsTerm\(\n"
            r"            func=manip_mdp\.object_obs,\n"
            r'            params=\{"left_eef_link_name": "left_wrist_yaw_link", '
            r'"right_eef_link_name": "right_wrist_yaw_link"\},\n'
            r"        \)\n)"
            r"(\n        def __post_init__\(self\):)",
            re.MULTILINE,
        )
        if not pat.search(text):
            print(f"[patch_g1_camera] fail obs insert: {path.name}", file=sys.stderr)
            return False
        text = pat.sub(r"\1" + OBS_BLOCK + r"\2", text, count=1)
        changed = True

    if changed or MARKER not in text:
        path.write_text(text, encoding="utf-8")
        print(f"[patch_g1_camera] patched: {path.name} ({CAM_WIDTH}x{CAM_HEIGHT})")
        return True

    print(f"[patch_g1_camera] no changes: {path.name}", file=sys.stderr)
    return False


def main() -> int:
    ok = sum(1 for p in TARGETS if patch_file(p))
    if ok == 0:
        print("[patch_g1_camera] FAILED: no files patched", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
