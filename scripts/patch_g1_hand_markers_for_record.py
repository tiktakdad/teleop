#!/usr/bin/env python3
"""G1 record: 손 트래킹은 유지하고, HDF5/robot_pov_cam에만 마커가 안 찍히게 조정.

잘못된 접근 (사용 안 함):
  enable_visualization=False  → OpenXR 리타게팅은 동작하지만 Quest/CloudXR 피드백(붉은 점)도 사라짐
  opacity=0.0                 → Replicator RGB 캡처에서 투명 구체가 검은 점으로 녹화됨

이 스크립트:
  1) env cfg: enable_visualization=True 복구 (이전 패치/문법 오류 복원)
  2) retargeter: RUN_MODE=record + HIDE_HAND_MARKERS_IN_RECORD=1 일 때
     마커 생성/visualize 호출을 건너뜀 (트래킹·리타게팅은 그대로)
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ISAACLAB_TASKS = Path("/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks")
ISAACLAB_DEVICES = Path("/workspace/isaaclab/source/isaaclab/isaaclab")

ENV_CFGS = [
    ISAACLAB_TASKS
    / "manager_based/locomanipulation/pick_place/locomanipulation_g1_env_cfg.py",
    ISAACLAB_TASKS
    / "manager_based/locomanipulation/pick_place/fixed_base_upper_body_ik_g1_env_cfg.py",
]

RETARGETER = (
    ISAACLAB_DEVICES
    / "devices/openxr/retargeters/humanoid/unitree/trihand/g1_upper_body_retargeter.py"
)

MARKER_TRUE = "enable_visualization=True"
_VIZ_BAD = re.compile(
    r"enable_visualization=False\s*,?\s*(#\s*\[teleop\].*)?"
)

PATCH_MARKER = "# [teleop] hide hand markers for record"

HELPER_FN = '''
# [teleop] hide hand markers for record
def _teleop_hide_hand_markers_for_record() -> bool:
    """record 모드 + HIDE_HAND_MARKERS_IN_RECORD=1 일 때만 POV/HDF5용 마커 숨김."""
    import os

    if os.environ.get("RUN_MODE", "teleop") != "record":
        return False
    return os.environ.get("HIDE_HAND_MARKERS_IN_RECORD", "1").lower() not in (
        "0",
        "false",
        "no",
    )

'''

INIT_OLD = """        if self._enable_visualization:
            marker_cfg = VisualizationMarkersCfg("""

INIT_NEW = """        if self._enable_visualization and not _teleop_hide_hand_markers_for_record():
            marker_cfg = VisualizationMarkersCfg("""

VIZ_OLD = """        # Visualization if enabled
        if self._enable_visualization:
            joints_position = np.zeros((self._num_open_xr_hand_joints, 3))
            joints_position[::2] = np.array([pose[:3] for pose in left_hand_poses.values()])
            joints_position[1::2] = np.array([pose[:3] for pose in right_hand_poses.values()])
            self._markers.visualize(translations=torch.tensor(joints_position, device=self._sim_device))"""

VIZ_NEW = """        # Visualization if enabled
        if self._enable_visualization and not _teleop_hide_hand_markers_for_record():
            joints_position = np.zeros((self._num_open_xr_hand_joints, 3))
            joints_position[::2] = np.array([pose[:3] for pose in left_hand_poses.values()])
            joints_position[1::2] = np.array([pose[:3] for pose in right_hand_poses.values()])
            self._markers.visualize(translations=torch.tensor(joints_position, device=self._sim_device))"""

# 이전 opacity 패치 되돌리기
MATERIAL_OPACITY = (
    "visual_material=sim_utils.PreviewSurfaceCfg("
    "diffuse_color=(1.0, 0.0, 0.0), opacity=0.0),  # [teleop] record: HDF5/POV 캡처용 투명"
)
MATERIAL_ORIGINAL = (
    "visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),"
)


def restore_env_cfg(path: Path) -> bool:
    if not path.is_file():
        print(f"[patch_g1_hand_markers] skip env (없음): {path.name}", file=sys.stderr)
        return False
    text = path.read_text(encoding="utf-8")
    new_text, n = _VIZ_BAD.subn(MARKER_TRUE, text)
    if n:
        path.write_text(new_text, encoding="utf-8")
        print(f"[patch_g1_hand_markers] enable_visualization=True 복구: {path.name} ({n}곳)")
        return True
    if MARKER_TRUE in text:
        print(f"[patch_g1_hand_markers] env OK: {path.name}")
        return True
    print(f"[patch_g1_hand_markers] enable_visualization not found: {path.name}", file=sys.stderr)
    return False


def _revert_opacity_patch(text: str) -> tuple[str, bool]:
    if MATERIAL_OPACITY not in text:
        return text, False
    return text.replace(MATERIAL_OPACITY, MATERIAL_ORIGINAL, 1), True


def patch_retargeter_skip_visualize() -> bool:
    hide = os.environ.get("HIDE_HAND_MARKERS_IN_RECORD", "1").lower() not in (
        "0",
        "false",
        "no",
    )
    if not hide:
        print("[patch_g1_hand_markers] HIDE_HAND_MARKERS_IN_RECORD=0 → retargeter 마커 그대로")
        return True

    path = RETARGETER
    if not path.is_file():
        print(f"[patch_g1_hand_markers] skip retargeter (없음): {path}", file=sys.stderr)
        return False

    text = path.read_text(encoding="utf-8")
    changed = False

    text, reverted = _revert_opacity_patch(text)
    if reverted:
        print("[patch_g1_hand_markers] 이전 opacity=0 패치 제거 (검은 점 원인)")
        changed = True

    if PATCH_MARKER not in text:
        insert_at = text.find("\n\n\nclass G1TriHandUpperBodyRetargeter")
        if insert_at == -1:
            insert_at = text.find("\n\nclass G1TriHandUpperBodyRetargeter")
        if insert_at == -1:
            print("[patch_g1_hand_markers] retargeter class anchor not found", file=sys.stderr)
            return False
        text = text[:insert_at] + "\n" + HELPER_FN + text[insert_at:]
        changed = True

    if INIT_OLD in text and INIT_NEW not in text:
        text = text.replace(INIT_OLD, INIT_NEW, 1)
        changed = True
    elif INIT_NEW in text:
        pass
    else:
        print("[patch_g1_hand_markers] init visualization pattern not found", file=sys.stderr)
        return False

    if VIZ_OLD in text and VIZ_NEW not in text:
        text = text.replace(VIZ_OLD, VIZ_NEW, 1)
        changed = True
    elif VIZ_NEW in text:
        pass
    else:
        print("[patch_g1_hand_markers] retarget visualize pattern not found", file=sys.stderr)
        return False

    if changed:
        path.write_text(text, encoding="utf-8")
        print(
            "[patch_g1_hand_markers] retargeter: record 시 마커 생성/visualize 생략 "
            "(트래킹 유지, teleop 모드에서는 붉은 점 표시)"
        )
    else:
        print("[patch_g1_hand_markers] retargeter already patched (skip visualize on record)")

    return True


def main() -> int:
    ok_env = sum(1 for p in ENV_CFGS if restore_env_cfg(p))
    ok_ret = patch_retargeter_skip_visualize()
    if ok_env == 0:
        return 1
    return 0 if ok_ret else 1


if __name__ == "__main__":
    sys.exit(main())
