#!/usr/bin/env python3
"""
Isaac Lab record_demos HDF5 → LeRobot Dataset v3.0 (공식 LeRobot API).

lerobot.datasets.LeRobotDataset.create / add_frame / save_episode / finalize 사용.
(lerobot examples/port_datasets 패턴과 동일)

의존성: conda activate isaaclab310 && pip install lerobot h5py

사용 예:
  conda activate isaaclab310
  python scripts/convert_isaac_hdf5_to_lerobot.py workspace/datasets/dataset_g1_260520_0652.hdf5
  # → workspace/datasets/dataset_g1_260520_0652_lerobot (자동)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as exc:
    raise SystemExit(
        "lerobot 패키지가 필요합니다. 예: conda activate isaaclab310 && pip install lerobot"
    ) from exc

DEFAULT_FPS = 50
VIDEO_KEY = "observation.images.robot_pov"
STATE_KEY = "observation.state"
ACTION_KEY = "action"


def default_output_dir(hdf5_path: Path) -> Path:
    """입력 HDF5와 같은 폴더에 <파일명>_lerobot 디렉터리."""
    return hdf5_path.parent / f"{hdf5_path.stem}_lerobot"


def default_repo_id(hdf5_path: Path) -> str:
    return f"local/{hdf5_path.stem}"


def convert(
    hdf5_path: Path,
    output_dir: Path,
    repo_id: str,
    fps: int,
    task: str,
    robot_type: str,
    push_to_hub: bool,
) -> None:
    print(f"[convert] LeRobot {LeRobotDataset.__module__} (LeRobotDataset.create)")

    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(
            [k for k in f["data"].keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[1]),
        )
        if not demo_keys:
            raise ValueError(f"demo_* 그룹 없음: {hdf5_path}")
        sample = f[f"data/{demo_keys[0]}"]
        state_dim = sample["obs/robot_joint_pos"].shape[1]
        action_dim = sample["processed_actions"].shape[1]
        h, w = sample["obs/robot_pov_cam"].shape[1:3]

    features = {
        STATE_KEY: {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [f"joint_{i}" for i in range(state_dim)],
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [f"action_{i}" for i in range(action_dim)],
        },
        VIDEO_KEY: {
            "dtype": "video",
            "shape": (3, h, w),
            "names": ["channels", "height", "width"],
        },
    }

    if output_dir.exists():
        raise FileExistsError(
            f"출력 디렉터리가 이미 있습니다: {output_dir}\n"
            "삭제 후 다시 실행하거나 --output-dir 을 다른 경로로 지정하세요."
        )

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        fps=fps,
        features=features,
        use_videos=True,
        robot_type=robot_type,
    )

    total_frames = 0
    with h5py.File(hdf5_path, "r") as f:
        for demo_key in demo_keys:
            grp = f[f"data/{demo_key}"]
            states = grp["obs/robot_joint_pos"][:]
            actions = grp["processed_actions"][:]
            images = grp["obs/robot_pov_cam"][:]
            for t in range(len(states)):
                frame = {
                    STATE_KEY: states[t].astype(np.float32),
                    ACTION_KEY: actions[t].astype(np.float32),
                    VIDEO_KEY: np.transpose(images[t], (2, 0, 1)),
                    "task": task,
                }
                dataset.add_frame(frame)
                total_frames += 1
            dataset.save_episode()
            print(f"[convert]   {demo_key}: {len(states)} frames")

    dataset.finalize()

    if push_to_hub:
        dataset.push_to_hub()

    print(f"[convert] 완료: {output_dir}")
    print(f"  episodes: {len(demo_keys)}, frames: {total_frames}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Isaac Lab HDF5 → LeRobot v3.0 (공식 LeRobotDataset API)"
    )
    parser.add_argument("hdf5_path", type=Path, help="입력 HDF5 (record_demos 출력)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="출력 디렉터리 (기본: <hdf5_stem>_lerobot)",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="LeRobot repo_id (기본: local/<hdf5_stem>)",
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="제어 주파수 (Hz)")
    parser.add_argument(
        "--task",
        default="pick and place steering wheel",
        help="단일 태스크 설명",
    )
    parser.add_argument("--robot-type", default="unitree_g1")
    parser.add_argument("--push-to-hub", action="store_true", help="Hugging Face Hub 업로드")
    args = parser.parse_args()

    hdf5_path = args.hdf5_path.resolve()
    if not hdf5_path.is_file():
        print(f"파일 없음: {hdf5_path}", file=sys.stderr)
        return 1

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else default_output_dir(hdf5_path).resolve()
    )
    repo_id = args.repo_id or default_repo_id(hdf5_path)

    print(f"[convert] 입력:  {hdf5_path}")
    print(f"[convert] 출력:  {output_dir}")
    print(f"[convert] repo_id: {repo_id}")

    convert(
        hdf5_path=hdf5_path,
        output_dir=output_dir,
        repo_id=repo_id,
        fps=args.fps,
        task=args.task,
        robot_type=args.robot_type,
        push_to_hub=args.push_to_hub,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
