#!/usr/bin/env python3
"""
[DEPRECATED] convert_isaac_hdf5_to_lerobot.py 사용 권장 (lerobot 공식 API 전용).

Isaac Lab record_demos HDF5 → LeRobot Dataset v3.0 변환기 (내장 ffmpeg 폴백 포함).

공식 변환:
  conda activate isaaclab310
  python scripts/convert_isaac_hdf5_to_lerobot.py <hdf5> --output-dir <out>
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CODEBASE_VERSION = "v3.0"
DEFAULT_FPS = 50
VIDEO_KEY = "observation.images.robot_pov"
STATE_KEY = "observation.state"
ACTION_KEY = "action"

DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
TASKS_PATH = "meta/tasks.parquet"


def _try_lerobot_convert(
    hdf5_path: Path,
    output_dir: Path,
    repo_id: str,
    fps: int,
    task: str,
    robot_type: str,
) -> bool:
    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError:
        return False

    print("[convert] lerobot 패키지 사용 (LeRobotDataset.create)")

    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(
            [k for k in f["data"].keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[1]),
        )
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

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        fps=fps,
        features=features,
        use_videos=True,
        robot_type=robot_type,
    )

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
            dataset.save_episode()

    dataset.finalize()
    print(f"[convert] 완료 (lerobot): {output_dir}")
    return True


def _run(cmd: list[str], stdin: bytes | None = None) -> None:
    proc = subprocess.run(cmd, input=stdin, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"명령 실패: {' '.join(cmd)}\n{proc.stderr.decode(errors='replace')}"
        )


def _encode_episode_mp4(frames: np.ndarray, fps: int, out_path: Path) -> float:
    """HWC uint8 (T,H,W,3) → mp4. 반환: 영상 길이(초)."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"예상 shape (T,H,W,3), 실제 {frames.shape}")
    t, h, w, _ = frames.shape
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    _run(cmd, stdin=frames.tobytes())
    return t / fps


def _probe_video(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,codec_name,pix_fmt,r_frame_rate,nb_frames",
        "-of",
        "json",
        path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    info = json.loads(proc.stdout)
    stream = info["streams"][0]
    num, den = stream.get("r_frame_rate", f"{DEFAULT_FPS}/1").split("/")
    fps = float(num) / float(den) if float(den) else DEFAULT_FPS
    return {
        "video.height": int(stream["height"]),
        "video.width": int(stream["width"]),
        "video.codec": stream.get("codec_name", "h264"),
        "video.pix_fmt": stream.get("pix_fmt", "yuv420p"),
        "video.is_depth_map": False,
        "video.fps": fps,
        "video.channels": 3,
    }


def _concat_mp4(paths: list[Path], out_path: Path) -> None:
    if len(paths) == 1:
        shutil.move(paths[0], out_path)
        return
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        list_path = Path(f.name)
        for p in paths:
            f.write(f"file '{p.resolve()}'\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            str(out_path),
        ]
    )
    list_path.unlink(missing_ok=True)
    for p in paths:
        p.unlink(missing_ok=True)


def _compute_feature_stats(arr: np.ndarray) -> dict:
    return {
        "mean": arr.mean(axis=0),
        "std": arr.std(axis=0),
        "min": arr.min(axis=0),
        "max": arr.max(axis=0),
        "count": np.array([len(arr)]),
    }


def _serialize_stats(stats: dict) -> dict:
    out = {}
    for key, sub in stats.items():
        out[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in sub.items()}
    return out


def _flatten_stats(prefix: str, stats: dict) -> dict:
    flat = {}
    for feat, values in stats.items():
        for stat_name, val in values.items():
            flat[f"{prefix}/{feat}/{stat_name}"] = val
    return flat


def _standalone_convert(
    hdf5_path: Path,
    output_dir: Path,
    fps: int,
    task: str,
    robot_type: str,
) -> None:
    print("[convert] 내장 변환기 사용 (LeRobot v3.0 레이아웃)")

    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(
            [k for k in f["data"].keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[1]),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(exist_ok=True)

    # tasks
    tasks_df = pd.DataFrame({"task_index": [0]}, index=pd.Index([task], name="task"))
    tasks_df.to_parquet(meta_dir / "tasks.parquet")

    all_rows: dict[str, list] = {
        "timestamp": [],
        "frame_index": [],
        "episode_index": [],
        "index": [],
        "task_index": [],
        STATE_KEY: [],
        ACTION_KEY: [],
    }
    episode_rows: list[dict] = []
    global_stats: dict | None = None
    global_index = 0
    video_ts = 0.0
    temp_videos: list[Path] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for ep_idx, demo_key in enumerate(demo_keys):
            with h5py.File(hdf5_path, "r") as f:
                grp = f[f"data/{demo_key}"]
                states = grp["obs/robot_joint_pos"][:].astype(np.float32)
                actions = grp["processed_actions"][:].astype(np.float32)
                images = grp["obs/robot_pov_cam"][:]

            ep_len = len(states)
            ep_mp4 = tmp_path / f"ep_{ep_idx:04d}.mp4"
            ep_dur = _encode_episode_mp4(images, fps, ep_mp4)
            temp_videos.append(ep_mp4)

            ep_stats = {
                STATE_KEY: _compute_feature_stats(states),
                ACTION_KEY: _compute_feature_stats(actions),
            }

            dataset_from = global_index
            for fi in range(ep_len):
                all_rows["timestamp"].append(fi / fps)
                all_rows["frame_index"].append(fi)
                all_rows["episode_index"].append(ep_idx)
                all_rows["index"].append(global_index)
                all_rows["task_index"].append(0)
                all_rows[STATE_KEY].append(states[fi])
                all_rows[ACTION_KEY].append(actions[fi])
                global_index += 1

            ep_meta = {
                "episode_index": ep_idx,
                "tasks": [task],
                "length": ep_len,
                "dataset_from_index": dataset_from,
                "dataset_to_index": global_index,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
                f"videos/{VIDEO_KEY}/chunk_index": 0,
                f"videos/{VIDEO_KEY}/file_index": 0,
                f"videos/{VIDEO_KEY}/from_timestamp": video_ts,
                f"videos/{VIDEO_KEY}/to_timestamp": video_ts + ep_dur,
            }
            ep_meta.update(_flatten_stats("stats", ep_stats))
            episode_rows.append(ep_meta)
            video_ts += ep_dur

        # 전역 stats 재계산
        with h5py.File(hdf5_path, "r") as f:
            all_states = []
            all_actions = []
            for demo_key in demo_keys:
                grp = f[f"data/{demo_key}"]
                all_states.append(grp["obs/robot_joint_pos"][:])
                all_actions.append(grp["processed_actions"][:])
        states_cat = np.concatenate(all_states, axis=0).astype(np.float32)
        actions_cat = np.concatenate(all_actions, axis=0).astype(np.float32)
        global_stats = {
            STATE_KEY: _compute_feature_stats(states_cat),
            ACTION_KEY: _compute_feature_stats(actions_cat),
        }

        # data parquet
        data_tbl = pa.table(
            {
                "timestamp": pa.array(all_rows["timestamp"], type=pa.float32()),
                "frame_index": pa.array(all_rows["frame_index"], type=pa.int64()),
                "episode_index": pa.array(all_rows["episode_index"], type=pa.int64()),
                "index": pa.array(all_rows["index"], type=pa.int64()),
                "task_index": pa.array(all_rows["task_index"], type=pa.int64()),
                STATE_KEY: pa.array(all_rows[STATE_KEY]),
                ACTION_KEY: pa.array(all_rows[ACTION_KEY]),
            }
        )
        data_path = output_dir / DATA_PATH.format(chunk_index=0, file_index=0)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(data_tbl, data_path, compression="snappy")

        # episodes parquet
        ep_df = pd.DataFrame(episode_rows)
        ep_path = output_dir / EPISODES_PATH.format(chunk_index=0, file_index=0)
        ep_path.parent.mkdir(parents=True, exist_ok=True)
        ep_df.to_parquet(ep_path)

        # videos
        video_out = output_dir / VIDEO_PATH.format(
            video_key=VIDEO_KEY, chunk_index=0, file_index=0
        )
        _concat_mp4(temp_videos, video_out)
        video_info = _probe_video(video_out)

    state_dim = states_cat.shape[1]
    action_dim = actions_cat.shape[1]
    h, w = images.shape[1], images.shape[2]

    features = {
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        STATE_KEY: {
            "dtype": "float32",
            "shape": [state_dim],
            "names": [f"joint_{i}" for i in range(state_dim)],
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": [action_dim],
            "names": [f"action_{i}" for i in range(action_dim)],
        },
        VIDEO_KEY: {
            "dtype": "video",
            "shape": [3, h, w],
            "names": ["channels", "height", "width"],
            "info": video_info,
        },
    }

    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "features": features,
        "total_episodes": len(demo_keys),
        "total_frames": global_index,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "robot_type": robot_type,
        "splits": {"train": f"0:{len(demo_keys)}"},
    }
    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    with open(meta_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(_serialize_stats(global_stats), f, indent=2)

    print(f"[convert] 완료: {output_dir}")
    print(f"  episodes: {len(demo_keys)}, frames: {global_index}")
    print(f"  video: {video_out} ({video_out.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Isaac Lab HDF5 → LeRobot v3.0")
    parser.add_argument("hdf5_path", type=Path, help="입력 HDF5 (record_demos 출력)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="출력 디렉터리 (기본: <hdf5_stem>_lerobot_v3)",
    )
    parser.add_argument("--repo-id", default="local/g1_pickplace", help="LeRobot repo_id")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="제어 주파수 (Hz)")
    parser.add_argument(
        "--task",
        default="pick and place steering wheel",
        help="단일 태스크 설명 (task-conditioned 학습용)",
    )
    parser.add_argument("--robot-type", default="unitree_g1")
    parser.add_argument(
        "--no-lerobot",
        action="store_true",
        help="lerobot가 설치되어 있어도 내장 변환기만 사용",
    )
    args = parser.parse_args()

    hdf5_path = args.hdf5_path.resolve()
    if not hdf5_path.is_file():
        print(f"파일 없음: {hdf5_path}", file=sys.stderr)
        return 1

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = hdf5_path.parent / f"{hdf5_path.stem}_lerobot_v3"
    output_dir = output_dir.resolve()

    if not args.no_lerobot and _try_lerobot_convert(
        hdf5_path, output_dir, args.repo_id, args.fps, args.task, args.robot_type
    ):
        return 0

    _standalone_convert(hdf5_path, output_dir, args.fps, args.task, args.robot_type)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
