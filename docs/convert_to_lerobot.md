# Isaac Lab HDF5 → LeRobot 변환

Isaac Lab `record_demos.py`로 수집한 HDF5를 **LeRobot Dataset v3.0** 형식으로 변환합니다.  
변환은 **lerobot 공식 API** (`LeRobotDataset.create` / `add_frame` / `save_episode` / `finalize`)만 사용합니다.

## 사전 요구사항

- conda 환경 `isaaclab310` (또는 `lerobot>=0.4`, `h5py` 설치된 환경)
- HDF5에 G1 로봇 POV 카메라가 포함되어 있어야 함 (`obs/robot_pov_cam`)

```bash
conda activate isaaclab310
python -c "import lerobot; print(lerobot.__version__)"   # 예: 0.4.4
```

## 빠른 사용 (권장)

```bash
# 프로젝트 루트에서
./scripts/convert_hdf5_to_lerobot.sh workspace/datasets/dataset_g1_260520_0652.hdf5
```

### 출력 경로 (자동)

입력 파일과 **같은 폴더**에 `<파일명>_lerobot` 디렉터리가 생성됩니다.

| 입력 | 출력 (자동) |
|------|-------------|
| `workspace/datasets/dataset_g1_260520_0652.hdf5` | `workspace/datasets/dataset_g1_260520_0652_lerobot/` |

`repo_id`도 기본값 `local/<파일명>` 으로 자동 설정됩니다.

## 셸 스크립트 옵션

```bash
./scripts/convert_hdf5_to_lerobot.sh <hdf5_path> [추가 인자...]
```

| 인자 | 설명 |
|------|------|
| `--task "..."` | 태스크 설명 (기본: `pick and place steering wheel`) |
| `--fps 50` | 제어 주파수 Hz |
| `--output-dir <path>` | 출력 경로 수동 지정 (기본: 자동) |
| `--repo-id local/my_dataset` | LeRobot repo_id 수동 지정 |
| `--push-to-hub` | Hugging Face Hub 업로드 |

다른 conda 환경을 쓰려면:

```bash
LEROBOT_CONDA_ENV=myenv ./scripts/convert_hdf5_to_lerobot.sh path/to/file.hdf5
```

## Python 직접 실행

```bash
conda activate isaaclab310
cd /path/to/teleop

python scripts/convert_isaac_hdf5_to_lerobot.py \
  workspace/datasets/dataset_g1_260520_0652.hdf5
```

`--output-dir`을 생략하면 셸 스크립트와 동일하게 `<파일명>_lerobot` 폴더가 생성됩니다.

## 변환 결과 구조

```
dataset_g1_260520_0652_lerobot/
├── data/chunk-000/file-000.parquet       # observation.state, action
├── videos/observation.images.robot_pov/  # 로봇 POV (AV1 mp4)
└── meta/
    ├── info.json
    ├── stats.json
    ├── tasks.parquet
    └── episodes/chunk-000/file-000.parquet
```

### LeRobot feature 매핑

| LeRobot 키 | HDF5 소스 |
|------------|-----------|
| `observation.state` | `obs/robot_joint_pos` |
| `action` | `processed_actions` |
| `observation.images.robot_pov` | `obs/robot_pov_cam` (1280×720 등) |
| `task` | CLI `--task` 문자열 |

## 메타데이터 확인

```bash
conda activate isaaclab310
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
m = LeRobotDatasetMetadata(
    'local/dataset_g1_260520_0652',
    root='workspace/datasets/dataset_g1_260520_0652_lerobot',
)
print('episodes:', m.total_episodes, 'frames:', m.total_frames, 'fps:', m.fps)
"
```

## 주의사항

- **출력 폴더가 이미 있으면** 변환이 중단됩니다. 덮어쓰려면 기존 폴더를 삭제한 뒤 다시 실행하세요.
- 구버전 `scripts/convert_isaac_hdf5_to_lerobot_v3.py`는 lerobot 미설치 시 ffmpeg 폴백이 있는 **비공식** 변환기입니다. 일반적으로 본 문서의 스크립트를 사용하세요.
- LeIsaac `isaaclab2lerobotv3.py`는 Isaac Sim `AppLauncher`가 필요해, 호스트 conda 환경의 HDF5 일괄 변환에는 **본 프로젝트 스크립트**가 적합합니다.

## 관련 파일

| 파일 | 역할 |
|------|------|
| `scripts/convert_hdf5_to_lerobot.sh` | conda 활성화 + 원클릭 변환 |
| `scripts/convert_isaac_hdf5_to_lerobot.py` | LeRobot 공식 API 변환 본체 |
| `scripts/convert_isaac_hdf5_to_lerobot_v3.py` | [DEPRECATED] 내장 ffmpeg 폴백 포함 |
