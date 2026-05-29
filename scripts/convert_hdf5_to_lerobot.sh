#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Isaac Lab HDF5 → LeRobot v3.0 변환 (conda isaaclab310 + LeRobot 0.4+)
#
# 사용:
#   ./scripts/convert_hdf5_to_lerobot.sh workspace/datasets/dataset_barcode_260529_0617.hdf5
#   ./scripts/convert_hdf5_to_lerobot.sh path/to/file.hdf5 --task "press the barcode button"
#
# 출력 디렉터리 (자동):
#   <hdf5와_같은_폴더>/<파일명>_lerobot_v3/
#   예) dataset_barcode_260529_0617.hdf5 → dataset_barcode_260529_0617_lerobot_v3/
#
# 변환 본체: scripts/convert_isaac_hdf5_to_lerobot_v3.py
#   - obs/ 그룹의 카메라(head_cam / left_hand_cam / right_hand_cam 등)를 자동 탐지
#   - lerobot 설치 시 공식 API(LeRobotDataset.create) 사용, 없으면 내장 ffmpeg 폴백
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONDA_ENV="${LEROBOT_CONDA_ENV:-isaaclab310}"
CONVERT_PY="${SCRIPT_DIR}/convert_isaac_hdf5_to_lerobot_v3.py"

# HDF5 파일이 다른 프로세스에 의해 열려 있어도 읽기 가능하도록 잠금 비활성화
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

usage() {
    cat <<EOF
사용법: $(basename "$0") <hdf5_path> [추가 인자...]

  hdf5_path   Isaac Lab record_demos / teleop_barcode_ffw --record 로 수집한 .hdf5 파일

출력 (자동):
  <hdf5와 같은 폴더>/<파일명>_lerobot_v3/

예:
  $(basename "$0") workspace/datasets/dataset_barcode_260529_0617.hdf5
  $(basename "$0") workspace/datasets/foo.hdf5 --task "press the barcode button" --fps 30

추가 인자는 convert_isaac_hdf5_to_lerobot_v3.py 로 그대로 전달됩니다.
  --output-dir <path>   출력 경로 수동 지정 (기본: 자동)
  --repo-id <id>        LeRobot repo_id (기본: local/<파일명>)
  --task <text>         태스크 설명
  --fps <hz>            FPS (기본: 50, RECORD_FPS=15로 수집했다면 --fps 15)
  --robot-type <name>   로봇 타입 (기본: unitree_g1, FFW는 ffw_sg2)
  --no-lerobot          lerobot 설치돼 있어도 내장 ffmpeg 폴백만 사용

환경:
  conda 환경 \${LEROBOT_CONDA_ENV:-isaaclab310} (lerobot 0.4+ 권장)
EOF
}

log()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*"; }
info() { echo -e "${CYAN}[i]${NC} $*"; }

if [[ $# -lt 1 ]] || [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

HDF5_ARG="$1"
shift

# 사용자가 넘긴 추가 인자에서 --fps / --output-dir 지정 여부 확인
HAS_FPS_ARG=0
HAS_OUTPUT_ARG=0
for arg in "$@"; do
    case "$arg" in
        --fps|--fps=*)               HAS_FPS_ARG=1 ;;
        --output-dir|--output-dir=*) HAS_OUTPUT_ARG=1 ;;
    esac
done

# --fps 미지정 시 .env 의 RECORD_FPS 를 반영
if [[ "$HAS_FPS_ARG" -eq 0 && -f "$PROJECT_DIR/.env" ]]; then
    ENV_RECORD_FPS="$(grep -E '^RECORD_FPS=' "$PROJECT_DIR/.env" | tail -n 1 | cut -d= -f2- | tr -d '"' | tr -d "'")"
    if [[ -n "${ENV_RECORD_FPS:-}" ]]; then
        info "RECORD_FPS=$ENV_RECORD_FPS (.env) → --fps $ENV_RECORD_FPS"
        set -- "$@" --fps "$ENV_RECORD_FPS"
    fi
fi

# 절대 경로로 정규화
if [[ "$HDF5_ARG" = /* ]]; then
    HDF5_PATH="$HDF5_ARG"
else
    HDF5_PATH="$(cd "$PROJECT_DIR" && realpath "$HDF5_ARG" 2>/dev/null || echo "$PROJECT_DIR/$HDF5_ARG")"
fi

if [[ ! -f "$HDF5_PATH" ]]; then
    err "HDF5 파일 없음: $HDF5_PATH"
    exit 1
fi

HDF5_STEM="$(basename "$HDF5_PATH" .hdf5)"
HDF5_DIR="$(dirname "$HDF5_PATH")"
OUTPUT_DIR="${HDF5_DIR}/${HDF5_STEM}_lerobot_v3"

info "프로젝트: $PROJECT_DIR"
info "입력:     $HDF5_PATH"
if [[ "$HAS_OUTPUT_ARG" -eq 0 ]]; then
    info "출력:     $OUTPUT_DIR (자동)"
else
    info "출력:     (--output-dir 로 지정됨)"
fi
info "conda:    $CONDA_ENV"

# conda 활성화
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
    err "conda.sh 를 찾을 수 없습니다. (miniconda3/anaconda3)"
    exit 1
fi

if ! conda activate "$CONDA_ENV" 2>/dev/null; then
    err "conda 환경 '$CONDA_ENV' 활성화 실패"
    err "예: conda create -n $CONDA_ENV python=3.10 && conda activate $CONDA_ENV && pip install lerobot h5py"
    exit 1
fi

# h5py 는 필수, lerobot 은 권장(없으면 내장 ffmpeg 폴백)
if ! python -c "import h5py" 2>/dev/null; then
    err "h5py 패키지 없음. 예: conda activate $CONDA_ENV && pip install h5py"
    exit 1
fi

if python -c "import lerobot" 2>/dev/null; then
    LEROBOT_VER="$(python -c "import lerobot; print(lerobot.__version__)")"
    log "lerobot $LEROBOT_VER (공식 API 사용)"
else
    warn "lerobot 미설치 → 내장 ffmpeg 폴백 변환기 사용"
    if ! command -v ffmpeg >/dev/null 2>&1; then
        err "ffmpeg 도 없습니다. lerobot 또는 ffmpeg 중 하나가 필요합니다."
        exit 1
    fi
fi

# 자동 출력 경로가 이미 있으면 중단 (사용자가 --output-dir 지정 시엔 스크립트에 위임)
if [[ "$HAS_OUTPUT_ARG" -eq 0 && -e "$OUTPUT_DIR" ]]; then
    warn "출력 폴더가 이미 있습니다: $OUTPUT_DIR"
    warn "삭제 후 재실행하거나 --output-dir 로 다른 경로를 지정하세요."
    exit 1
fi

cd "$PROJECT_DIR"
if [[ "$HAS_OUTPUT_ARG" -eq 0 ]]; then
    python "$CONVERT_PY" "$HDF5_PATH" --output-dir "$OUTPUT_DIR" "$@"
    log "변환 완료: $OUTPUT_DIR"
else
    python "$CONVERT_PY" "$HDF5_PATH" "$@"
    log "변환 완료"
fi
