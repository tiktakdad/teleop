#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Isaac Lab HDF5 → LeRobot v3.0 변환 (conda isaaclab310 + 공식 LeRobot API)
#
# 사용:
#   ./scripts/convert_hdf5_to_lerobot.sh workspace/datasets/dataset_g1_260520_0652.hdf5
#   ./scripts/convert_hdf5_to_lerobot.sh path/to/file.hdf5 --task "pick orange"
#
# 출력 디렉터리 (자동):
#   <hdf5와_같은_폴더>/<파일명>_lerobot/
#   예) dataset_g1_260520_0652.hdf5 → dataset_g1_260520_0652_lerobot/
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONDA_ENV="${LEROBOT_CONDA_ENV:-isaaclab310}"
CONVERT_PY="${SCRIPT_DIR}/convert_isaac_hdf5_to_lerobot.py"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

usage() {
    cat <<EOF
사용법: $(basename "$0") <hdf5_path> [추가 인자...]

  hdf5_path   Isaac Lab record_demos 로 수집한 .hdf5 파일

출력 (자동):
  <hdf5와 같은 폴더>/<파일명>_lerobot/

예:
  $(basename "$0") workspace/datasets/dataset_g1_260520_0652.hdf5
  $(basename "$0") workspace/datasets/foo.hdf5 --task "pick and place" --fps 50

추가 인자는 convert_isaac_hdf5_to_lerobot.py 로 그대로 전달됩니다.
  --output-dir <path>   출력 경로 수동 지정 (기본: 자동)
  --repo-id <id>        LeRobot repo_id (기본: local/<파일명>)
  --task <text>         태스크 설명
  --fps <hz>            FPS (기본: 50)
  --push-to-hub         Hugging Face Hub 업로드

환경:
  conda 환경 \${LEROBOT_CONDA_ENV:-isaaclab310} (lerobot 0.4+ 필요)
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
OUTPUT_DIR="${HDF5_DIR}/${HDF5_STEM}_lerobot"

info "프로젝트: $PROJECT_DIR"
info "입력:     $HDF5_PATH"
info "출력:     $OUTPUT_DIR (자동)"
info "conda:    $CONDA_ENV"

# conda 활성화
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
    err "conda.sh 를 찾을 수 없습니다."
    exit 1
fi

if ! conda activate "$CONDA_ENV" 2>/dev/null; then
    err "conda 환경 '$CONDA_ENV' 활성화 실패"
    exit 1
fi

if ! python -c "import lerobot" 2>/dev/null; then
    err "lerobot 패키지 없음. 예: conda activate $CONDA_ENV && pip install lerobot h5py"
    exit 1
fi

LEROBOT_VER="$(python -c "import lerobot; print(lerobot.__version__)")"
log "lerobot $LEROBOT_VER"

if [[ -d "$OUTPUT_DIR" ]]; then
    warn "출력 폴더가 이미 있습니다: $OUTPUT_DIR"
    warn "삭제 후 재실행하거나 --output-dir 로 다른 경로를 지정하세요."
    exit 1
fi

cd "$PROJECT_DIR"
python "$CONVERT_PY" "$HDF5_PATH" "$@"

log "변환 완료: $OUTPUT_DIR"
