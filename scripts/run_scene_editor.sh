#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Launch the Isaac Sim scene editor from the Isaac Lab Docker image.
#
# This bypasses the teleoperation entrypoint and does not start CloudXR.
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="${PROJECT_DIR}/docker-compose.yml"

WRITABLE_CUSTOM_ASSETS=true
LOCAL_ASSETS=false
OPEN_STAGE=""
EXTRA_ARGS=()

usage() {
    cat <<EOF
사용법: $(basename "$0") [옵션] [-- Isaac Sim 인자...]

현재 Docker Isaac Lab 이미지로 Isaac Sim GUI scene editor만 실행합니다.
CloudXR 및 teleop task는 시작하지 않습니다.

옵션:
  --open <usd>                 시작할 USD 경로
                               예: custom_assets/env/foo.usd
  --writable-custom-assets     custom_assets 쓰기 허용 (기본값, 호환 옵션)
  --read-only-custom-assets    custom_assets를 읽기 전용으로 마운트
  --local-assets               assets/를 Isaac asset root로 사용
  -h, --help                   도움말

저장 경로:
  기본 권장: /workspace/user/scenes/<scene>.usd
             (호스트의 workspace/scenes/에 저장됨)
  custom_assets/는 기본적으로 직접 저장할 수 있습니다.

예:
  ./scripts/run_scene_editor.sh
  ./scripts/run_scene_editor.sh --local-assets
  ./scripts/run_scene_editor.sh --open custom_assets/env/server_rack_v6.1/server_rack_teleop.usd
EOF
}

err() {
    echo "[scene-editor] ERROR: $*" >&2
}

info() {
    echo "[scene-editor] $*"
}

to_container_path() {
    local path="$1"

    case "$path" in
        /workspace/user/*)
            printf '%s\n' "$path"
            ;;
        "$PROJECT_DIR"/custom_assets/*)
            printf '/workspace/user/custom_assets/%s\n' "${path#"$PROJECT_DIR"/custom_assets/}"
            ;;
        "$PROJECT_DIR"/assets/*)
            printf '/workspace/user/assets/%s\n' "${path#"$PROJECT_DIR"/assets/}"
            ;;
        "$PROJECT_DIR"/workspace/*)
            printf '/workspace/user/%s\n' "${path#"$PROJECT_DIR"/workspace/}"
            ;;
        custom_assets/*)
            printf '/workspace/user/%s\n' "$path"
            ;;
        assets/*)
            printf '/workspace/user/%s\n' "$path"
            ;;
        workspace/*)
            printf '/workspace/user/%s\n' "${path#workspace/}"
            ;;
        *)
            err "--open은 custom_assets/, assets/, workspace/ 또는 /workspace/user/ 경로를 사용하세요: $path"
            exit 1
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --open)
            [[ $# -ge 2 ]] || { err "--open에는 USD 경로가 필요합니다."; exit 1; }
            OPEN_STAGE="$(to_container_path "$2")"
            shift 2
            ;;
        --writable-custom-assets)
            WRITABLE_CUSTOM_ASSETS=true
            shift
            ;;
        --read-only-custom-assets)
            WRITABLE_CUSTOM_ASSETS=false
            shift
            ;;
        --local-assets)
            LOCAL_ASSETS=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            err "알 수 없는 옵션: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ ! -f "$COMPOSE_FILE" ]]; then
    err "docker-compose.yml을 찾을 수 없습니다: $COMPOSE_FILE"
    exit 1
fi

if ! command -v docker >/dev/null 2>&1 || ! docker compose version >/dev/null 2>&1; then
    err "Docker Compose v2가 필요합니다."
    exit 1
fi

HOST_DISPLAY="${DISPLAY:-}"
if [[ -z "$HOST_DISPLAY" && -f "$PROJECT_DIR/.env" ]]; then
    HOST_DISPLAY="$(sed -n 's/^DISPLAY=//p' "$PROJECT_DIR/.env" | tail -n 1)"
fi
HOST_DISPLAY="${HOST_DISPLAY:-:0}"

if command -v xhost >/dev/null 2>&1; then
    if ! DISPLAY="$HOST_DISPLAY" xhost +local:docker >/dev/null; then
        err "X display에 접속할 수 없습니다: DISPLAY=$HOST_DISPLAY"
        exit 1
    fi
else
    err "xhost 명령이 없습니다. GUI용 X11 접근을 설정할 수 없습니다."
    exit 1
fi

mkdir -p "$PROJECT_DIR/workspace/scenes"

# The Isaac Lab container runs as root; Kit requires this opt-in for GUI startup.
KIT_ARGS=(--allow-root)
if [[ "$LOCAL_ASSETS" == true ]]; then
    KIT_ARGS+=(
        "--/persistent/isaac/asset_root/cloud=file:///workspace/user/assets"
        "--/persistent/isaac/asset_root/default=file:///workspace/user/assets"
        "--/persistent/isaac/asset_root/nvidia=file:///workspace/user/assets"
    )
fi
if [[ -n "$OPEN_STAGE" ]]; then
    KIT_ARGS+=("$OPEN_STAGE")
fi
KIT_ARGS+=("${EXTRA_ARGS[@]}")

info "Isaac Sim scene editor 시작 (DISPLAY=$HOST_DISPLAY)"
info "새 씬 저장 경로: /workspace/user/scenes/  (host: workspace/scenes/)"
if [[ "$WRITABLE_CUSTOM_ASSETS" == true ]]; then
    info "custom_assets 쓰기 허용됨 (host: custom_assets/)"
else
    info "custom_assets 읽기 전용 모드"
fi

cd "$PROJECT_DIR"
SIM_COMMAND='cd /workspace/isaaclab && exec ./isaaclab.sh -s "$@"'

if [[ "$WRITABLE_CUSTOM_ASSETS" == true ]]; then
    printf '%s\n' \
        'services:' \
        '  isaac-lab:' \
        '    volumes:' \
        '      - ./custom_assets:/workspace/user/custom_assets:rw' |
        DISPLAY="$HOST_DISPLAY" docker compose -f "$COMPOSE_FILE" -f - run --rm --no-deps \
            --entrypoint /bin/bash isaac-lab -lc "$SIM_COMMAND" scene-editor "${KIT_ARGS[@]}"
else
    DISPLAY="$HOST_DISPLAY" docker compose -f "$COMPOSE_FILE" run --rm --no-deps \
        --entrypoint /bin/bash isaac-lab -lc "$SIM_COMMAND" scene-editor "${KIT_ARGS[@]}"
fi
