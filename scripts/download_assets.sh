#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Isaac PickPlace GR1T2 텔레오프 — Nucleus 에셋 로컬 캐시
# S3 목록 조회 후 USD·메시·텍스처 등 관련 파일 전체 저장
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ASSETS_DIR="$PROJECT_DIR/assets"

S3_HOST="omniverse-content-production.s3-us-west-2.amazonaws.com"
S3_ISAAC_PREFIX="Assets/Isaac/5.1/Isaac"
BASE_URL="https://${S3_HOST}"

# 🔹 다운로드할 에셋 번들 (S3 경로 → 로컬 저장 경로)
# 형식: "S3상대경로|로컬상대경로"
ASSET_BUNDLES=(
    "Robots/FourierIntelligence/GR-1/GR1T2_fourier_hand_6dof|GR1T2_fourier_hand_6dof"
    "Props/PackingTable|Isaac/Props/PackingTable"
    "IsaacLab/Mimic/pick_place_task/pick_place_assets|Isaac/IsaacLab/Mimic/pick_place_task/pick_place_assets"
)

CURL_OPTS=(-fsSL --retry 3 --connect-timeout 60)
INCLUDE_THUMBS=false
FORCE=false

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*"; }
info() { echo -e "${CYAN}[i]${NC} $*"; }

usage() {
    cat <<EOF
사용법: $(basename "$0") [옵션]

PickPlace GR1T2 텔레오프에 필요한 Nucleus 에셋을 로컬에 저장합니다:
  - GR1T2 로봇
  - PackingTable (작업대)
  - steering_wheel 등 pick_place 에셋

옵션:
  --force         이미 받은 파일도 다시 다운로드
  --with-thumbs   Omniverse 썸네일(.thumbs)까지 포함
  -h, --help      도움말
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --force) FORCE=true; shift ;;
            --with-thumbs) INCLUDE_THUMBS=true; shift ;;
            -h|--help) usage; exit 0 ;;
            *) err "알 수 없는 옵션: $1"; usage; exit 1 ;;
        esac
    done
}

ensure_writable() {
    if mkdir -p "$ASSETS_DIR" 2>/dev/null && touch "$ASSETS_DIR/.write_test" 2>/dev/null; then
        rm -f "$ASSETS_DIR/.write_test"
        return 0
    fi
    err "에셋 저장 경로에 쓰기 권한이 없습니다: $ASSETS_DIR"
    exit 1
}

fetch_s3_keys() {
    local list_prefix="$1"
    local url="https://${S3_HOST}/?list-type=2&prefix=${list_prefix}"
    local xml
    xml=$(curl "${CURL_OPTS[@]}" -k "$url") || return 1
    echo "$xml" | grep -oP '(?<=<Key>)[^<]+' || true
}

should_download_key() {
    local rel="$1"
    [[ -z "$rel" ]] && return 1
    [[ "$rel" == */ ]] && return 1
    if [[ "$INCLUDE_THUMBS" == false ]]; then
        [[ "$rel" == .thumbs/* || "$rel" == */.thumbs/* ]] && return 1
        [[ "$rel" == *.last_generated ]] && return 1
        [[ "$rel" == *.thumb.usd ]] && return 1
        [[ "$rel" == *.auto.png ]] && return 1
    fi
    return 0
}

download_bundle() {
    local s3_rel="$1"
    local local_rel="$2"
    local list_prefix="${S3_ISAAC_PREFIX}/${s3_rel}/"
    local dest_root="${ASSETS_DIR}/${local_rel}"

    info "번들: ${s3_rel}"
    info "  → ${dest_root}/"

    mapfile -t ALL_KEYS < <(fetch_s3_keys "$list_prefix") || {
        warn "목록 조회 실패: ${s3_rel}"
        return 1
    }

    local keys=()
    for key in "${ALL_KEYS[@]}"; do
        local rel="${key#${list_prefix}}"
        if should_download_key "$rel"; then
            keys+=("$key")
        fi
    done

    local total=${#keys[@]}
    if [[ $total -eq 0 ]]; then
        warn "  다운로드할 파일 없음"
        return 1
    fi

    info "  ${total}개 파일 다운로드..."
    local i=0 failed=0
    for key in "${keys[@]}"; do
        i=$((i + 1))
        local rel="${key#${list_prefix}}"
        local dest="${dest_root}/${rel}"
        local url="${BASE_URL}/${key}"

        if [[ "$FORCE" == false && -f "$dest" ]]; then
            continue
        fi

        mkdir -p "$(dirname "$dest")"
        printf "\r  [%d/%d] %s" "$i" "$total" "$rel"
        if ! curl "${CURL_OPTS[@]}" -k -o "$dest" "$url" 2>/dev/null; then
            echo ""
            warn "  실패: $rel"
            failed=$((failed + 1))
        fi
    done
    echo ""
    if [[ $failed -gt 0 ]]; then
        warn "  ${failed}개 실패"
    else
        log "  완료 ($(du -sh "$dest_root" | cut -f1))"
    fi
}

is_complete() {
    [[ -f "${ASSETS_DIR}/GR1T2_fourier_hand_6dof/GR1T2_fourier_hand_6dof.usd" ]] \
        && [[ -f "${ASSETS_DIR}/Isaac/Props/PackingTable/packing_table.usd" ]] \
        && [[ -f "${ASSETS_DIR}/Isaac/IsaacLab/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd" ]]
}

main() {
    parse_args "$@"

    echo ""
    echo "=============================================="
    echo "  PickPlace GR1T2 Nucleus 에셋 다운로드"
    echo "=============================================="
    echo ""

    if [[ "$FORCE" == false ]] && is_complete; then
        log "모든 필수 에셋이 이미 존재합니다: ${ASSETS_DIR}/"
        info "재다운로드: ./scripts/download_assets.sh --force"
        exit 0
    fi

    command -v curl &>/dev/null || { err "curl이 필요합니다."; exit 1; }
    ensure_writable

    for bundle in "${ASSET_BUNDLES[@]}"; do
        local s3_rel="${bundle%%|*}"
        local local_rel="${bundle##*|}"
        echo ""
        download_bundle "$s3_rel" "$local_rel" || true
    done

    echo ""
    if is_complete; then
        log "전체 다운로드 완료!"
        log "  로봇:  assets/GR1T2_fourier_hand_6dof/"
        log "  테이블: assets/Isaac/Props/PackingTable/"
        log "  물체:  assets/Isaac/IsaacLab/Mimic/pick_place_task/pick_place_assets/"
        du -sh "$ASSETS_DIR" | awk '{print "  총 용량: " $1}'
    else
        err "필수 파일 누락. --force 로 재실행하세요."
        exit 1
    fi

    echo ""
    info "이후 docker compose up 시 외부망 없이 로컬 에셋을 사용합니다."
    echo ""
}

main "$@"
