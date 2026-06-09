#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Isaac Teleop 통합 환경 초기 셋업 스크립트
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*"; }
info() { echo -e "${CYAN}[i]${NC} $*"; }

echo ""
echo "=============================================="
echo "  Isaac Teleop 통합 환경 셋업"
echo "=============================================="
echo ""

# ── 1. 사전 요구사항 확인 ──────────────────────────────────────────────────
check_prereqs() {
    local ok=true

    info "사전 요구사항 확인 중..."

    if ! command -v docker &>/dev/null; then
        err "Docker가 설치되어 있지 않습니다."
        err "  → https://docs.docker.com/engine/install/"
        ok=false
    else
        local docker_ver
        docker_ver=$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo "unknown")
        log "Docker: v${docker_ver}"
    fi

    if ! docker compose version &>/dev/null; then
        err "Docker Compose v2가 설치되어 있지 않습니다."
        err "  → https://docs.docker.com/compose/install/"
        ok=false
    else
        local compose_ver
        compose_ver=$(docker compose version --short 2>/dev/null || echo "unknown")
        log "Docker Compose: v${compose_ver}"
    fi

    if ! command -v nvidia-smi &>/dev/null; then
        warn "nvidia-smi를 찾을 수 없습니다. GPU 드라이버를 확인하세요."
    else
        local gpu_info
        gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        log "GPU: ${gpu_info}"
    fi

    if ! docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
        err "NVIDIA Container Toolkit이 정상 동작하지 않습니다."
        err "  → https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        ok=false
    else
        log "NVIDIA Container Toolkit: OK"
    fi

    if [ "$ok" = false ]; then
        echo ""
        err "사전 요구사항을 충족하지 못했습니다. 위의 안내를 따라 설치해 주세요."
        exit 1
    fi

    echo ""
    log "모든 사전 요구사항 충족!"
}

# ── 2. 방화벽 포트 설정 ───────────────────────────────────────────────────
setup_firewall() {
    info "방화벽 포트 설정 중 (Meta Quest 3 WebXR 연결용)..."

    if command -v ufw &>/dev/null; then
        # CloudXR Runtime (Quest 3 ↔ 워크스테이션)
        sudo ufw allow 47998/udp comment "CloudXR media streaming" 2>/dev/null || true
        sudo ufw allow 49100/tcp comment "CloudXR WSS signaling" 2>/dev/null || true
        sudo ufw allow 48322/tcp comment "CloudXR WSS proxy" 2>/dev/null || true
        # WebXR Client 서빙
        sudo ufw allow 8080/tcp comment "WebXR HTTP" 2>/dev/null || true
        sudo ufw allow 8443/tcp comment "WebXR HTTPS" 2>/dev/null || true
        log "UFW 방화벽 규칙 추가 완료 (Quest 3 WebXR)"
    else
        warn "ufw가 설치되어 있지 않습니다. 수동으로 포트를 열어주세요:"
        warn "  UDP: 47998"
        warn "  TCP: 49100, 48322, 8080, 8443"
    fi
}

# ── 3. 환경 변수 설정 ─────────────────────────────────────────────────────
setup_env() {
    info "환경 변수 확인 중..."

    if [ ! -f "$PROJECT_DIR/.env" ]; then
        err ".env 파일이 없습니다."
        exit 1
    fi

    # UID/GID 자동 설정
    sed -i "s/^HOST_UID=.*/HOST_UID=$(id -u)/" "$PROJECT_DIR/.env"
    sed -i "s/^HOST_GID=.*/HOST_GID=$(id -g)/" "$PROJECT_DIR/.env"
    log "HOST_UID/HOST_GID 설정: $(id -u):$(id -g)"

    # DISPLAY 설정
    if [ -n "${DISPLAY:-}" ]; then
        sed -i "s/^DISPLAY=.*/DISPLAY=${DISPLAY}/" "$PROJECT_DIR/.env"
        log "DISPLAY 설정: ${DISPLAY}"
    fi
}

# ── 4. X11 포워딩 ────────────────────────────────────────────────────────
setup_x11() {
    info "X11 포워딩 설정 중..."
    if command -v xhost &>/dev/null; then
        xhost +SI:localuser:root 2>/dev/null || true
        log "X11 포워딩 활성화 (localuser:root)"
    else
        warn "xhost가 없습니다. GUI가 필요하면 xhost를 설치하세요."
    fi
}

# ── 5. 작업 디렉터리 생성 ─────────────────────────────────────────────────
setup_dirs() {
    info "작업 디렉터리 생성 중..."
    mkdir -p "$PROJECT_DIR/workspace" "$PROJECT_DIR/assets"
    log "workspace/, assets/ 디렉터리 생성 완료"
}

# ── 5-1. 로봇 USD 에셋 다운로드 (최초 1회) ────────────────────────────────
download_robot_assets() {
    info "로봇 USD 에셋 확인 중..."
    if [ -f "$PROJECT_DIR/assets/GR1T2_fourier_hand_6dof/GR1T2_fourier_hand_6dof.usd" ] \
        && [ -f "$PROJECT_DIR/assets/Isaac/Props/PackingTable/packing_table.usd" ]; then
        log "Nucleus 에셋 이미 존재 (스킵)"
    else
        info "GR1T2 USD 에셋 다운로드 중 (최초 1회, 이후 오프라인 사용 가능)..."
        bash "$SCRIPT_DIR/download_assets.sh" || {
            warn "에셋 다운로드 실패. 나중에 수동으로 실행하세요:"
            warn "  ./scripts/download_assets.sh"
        }
    fi
}

# ── 6. Docker 이미지 풀 ──────────────────────────────────────────────────
pull_images() {
    info "NGC 이미지 다운로드 중 (시간이 걸릴 수 있습니다)..."
    echo ""

    source "$PROJECT_DIR/.env"

    docker pull "nvcr.io/nvidia/isaac-lab:${ISAAC_LAB_VERSION}" || {
        warn "Isaac Lab 이미지 풀 실패. 네트워크 연결을 확인하세요."
    }

    # CloudXR Runtime은 isaac-teleop 이미지에 포함되어 별도 풀 불필요
}

# ── 7. 커스텀 이미지 빌드 ─────────────────────────────────────────────────
build_custom() {
    info "커스텀 Docker 이미지 빌드 중..."
    cd "$PROJECT_DIR"

    local max_retries=3
    local attempt=1

    while [ $attempt -le $max_retries ]; do
        info "빌드 시도 ${attempt}/${max_retries}..."
        if docker compose build isaac-teleop webxr-client; then
            log "isaac-teleop, webxr-client 이미지 빌드 완료"
            return 0
        fi

        if [ $attempt -lt $max_retries ]; then
            warn "빌드 실패. 30초 후 재시도합니다..."
            sleep 30
        fi
        attempt=$((attempt + 1))
    done

    err "빌드가 ${max_retries}회 모두 실패했습니다. 네트워크 연결을 확인하세요."
    exit 1
}

# ── 메인 ──────────────────────────────────────────────────────────────────
main() {
    check_prereqs
    setup_firewall
    setup_env
    setup_x11
    setup_dirs
    download_robot_assets
    pull_images
    build_custom

    echo ""
    echo "=============================================="
    echo "  셋업 완료!"
    echo "=============================================="
    echo ""
    info "다음 단계:"
    echo "  1. .env 파일에서 CXR_PUBLIC_IP를 서버 공인 IP로 설정"
    echo "  2. 라우터에서 포트포워딩 설정 (TCP 8453, 48322 / UDP 47998)"
    echo ""
    info "실행 명령어:"
    echo "  docker compose up -d               # 전체 실행"
    echo ""
    info "Quest 3 접속:"
    echo "  https://<서버_공인_IP>:8453"
    echo ""
    info "로그 확인:"
    echo "  docker compose logs -f"
    echo ""
    info "종료:"
    echo "  docker compose down"
    echo ""
}

main "$@"
