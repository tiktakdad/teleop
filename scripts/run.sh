#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_DIR/.env"

# Colors (safe fallback when terminal does not support colors)
if [[ -t 1 ]] && command -v tput >/dev/null 2>&1; then
    BOLD="$(tput bold || true)"
    DIM="$(tput dim || true)"
    RESET="$(tput sgr0 || true)"
    FG_CYAN="$(tput setaf 6 || true)"
    FG_GREEN="$(tput setaf 2 || true)"
    FG_YELLOW="$(tput setaf 3 || true)"
    FG_RED="$(tput setaf 1 || true)"
    FG_WHITE="$(tput setaf 7 || true)"
    BG_CYAN="$(tput setab 6 || true)"
else
    BOLD=""
    DIM=""
    RESET=""
    FG_CYAN=""
    FG_GREEN=""
    FG_YELLOW=""
    FG_RED=""
    FG_WHITE=""
    BG_CYAN=""
fi

print_banner() {
    echo
    echo "${FG_CYAN}${BOLD}+--------------------------------------------------------------+${RESET}"
    echo "${FG_CYAN}${BOLD}|                   Isaac Teleop Launcher                        |${RESET}"
    echo "${FG_CYAN}${BOLD}+--------------------------------------------------------------+${RESET}"
}

print_section() {
    local title="$1"
    echo
    echo "${FG_WHITE}${BOLD}[$title]${RESET}"
}

info() {
    echo "${FG_GREEN}[run]${RESET} $*"
}

warn() {
    echo "${FG_YELLOW}[run]${RESET} $*"
}

err() {
    echo "${FG_RED}[run]${RESET} $*" >&2
}

is_true() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

load_env_value() {
    local key="$1"
    local file="$2"
    [[ -f "$file" ]] || return 1
    sed -n "s/^${key}=//p" "$file" | tail -n 1
}

clean_optional_env_keys() {
    local file="$1"
    [[ -f "$file" ]] || return 0
    sed -i \
        -e '/^XR_HEADLESS=/d' \
        -e '/^RUN_MODE=/d' \
        -e '/^TELEOP_TASK=/d' \
        "$file"
}

prompt_value() {
    local label="$1"
    local default_val="$2"
    local result
    echo -ne "${FG_WHITE}${label}${RESET} [${default_val}]: " >&2
    read -r result
    printf '%s\n' "${result:-$default_val}"
}

# Numeric fallback for non-interactive stdin
choose_from_menu_number() {
    local title="$1"
    shift
    local options=("$@")
    local choice

    print_section "$title" >&2
    local i=1
    for item in "${options[@]}"; do
        echo "  $i) $item" >&2
        i=$((i + 1))
    done

    while true; do
        echo -n "> 번호 선택: " >&2
        read -r choice
        if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#options[@]} )); then
            printf '%s\n' "${options[$((choice - 1))]}"
            return 0
        fi
        warn "잘못된 입력입니다. 다시 선택하세요." >&2
    done
}

# Arrow-key menu (up/down + enter)
choose_from_menu_arrow() {
    local title="$1"
    shift
    local options=("$@")
    local selected=0
    local key

    while true; do
        clear >&2
        print_banner >&2
        print_section "$title" >&2
        echo "${DIM}Use Arrow Keys (Up/Down) + Enter${RESET}" >&2
        echo >&2

        local i
        for i in "${!options[@]}"; do
            if (( i == selected )); then
                echo "  ${BG_CYAN}${FG_WHITE}${BOLD}> ${options[$i]}${RESET}" >&2
            else
                echo "    ${options[$i]}" >&2
            fi
        done

        IFS= read -rsn1 key
        if [[ "$key" == $'\x1b' ]]; then
            # Read the next two bytes for arrow sequence
            IFS= read -rsn2 key || true
            case "$key" in
                '[A')
                    selected=$(( (selected - 1 + ${#options[@]}) % ${#options[@]} ))
                    ;;
                '[B')
                    selected=$(( (selected + 1) % ${#options[@]} ))
                    ;;
            esac
        elif [[ -z "$key" ]]; then
            printf '%s\n' "${options[$selected]}"
            return 0
        fi
    done
}

choose_from_menu() {
    local title="$1"
    shift
    if [[ -t 0 ]]; then
        choose_from_menu_arrow "$title" "$@"
    else
        choose_from_menu_number "$title" "$@"
    fi
}

# Defaults from shell env -> .env -> hardcoded
DEFAULT_XR_HEADLESS="${XR_HEADLESS:-}"
DEFAULT_RUN_MODE="${RUN_MODE:-}"
DEFAULT_TELEOP_TASK="${TELEOP_TASK:-}"
DEFAULT_DISPLAY="${DISPLAY:-}"

if [[ -z "$DEFAULT_XR_HEADLESS" ]]; then
    DEFAULT_XR_HEADLESS="$(load_env_value XR_HEADLESS "$ENV_FILE" || true)"
fi
if [[ -z "$DEFAULT_RUN_MODE" ]]; then
    DEFAULT_RUN_MODE="$(load_env_value RUN_MODE "$ENV_FILE" || true)"
fi
if [[ -z "$DEFAULT_TELEOP_TASK" ]]; then
    DEFAULT_TELEOP_TASK="$(load_env_value TELEOP_TASK "$ENV_FILE" || true)"
fi
if [[ -z "$DEFAULT_DISPLAY" ]]; then
    DEFAULT_DISPLAY="$(load_env_value DISPLAY "$ENV_FILE" || true)"
fi

DEFAULT_XR_HEADLESS="${DEFAULT_XR_HEADLESS:-false}"
DEFAULT_RUN_MODE="${DEFAULT_RUN_MODE:-record}"
DEFAULT_TELEOP_TASK="${DEFAULT_TELEOP_TASK:-Isaac-BarcodePress-FFW-SG2-Abs-v0}"
DEFAULT_DISPLAY="${DEFAULT_DISPLAY:-:0}"

BASE_MODE="$(choose_from_menu "기본 실행환경 선택" "GUI(X11)" "XR_HEADLESS(권장 원격)")"
if [[ "$BASE_MODE" == "GUI(X11)" ]]; then
    XR_HEADLESS_VAL="false"
else
    XR_HEADLESS_VAL="true"
fi

RUN_MODE_VAL="$(choose_from_menu "RUN_MODE 선택" "teleop" "record")"

TASK_PRESET="$(choose_from_menu "TELEOP_TASK 선택" "Isaac-BarcodePress-FFW-SG2-Abs-v0" "Isaac-PickPlace-GR1T2-Abs-v0" "직접 입력")"
if [[ "$TASK_PRESET" == "직접 입력" ]]; then
    clear
    print_banner
    print_section "TELEOP_TASK 입력"
    TELEOP_TASK_VAL="$(prompt_value "TELEOP_TASK" "$DEFAULT_TELEOP_TASK")"
else
    TELEOP_TASK_VAL="$TASK_PRESET"
fi

HOST_DISPLAY="$DEFAULT_DISPLAY"

clear >&2
print_banner >&2
print_section "실행 요약" >&2
echo "  XR_HEADLESS=${BOLD}$XR_HEADLESS_VAL${RESET}" >&2
echo "  RUN_MODE=${BOLD}$RUN_MODE_VAL${RESET}" >&2
echo "  TELEOP_TASK=${BOLD}$TELEOP_TASK_VAL${RESET}" >&2
if ! is_true "$XR_HEADLESS_VAL"; then
    echo "  DISPLAY=${BOLD}$HOST_DISPLAY${RESET}" >&2
fi
echo >&2

# Requested behavior: remove optional launch keys from .env and run with selected values
clean_optional_env_keys "$ENV_FILE"
info ".env에서 XR_HEADLESS/RUN_MODE/TELEOP_TASK 키를 제거했습니다."

if ! is_true "$XR_HEADLESS_VAL"; then
    if ! command -v xhost >/dev/null 2>&1; then
        err "xhost 명령이 없어 X11 권한을 열 수 없습니다."
        err "GUI 모드라면 xhost를 설치하거나 XR_HEADLESS=true를 사용하세요."
        exit 1
    fi

    if ! DISPLAY="$HOST_DISPLAY" xhost +SI:localuser:root >/dev/null 2>&1; then
        err "X11 권한 열기 실패 (DISPLAY=$HOST_DISPLAY)."
        err "호스트에서 X 서버가 떠 있는지 확인하세요."
        exit 1
    fi

    info "X11 권한 열림: SI:localuser:root (DISPLAY=$HOST_DISPLAY)"
else
    info "XR_HEADLESS=true: X11 권한 단계 생략"
fi

cd "$PROJECT_DIR"
exec env XR_HEADLESS="$XR_HEADLESS_VAL" \
    RUN_MODE="$RUN_MODE_VAL" \
    TELEOP_TASK="$TELEOP_TASK_VAL" \
    DISPLAY="$HOST_DISPLAY" \
    docker compose up "$@"
