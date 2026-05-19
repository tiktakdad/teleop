#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Isaac Lab 엔트리포인트 — 로컬 캐시 USD가 있으면 Nucleus URL 대신 사용
###############################################################################

ASSETS="/workspace/user/assets"
FOURIER_PY="/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/robots/fourier.py"
PICKPLACE_CFG="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place/pickplace_gr1t2_env_cfg.py"
RETARGETING_PY="/workspace/isaaclab/source/isaaclab/isaaclab/devices/openxr/retargeters/humanoid/fourier/gr1_t2_dex_retargeting_utils.py"

echo "[isaac-lab] 텔레오퍼레이션 시뮬레이션 시작 중..."

# 🔹 nucleus 패턴 → 로컬 경로 (파일 존재 시에만 치환)
patch_usd_path() {
    local cfg_file="$1"
    local nucleus_pattern="$2"
    local local_path="$3"
    local label="$4"

    [[ -f "$cfg_file" ]] || return 1
    [[ -f "$local_path" ]] || return 1

    if grep -qF "$local_path" "$cfg_file"; then
        echo "[isaac-lab] ✓ ${label}: 이미 로컬 경로"
        return 0
    fi

    if grep -qF "$nucleus_pattern" "$cfg_file"; then
        sed -i "s|${nucleus_pattern}|\"${local_path}\"|g" "$cfg_file"
        echo "[isaac-lab] ✓ ${label}: ${local_path}"
        return 0
    fi

    echo "[isaac-lab] ! ${label}: 패턴 없음"
    return 1
}

# 🔹 GroundPlaneCfg() 기본 Nucleus URL → 로컬 default_environment.usd
patch_ground_plane() {
    local cfg_file="$1"
    local local_path="$2"

    [[ -f "$cfg_file" ]] || return 1
    [[ -f "$local_path" ]] || return 1

    if grep -qF "usd_path=\"${local_path}\"" "$cfg_file"; then
        echo "[isaac-lab] ✓ GroundPlane: 이미 로컬 경로"
        return 0
    fi

    if grep -qF 'spawn=GroundPlaneCfg(),' "$cfg_file"; then
        sed -i "s|spawn=GroundPlaneCfg(),|spawn=GroundPlaneCfg(usd_path=\"${local_path}\"),|" "$cfg_file"
        echo "[isaac-lab] ✓ GroundPlane: ${local_path}"
        return 0
    fi

    echo "[isaac-lab] ! GroundPlane: spawn=GroundPlaneCfg() 패턴 없음"
    return 1
}

# GR1T2 로봇 (fourier.py)
patch_usd_path "$FOURIER_PY" \
    'f"{ISAAC_NUCLEUS_DIR}/Robots/FourierIntelligence/GR-1/GR1T2_fourier_hand_6dof/GR1T2_fourier_hand_6dof.usd"' \
    "${ASSETS}/GR1T2_fourier_hand_6dof/GR1T2_fourier_hand_6dof.usd" \
    "GR1T2 로봇" || true

# PickPlace 환경 (작업대·조향휠)
patch_usd_path "$PICKPLACE_CFG" \
    'f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd"' \
    "${ASSETS}/Isaac/Props/PackingTable/packing_table.usd" \
    "PackingTable" || true

patch_usd_path "$PICKPLACE_CFG" \
    'f"{ISAACLAB_NUCLEUS_DIR}/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd"' \
    "${ASSETS}/Isaac/IsaacLab/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd" \
    "steering_wheel" || true

# 바닥(Grid) — spawn 시 Plane prim 필요 (없으면 GetPrimAtPath(None) 오류)
patch_ground_plane "$PICKPLACE_CFG" \
    "${ASSETS}/Isaac/Environments/Grid/default_environment.usd" || true

# GR1T2 핸드트래킹 리타게팅 URDF (teleop device 생성 시 필요)
patch_usd_path "$RETARGETING_PY" \
    'f"{ISAACLAB_NUCLEUS_DIR}/Mimic/GR1T2_assets/GR1_T2_left_hand.urdf"' \
    "${ASSETS}/Isaac/IsaacLab/Mimic/GR1T2_assets/GR1_T2_left_hand.urdf" \
    "GR1T2 left hand URDF" || true

patch_usd_path "$RETARGETING_PY" \
    'f"{ISAACLAB_NUCLEUS_DIR}/Mimic/GR1T2_assets/GR1_T2_right_hand.urdf"' \
    "${ASSETS}/Isaac/IsaacLab/Mimic/GR1T2_assets/GR1_T2_right_hand.urdf" \
    "GR1T2 right hand URDF" || true

# 누락 에셋 안내
missing=false
for f in \
    "${ASSETS}/GR1T2_fourier_hand_6dof/GR1T2_fourier_hand_6dof.usd" \
    "${ASSETS}/Isaac/Props/PackingTable/packing_table.usd" \
    "${ASSETS}/Isaac/IsaacLab/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd" \
    "${ASSETS}/Isaac/Environments/Grid/default_environment.usd" \
    "${ASSETS}/Isaac/IsaacLab/Mimic/GR1T2_assets/GR1_T2_left_hand.urdf" \
    "${ASSETS}/Isaac/IsaacLab/Mimic/GR1T2_assets/GR1_T2_right_hand.urdf"
do
    if [[ ! -f "$f" ]]; then
        echo "[isaac-lab] ! 없음: $f"
        missing=true
    fi
done
if [[ "$missing" == true ]]; then
    echo "[isaac-lab]   호스트에서: ./scripts/download_assets.sh"
fi

echo "[isaac-lab] 시뮬레이션 로딩 후 UI에서 AR 패널 → Start AR 클릭 필요"

cd /workspace/isaaclab

RUN_MODE="${RUN_MODE:-teleop}"
if [ "$RUN_MODE" = "record" ]; then
    echo "[isaac-lab] 🔴 데이터 수집 모드 (record_demos.py) 실행"
    DATASET_DIR="/workspace/user/datasets"
    mkdir -p "$DATASET_DIR"
    
    exec ./isaaclab.sh -p scripts/tools/record_demos.py \
        --task "${TELEOP_TASK:-Isaac-PickPlace-GR1T2-Abs-v0}" \
        --teleop_device "${TELEOP_DEVICE:-handtracking}" \
        --dataset_file "${DATASET_FILE:-$DATASET_DIR/dataset.hdf5}" \
        --num_demos "${NUM_DEMOS:-10}" \
        --enable_pinocchio \
        --xr
elif [ "$RUN_MODE" = "teleop" ]; then
    echo "[isaac-lab] 🟢 단순 텔레옵 모드 (teleop_se3_agent.py) 실행"
    exec ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
        --task "${TELEOP_TASK:-Isaac-PickPlace-GR1T2-Abs-v0}" \
        --teleop_device "${TELEOP_DEVICE:-handtracking}" \
        --enable_pinocchio
else
    echo "[isaac-lab] 🟡 커스텀 스크립트 모드: $RUN_MODE"
    exec ./isaaclab.sh -p "$RUN_MODE" \
        --task "${TELEOP_TASK:-Isaac-PickPlace-GR1T2-Abs-v0}" \
        --teleop_device "${TELEOP_DEVICE:-handtracking}" \
        --enable_pinocchio
fi
