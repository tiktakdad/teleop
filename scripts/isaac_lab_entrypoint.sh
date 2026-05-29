#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Isaac Lab 엔트리포인트
###############################################################################

ASSETS="/workspace/user/assets"
CUSTOM_ASSETS="${CUSTOM_ASSETS_DIR:-/workspace/user/custom_assets}"
CUSTOM_TASKS="/workspace/user/scripts/custom_tasks"
export CUSTOM_ASSETS_DIR="${CUSTOM_ASSETS}"
export PYTHONPATH="${CUSTOM_TASKS}:${PYTHONPATH:-}"
LOCO_G1_CFG="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/locomanipulation_g1_env_cfg.py"
FIXED_G1_CFG="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/fixed_base_upper_body_ik_g1_env_cfg.py"
UNITREE_PY="/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py"
G1_RETARGETING_PY="/workspace/isaaclab/source/isaaclab/isaaclab/devices/openxr/retargeters/humanoid/unitree/trihand/g1_dex_retargeting_utils.py"
FOURIER_PY="/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/robots/fourier.py"
PICKPLACE_CFG="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place/pickplace_gr1t2_env_cfg.py"
RETARGETING_PY="/workspace/isaaclab/source/isaaclab/isaaclab/devices/openxr/retargeters/humanoid/fourier/gr1_t2_dex_retargeting_utils.py"

OFFLINE_KIT_ARGS=(
    "--/persistent/isaac/asset_root/cloud=file://${ASSETS}"
    "--/persistent/isaac/asset_root/default=file://${ASSETS}"
    "--/persistent/isaac/asset_root/nvidia=file://${ASSETS}"
)

echo "[isaac-lab] 텔레오퍼레이션 시뮬레이션 시작 중..."

patch_usd_path() {
    local cfg_file="$1"
    local nucleus_pattern="$2"
    local local_path="$3"
    local label="$4"

    [[ -f "$cfg_file" ]] || return 1
    [[ -f "$local_path" ]] || return 1
    grep -qF "$local_path" "$cfg_file" && return 0
    grep -qF "$nucleus_pattern" "$cfg_file" || return 1
    sed -i "s|${nucleus_pattern}|\"${local_path}\"|g" "$cfg_file"
    echo "[isaac-lab] ✓ ${label}"
}

patch_ground_plane() {
    local cfg_file="$1"
    local local_path="$2"
    [[ -f "$cfg_file" && -f "$local_path" ]] || return 0
    grep -qF 'spawn=GroundPlaneCfg(),' "$cfg_file" || return 0
    sed -i "s|spawn=GroundPlaneCfg(),|spawn=GroundPlaneCfg(usd_path=\"${local_path}\"),|" "$cfg_file"
    echo "[isaac-lab] ✓ GroundPlane: $(basename "$cfg_file")"
}

# 🔹 로컬 Nucleus 미러
OFFLINE_ARGS=()
if [[ -f "${ASSETS}/.offline_ready" ]] || [[ -d "${ASSETS}/Isaac" ]]; then
    OFFLINE_ARGS=(--kit_args "${OFFLINE_KIT_ARGS[*]}")
    echo "[isaac-lab] ✓ 로컬 Nucleus 미러 (kit_args)"
fi

# Built-in G1 / GR1T2 path patches are unrelated to the custom FFW robot.
if [[ "${TELEOP_TASK:-Isaac-PickPlace-GR1T2-Abs-v0}" != *BarcodePress-FFW* ]]; then
    for g1_cfg in "$LOCO_G1_CFG" "$FIXED_G1_CFG"; do
        patch_usd_path "$g1_cfg" \
            'f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd"' \
            "${ASSETS}/Isaac/Props/PackingTable/packing_table.usd" "G1 PackingTable" || true
        patch_usd_path "$g1_cfg" \
            'f"{ISAACLAB_NUCLEUS_DIR}/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd"' \
            "${ASSETS}/Isaac/IsaacLab/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd" "G1 steering_wheel" || true
        patch_ground_plane "$g1_cfg" "${ASSETS}/Isaac/Environments/Grid/default_environment.usd" || true
    done

    patch_usd_path "$UNITREE_PY" \
        'f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd"' \
        "${ASSETS}/Isaac/Robots/Unitree/G1/g1.usd" "Unitree G1" || true

    patch_usd_path "$G1_RETARGETING_PY" \
        'f"{ISAACLAB_NUCLEUS_DIR}/Controllers/LocomanipulationAssets/unitree_g1_dexpilot_asset/G1_left_hand.urdf"' \
        "${ASSETS}/Isaac/IsaacLab/Controllers/LocomanipulationAssets/unitree_g1_dexpilot_asset/G1_left_hand.urdf" \
        "G1 left hand URDF" || true
    patch_usd_path "$G1_RETARGETING_PY" \
        'f"{ISAACLAB_NUCLEUS_DIR}/Controllers/LocomanipulationAssets/unitree_g1_dexpilot_asset/G1_right_hand.urdf"' \
        "${ASSETS}/Isaac/IsaacLab/Controllers/LocomanipulationAssets/unitree_g1_dexpilot_asset/G1_right_hand.urdf" \
        "G1 right hand URDF" || true

    patch_usd_path "$FOURIER_PY" \
        'f"{ISAAC_NUCLEUS_DIR}/Robots/FourierIntelligence/GR-1/GR1T2_fourier_hand_6dof/GR1T2_fourier_hand_6dof.usd"' \
        "${ASSETS}/GR1T2_fourier_hand_6dof/GR1T2_fourier_hand_6dof.usd" "GR1T2" || true
    patch_usd_path "$PICKPLACE_CFG" \
        'f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd"' \
        "${ASSETS}/Isaac/Props/PackingTable/packing_table.usd" "PackingTable" || true
    patch_usd_path "$PICKPLACE_CFG" \
        'f"{ISAACLAB_NUCLEUS_DIR}/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd"' \
        "${ASSETS}/Isaac/IsaacLab/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd" "steering_wheel" || true
    patch_ground_plane "$PICKPLACE_CFG" "${ASSETS}/Isaac/Environments/Grid/default_environment.usd" || true
fi

TELEOP_TASK="${TELEOP_TASK:-Isaac-PickPlace-GR1T2-Abs-v0}"
RUN_MODE="${RUN_MODE:-teleop}"

# Previous configurations may still pass the unversioned ID, but Gym registers -v0.
if [[ "$TELEOP_TASK" == "Isaac-BarcodePress-FFW-SG2-Abs" ]]; then
    TELEOP_TASK="${TELEOP_TASK}-v0"
    echo "[isaac-lab] task id normalized: ${TELEOP_TASK}"
fi

is_barcode_ffw_task() {
    [[ "$TELEOP_TASK" == *BarcodePress-FFW* ]]
}

# 🔹 custom asset 태스크 등록 (gym id)
if is_barcode_ffw_task; then
    [[ -d "${CUSTOM_TASKS}/teleop_barcode_press" ]] || {
        echo "[isaac-lab] ! custom task package 없음: ${CUSTOM_TASKS}/teleop_barcode_press"
        exit 1
    }
    if [[ -f "${CUSTOM_ASSETS}/scene/reference.usd" && -f "${CUSTOM_ASSETS}/robot/ai_worker/FFW_SG2.usd" ]]; then
        echo "[isaac-lab] ✓ custom task: Isaac-BarcodePress-FFW-SG2-Abs-v0 (scene/reference.usd)"
    else
        echo "[isaac-lab] ! custom_assets 미비 — scene/reference.usd / FFW_SG2.usd 확인"
        exit 1
    fi
fi

# 🔹 G1 텔레옵 수집: 로봇 POV 카메라 → HDF5 obs/robot_pov_cam
is_g1_task() {
    [[ "$TELEOP_TASK" == *G1* ]] || [[ "$TELEOP_TASK" == *Locomanipulation* ]]
}

if [[ "$RUN_MODE" == "record" ]] && is_g1_task; then
    # 컨테이너 기본 PATH에 python3 없을 수 있음 → Isaac Sim 번들 Python 사용
    PATCH_PY="/workspace/isaaclab/_isaac_sim/python.sh"
    if [[ ! -x "$PATCH_PY" ]]; then
        PATCH_PY="$(command -v python3 || true)"
    fi
    if [[ -n "$PATCH_PY" ]] && "$PATCH_PY" /workspace/user/scripts/patch_g1_robot_camera.py; then
        echo "[isaac-lab] ✓ G1 robot_pov_cam 패치 (d435 ${ROBOT_CAM_WIDTH:-256}x${ROBOT_CAM_HEIGHT:-160}, yaw=${ROBOT_CAM_YAW_DEG:--25}° focal=${ROBOT_CAM_FOCAL_LENGTH:-5.5})"
    else
        echo "[isaac-lab] ! G1 카메라 패치 실패 — 상태만 기록될 수 있음"
        echo "[isaac-lab]   로그: docker compose logs isaac-lab | grep patch_g1"
    fi
    if [[ -n "$PATCH_PY" ]] && "$PATCH_PY" /workspace/user/scripts/patch_g1_hand_markers_for_record.py; then
        echo "[isaac-lab] ✓ G1 손 트래킹 유지 + record 영상용 마커 투명화 (enable_visualization=True)"
    else
        echo "[isaac-lab] ! 손 마커 패치 실패 — 로그: docker compose logs isaac-lab | grep patch_g1_hand"
    fi
fi

# 🔹 XR 세션 자동 시작 (원격 PC에서 Start AR 클릭 생략)
# Isaac Lab: --headless + --xr → isaaclab.python.xr.openxr.headless.kit, AR 프로필 자동 활성화
XR_LAUNCH_ARGS=()
case "${XR_HEADLESS:-false}" in
    1|true|TRUE|yes|YES)
        XR_LAUNCH_ARGS=(--headless)
        echo "[isaac-lab] ✓ XR headless — Start AR 자동 시작 (Quest는 WebXR Connect만)"
        ;;
    *)
        echo "[isaac-lab] GUI 모드 — 로딩 후 AR 패널 → Start AR 클릭 (또는 .env에 XR_HEADLESS=true)"
        ;;
esac

cd /workspace/isaaclab

COMMON_ARGS=(
    "${OFFLINE_ARGS[@]}"
    "${XR_LAUNCH_ARGS[@]}"
    --task "$TELEOP_TASK"
    --teleop_device "${TELEOP_DEVICE:-handtracking}"
    --enable_pinocchio
)

if [[ "$RUN_MODE" == "record" ]]; then
    echo "[isaac-lab] 🔴 데이터 수집 (record_demos.py)"
    DATASET_DIR="/workspace/user/datasets"
    mkdir -p "$DATASET_DIR"

    BASE_DATASET_FILE="${DATASET_FILE:-$DATASET_DIR/dataset.hdf5}"
    DATASET_TIMESTAMP="$(date +%y%m%d_%H%M)"
    DATASET_BASENAME="$(basename "$BASE_DATASET_FILE")"
    DATASET_NAME="${DATASET_BASENAME%.*}"
    DATASET_EXT="${DATASET_BASENAME##*.}"
    if [[ "$DATASET_NAME" == "$DATASET_EXT" ]]; then
        DATASET_FILE="${BASE_DATASET_FILE}_${DATASET_TIMESTAMP}"
    else
        DATASET_FILE="$(dirname "$BASE_DATASET_FILE")/${DATASET_NAME}_${DATASET_TIMESTAMP}.${DATASET_EXT}"
    fi
    echo "[isaac-lab]   저장 경로: $DATASET_FILE"

    # 🔹 바코드 FFW 태스크: record_demos.py 대신 텔레옵 스크립트(--record)로 수집해
    #    텔레옵과 동일하게 동작(카메라 프리뷰/네비/3초 성공) + 시각화 인디케이터는 녹화에서 제거.
    if is_barcode_ffw_task; then
        echo "[isaac-lab] 🔴 데이터 수집 — FFW 바코드 (teleop_barcode_ffw.py --record, hand cam ON)"
        BARCODE_RECORD_ARGS=("${COMMON_ARGS[@]}"
            --enable_cameras
            --record
            --dataset_file "$DATASET_FILE"
            --num_demos "${NUM_DEMOS:-10}"
        )
        if [[ -n "${NUM_SUCCESS_STEPS:-}" ]]; then
            BARCODE_RECORD_ARGS+=(--num_success_steps "$NUM_SUCCESS_STEPS")
        fi
        if [[ "${TELEOP_DEVICE:-handtracking}" == *handtracking* ]]; then
            BARCODE_RECORD_ARGS+=(--xr)
        fi
        exec ./isaaclab.sh -p /workspace/user/scripts/teleop_barcode_ffw.py "${BARCODE_RECORD_ARGS[@]}"
    fi

    if [[ -n "${RECORD_FPS:-}" ]]; then
        RECORD_DEMOS_PY="/workspace/isaaclab/scripts/tools/record_demos.py"
        if grep -qF "# [teleop] RECORD_FPS env-step patch" "$RECORD_DEMOS_PY"; then
            python3 - "$RECORD_DEMOS_PY" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old_marker = "    # [teleop] RECORD_FPS env-step patch"
next_block = "    # extract success checking function"
start = text.find(old_marker)
if start != -1:
    end = text.find(next_block, start)
    if end == -1:
        raise SystemExit("old RECORD_FPS env-step patch end marker not found")
    text = text[:start] + text[end:]
    path.write_text(text)
PY
            echo "[isaac-lab] ✓ old RECORD_FPS env-step patch removed"
        fi
        if grep -qF "# [teleop] RECORD_FPS recorder sampling patch" "$RECORD_DEMOS_PY"; then
            :
        elif grep -qF "    # Run simulation loop" "$RECORD_DEMOS_PY"; then
            python3 - "$RECORD_DEMOS_PY" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
needle = "    # Run simulation loop"
patch = '''    # [teleop] RECORD_FPS recorder sampling patch
    record_fps = os.environ.get("RECORD_FPS", "").strip()
    if record_fps:
        try:
            target_fps = float(record_fps)
            if target_fps <= 0:
                raise ValueError
        except ValueError:
            omni.log.error(f"Invalid RECORD_FPS={record_fps!r}. Use a positive number, e.g. 15.")
            exit(1)

        base_step_dt = float(env.step_dt)
        if target_fps > (1.0 / base_step_dt) + 1e-6:
            omni.log.error(
                f"RECORD_FPS={target_fps:g} exceeds env step rate {1.0 / base_step_dt:g} Hz. "
                "Use a lower value or increase the environment rate."
            )
            exit(1)

        orig_record_pre_step = env.recorder_manager.record_pre_step
        orig_record_post_step = env.recorder_manager.record_post_step
        sample_state = {"step_index": 0, "next_t": 0.0, "record_this_step": False}
        sample_dt = 1.0 / target_fps

        def sampled_record_pre_step():
            step_t = sample_state["step_index"] * base_step_dt
            record_this_step = step_t + 0.5 * base_step_dt >= sample_state["next_t"] - 1e-9
            sample_state["record_this_step"] = record_this_step
            if record_this_step:
                while sample_state["next_t"] <= step_t + 0.5 * base_step_dt + 1e-9:
                    sample_state["next_t"] += sample_dt
                return orig_record_pre_step()

        def sampled_record_post_step():
            try:
                if sample_state["record_this_step"]:
                    return orig_record_post_step()
            finally:
                sample_state["step_index"] += 1
                sample_state["record_this_step"] = False

        env.recorder_manager.record_pre_step = sampled_record_pre_step
        env.recorder_manager.record_post_step = sampled_record_post_step
        omni.log.info(
            f"RECORD_FPS={target_fps:g}: recording samples at {target_fps:g} Hz while env control stays "
            f"at {1.0 / base_step_dt:g} Hz (step_dt={base_step_dt:.6f}s)."
        )

'''
path.write_text(text.replace(needle, patch + needle, 1))
PY
            echo "[isaac-lab] ✓ RECORD_FPS recorder sampling enabled: ${RECORD_FPS} Hz"
        else
            echo "[isaac-lab] ! RECORD_FPS patch 위치를 찾지 못함" >&2
            exit 1
        fi
    fi

    RECORD_ARGS=("${COMMON_ARGS[@]}"
        --dataset_file "$DATASET_FILE"
        --num_demos "${NUM_DEMOS:-10}"
        --xr
    )
    if is_g1_task; then
        RECORD_ARGS+=(--enable_cameras)
        echo "[isaac-lab]   G1 로봇 POV: --enable_cameras (obs/robot_pov_cam → HDF5)"
    fi
    if is_barcode_ffw_task; then
        RECORD_ARGS+=(--enable_cameras)
        echo "[isaac-lab]   FFW 양손 cam: --enable_cameras (obs/left_hand_cam, obs/right_hand_cam → HDF5)"

        # record_demos imports built-in tasks only; register the mounted custom task after app startup.
        RECORD_DEMOS_PY="/workspace/isaaclab/scripts/tools/record_demos.py"
        if grep -qF "import teleop_barcode_press" "$RECORD_DEMOS_PY"; then
            :
        elif grep -qF "import isaaclab_tasks  # noqa: F401" "$RECORD_DEMOS_PY"; then
            sed -i '/^import isaaclab_tasks  # noqa: F401$/a import teleop_barcode_press  # noqa: F401 - custom Gym task registration' "$RECORD_DEMOS_PY"
            echo "[isaac-lab] ✓ record custom task registration: teleop_barcode_press"
        else
            echo "[isaac-lab] ! record_demos.py custom task 등록 위치를 찾지 못함" >&2
            exit 1
        fi

        # The factory is already loaded at this point; add the custom retargeter without early imports.
        if grep -qF "RETARGETER_MAP[FfwSg2RetargeterCfg] = FfwSg2Retargeter" "$RECORD_DEMOS_PY"; then
            :
        elif grep -qF "import teleop_barcode_press" "$RECORD_DEMOS_PY"; then
            sed -i '/^import teleop_barcode_press /a from teleop_barcode_press.retargeters import FfwSg2Retargeter, FfwSg2RetargeterCfg\nimport isaaclab.devices.teleop_device_factory as teleop_device_factory\nteleop_device_factory.RETARGETER_MAP[FfwSg2RetargeterCfg] = FfwSg2Retargeter' "$RECORD_DEMOS_PY"
            echo "[isaac-lab] ✓ record retargeter registration: FfwSg2Retargeter"
        else
            echo "[isaac-lab] ! record_demos.py custom retargeter 등록 위치를 찾지 못함" >&2
            exit 1
        fi
    fi

    exec ./isaaclab.sh -p scripts/tools/record_demos.py "${RECORD_ARGS[@]}"
elif [[ "$RUN_MODE" == "teleop" ]]; then
    TELEOP_ARGS=("${COMMON_ARGS[@]}")
    if [[ "${TELEOP_DEVICE:-handtracking}" == *handtracking* ]]; then
        TELEOP_ARGS+=(--xr)
    fi
    if is_barcode_ffw_task; then
        TELEOP_ARGS+=(--enable_cameras)
        echo "[isaac-lab] 🟢 텔레옵 — FFW_SG2 바코드 프레스 (teleop_barcode_ffw.py, hand cam ON)"
        exec ./isaaclab.sh -p /workspace/user/scripts/teleop_barcode_ffw.py "${TELEOP_ARGS[@]}"
    else
        echo "[isaac-lab] 🟢 텔레옵 (teleop_se3_agent.py)"
        exec ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py "${TELEOP_ARGS[@]}"
    fi
else
    echo "[isaac-lab] 🟡 커스텀: $RUN_MODE"
    exec ./isaaclab.sh -p "$RUN_MODE" "${COMMON_ARGS[@]}"
fi
