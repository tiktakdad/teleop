# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
키보드로 관절(액션 차원)을 직접 제어하여 관절 테깅 확인용 데모를 녹화하는 스크립트.

Isaac-PickPlace-Locomanipulation-G1-Abs-v0 환경의 액션 공간 (총 32차원):
  [0:7]   Left Wrist 포즈  : pos(x,y,z) + quat(w,x,y,z)
  [7:14]  Right Wrist 포즈 : pos(x,y,z) + quat(w,x,y,z)
  [14:28] Hand 관절 14개   : L(index0, middle0, thumb0, index1, middle1, thumb1, thumb2)
                             R(index0, middle0, thumb0, index1, middle1, thumb1, thumb2)
  [28:32] Lower Body 명령  : [vx, vy, wz, hip_height]

키 바인딩 (QWERTY 순서대로 액션 dim 0→25):
  키 단독     → 해당 dim +delta (증가)
  Shift+키    → 해당 dim -delta (감소)
  Backspace   → 모든 액션 값을 0으로 리셋
  R           → 녹화 인스턴스 리셋

  Q→dim0  W→dim1  E→dim2  R→dim3  T→dim4  Y→dim5  U→dim6  I→dim7  O→dim8  P→dim9
  A→dim10 S→dim11 D→dim12 F→dim13 G→dim14 H→dim15 J→dim16 K→dim17 L→dim18
  Z→dim19 X→dim20 C→dim21 V→dim22 B→dim23 N→dim24 M→dim25

required arguments:
    --task                    Name of the task.

optional arguments:
    -h, --help                Show this help message and exit
    --dataset_file            File path to export recorded demos. (default: "./datasets/dataset.hdf5")
    --step_hz                 Environment stepping rate in Hz. (default: 30)
    --num_demos               Number of demonstrations to record. (default: 0)
    --num_success_steps       Number of continuous steps with task success. (default: 10)
    --joint_delta             Delta value per key press. (default: 0.02)
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib

from isaaclab.app import AppLauncher

import os
import h5py
import numpy as np

# ─────────────────────────────────────────────────────
# 관절 테깅 유틸리티
# ─────────────────────────────────────────────────────
from isaaclab_hdf5_tagger import IsaacLabHdf5Tagger



# ─────────────────────────────────────────────────────
# CLI 인자 파싱
# ─────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="키보드 관절 제어로 관절 테깅 확인용 데모 녹화.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument("--num_demos", type=int, default=0, help="Number of demonstrations to record. 0=infinite.")
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Number of continuous steps with task success for concluding a demo as successful.",
)
parser.add_argument(
    "--joint_delta",
    type=float,
    default=0.02,
    help="키 1회 입력 시 액션 차원에 더해지는 delta 값. (default: 0.02)",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio (required for IK controller).",
)
parser.add_argument(
    "--auto_align",
    action="store_true",
    default=False,
    help=(
        "action과 state의 표현 방식이 불일치할 경우 자동으로 관절 공간으로 변환합니다. "
        "IK/EEF 포즈 action을 obs/robot_joint_pos[t+1]으로 대체합니다."
    ),
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.task is None:
    parser.error("--task is required")

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging
import sys
import time

import gymnasium as gym
import torch
import omni

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import isaaclab_mimic.envs  # noqa: F401
from isaaclab_mimic.ui.instruction_display import InstructionDisplay, show_subtask_instructions

# headless 모드에서는 omni.ui / EmptyWindow 가 존재하지 않음 — 조건부 임포트
_USE_GUI = not args_cli.headless
if _USE_GUI:
    try:
        import omni.ui as ui
        from isaaclab.envs.ui import EmptyWindow
    except ModuleNotFoundError:
        print("[WARN] omni.ui 를 불러올 수 없습니다. GUI 없이 실행합니다.")
        _USE_GUI = False

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# 관절(액션 차원) 키보드 컨트롤러  ─  터미널 stdin 방식
# ─────────────────────────────────────────────────────
class JointKeyboardController:
    """터미널 stdin을 직접 읽어 액션 차원을 제어하는 컨트롤러.

    carb.input 은 OS 윈도우(GLFW)가 있어야 동작하므로 headless 환경에서
    사용 불가. 이 구현은 별도 스레드에서 tty raw 모드로 stdin을 읽어
    headless/GUI 모두에서 키보드 입력을 처리한다.

    키 바인딩:
      소문자 q,w,e,...,m → 해당 dim + delta  (증가)
      대문자 Q,W,E,...,M → 해당 dim - delta  (감소, Shift+키)
      Enter(\\r/\\n)      → ENTER 콜백 (수동 저장)
      r / R              → R 콜백 (녹화 리셋)
      Backspace(\\x7f)   → 액션값 초기 포즈로 복원
      Ctrl+C(\\x03)      → 종료
    """

    # QWERTY 순서 소문자 → dim 인덱스
    KEY_TO_DIM: dict[str, int] = {
        'q': 0,  'w': 1,  'e': 2,  'r': 3,  't': 4,
        'y': 5,  'u': 6,  'i': 7,  'o': 8,  'p': 9,
        'a': 10, 's': 11, 'd': 12, 'f': 13, 'g': 14,
        'h': 15, 'j': 16, 'k': 17, 'l': 18,
        'z': 19, 'x': 20, 'c': 21, 'v': 22, 'b': 23,
        'n': 24, 'm': 25,
    }

    DIM_LABELS: dict[int, str] = {
        0:  "L_Wrist pos_x",   1:  "L_Wrist pos_y",   2:  "L_Wrist pos_z",
        3:  "L_Wrist quat_w",  4:  "L_Wrist quat_x",  5:  "L_Wrist quat_y",
        6:  "L_Wrist quat_z",
        7:  "R_Wrist pos_x",   8:  "R_Wrist pos_y",   9:  "R_Wrist pos_z",
        10: "R_Wrist quat_w",  11: "R_Wrist quat_x",  12: "R_Wrist quat_y",
        13: "R_Wrist quat_z",
        14: "L_Hand index0",   15: "L_Hand middle0",  16: "L_Hand thumb0",
        17: "R_Hand index0",   18: "R_Hand middle0",  19: "R_Hand thumb0",
        20: "L_Hand index1",   21: "L_Hand middle1",  22: "L_Hand thumb1",
        23: "R_Hand index1",   24: "R_Hand middle1",  25: "R_Hand thumb1",
        26: "L_Hand thumb2",   27: "R_Hand thumb2",
        28: "LowerBody vx",    29: "LowerBody vy",    30: "LowerBody wz",
        31: "LowerBody hip_h",
    }

    def __init__(self, action_dim: int, delta: float = 0.02, device: str = "cpu",
                 initial_actions: np.ndarray | None = None):
        import select
        import threading
        import atexit

        self._action_dim = action_dim
        self._delta = delta
        self._sim_device = device
        self._additional_callbacks: dict = {}
        self._lock = threading.Lock()
        self._select = select
        self._raw_mode = False
        self._fd = None
        self._old_term = None

        # 초기 액션 설정
        if initial_actions is not None:
            self._actions = initial_actions.astype(np.float32).copy()
        else:
            self._actions = np.zeros(action_dim, dtype=np.float32)
            if action_dim > 3:
                self._actions[3] = 1.0   # left wrist quat_w
            if action_dim > 10:
                self._actions[10] = 1.0  # right wrist quat_w
        self._initial_actions = self._actions.copy()

        # 터미널 raw 모드 시도 (단일 문자 즉시 읽기)
        # SSH / pipe 환경에서는 isatty()=False 이므로 setraw() 실패 → line 모드로 폴백
        if sys.stdin.isatty():
            try:
                import termios
                import tty
                self._fd = sys.stdin.fileno()
                self._old_term = termios.tcgetattr(self._fd)
                tty.setraw(self._fd)
                self._raw_mode = True
                atexit.register(self._restore_terminal)
                print("[JointKeyboard] ✅ Raw 터미널 모드 활성화 — 키를 누르면 즉시 반응합니다.")
            except Exception as e:
                print(f"[JointKeyboard] ⚠ Raw 모드 설정 실패 ({e}) → Line 모드로 폴백합니다.")
                self._raw_mode = False
        else:
            print("[JointKeyboard] ℹ stdin이 TTY가 아닙니다 (pipe/SSH?) → Line 모드로 실행합니다.")
            print("                각 명령 입력 후 Enter를 누르세요.")

        # 키 읽기 스레드 시작
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

        self._print_keybindings()

    def _restore_terminal(self):
        if self._raw_mode and self._fd is not None and self._old_term is not None:
            try:
                import termios
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_term)
            except Exception:
                pass

    def __del__(self):
        self._running = False
        self._restore_terminal()

    def _read_loop(self):
        """백그라운드 스레드: stdin에서 문자를 읽어 액션 버퍼 갱신."""
        try:
            if self._raw_mode:
                # ── Raw 모드: 키 1개씩 즉시 읽기 ──
                while self._running:
                    if self._select.select([sys.stdin], [], [], 0.05)[0]:
                        ch = sys.stdin.read(1)
                        self._handle_char(ch)
            else:
                # ── Line 모드 폴백: 줄 단위 읽기 (Enter 필요) ──
                print("\n[Line 모드] 명령 형식:  <키문자>  또는  <키문자><Enter>")
                print("  예) q  → dim[0] +delta,   Q  → dim[0] -delta")
                print("  예) enter 단독 → 저장,   r → 리셋\n")
                while self._running:
                    try:
                        line = input()
                        for ch in line.strip() or '\n':
                            self._handle_char(ch)
                        if not line.strip():   # 빈 줄 = Enter
                            self._handle_char('\n')
                    except EOFError:
                        break
        except Exception:
            pass
        finally:
            self._restore_terminal()

    def _handle_char(self, ch: str):
        """단일 문자 입력을 처리한다."""
        if ch == '\x03':          # Ctrl+C
            self._running = False
            raise KeyboardInterrupt

        if ch in ('\r', '\n'):    # Enter → 수동 저장
            if 'ENTER' in self._additional_callbacks:
                self._additional_callbacks['ENTER']()
            return

        if ch in ('\x7f', '\x08'):  # Backspace → 초기 포즈 복원
            with self._lock:
                self._actions = self._initial_actions.copy()
            print("\r[JointKeyboard] ↩  액션값을 초기 포즈로 복원했습니다.              ")
            return

        lower = ch.lower()
        is_upper = ch.isupper()

        if lower == 'r':          # r/R → 녹화 리셋 콜백
            if 'R' in self._additional_callbacks:
                self._additional_callbacks['R']()
            return

        if lower in self.KEY_TO_DIM:
            dim = self.KEY_TO_DIM[lower]
            if dim < self._action_dim:
                sign = -1.0 if is_upper else +1.0
                with self._lock:
                    self._actions[dim] += sign * self._delta
                    val = self._actions[dim]
                label = self.DIM_LABELS.get(dim, f"action[{dim}]")
                arrow = "▼" if is_upper else "▲"
                print(f"\r[JointKeyboard] {arrow} {ch}  dim[{dim:2d}] {label:20s} = {val:+.4f}    ")

    def _print_keybindings(self):
        print("\n" + "=" * 70)
        print("🎹  Joint Keyboard Controller  (터미널 stdin 방식)")
        print("=" * 70)
        print(f"  action_dim = {self._action_dim},  delta per key = {self._delta}")
        print()
        print("  소문자 키    → 해당 액션 차원 + delta  (증가)")
        print("  대문자 키    → 해당 액션 차원 - delta  (감소, Shift+키)")
        print("  Enter        → 현재 에피소드 저장 후 리셋")
        print("  r / R        → 저장 없이 에피소드 리셋")
        print("  Backspace    → 액션값 초기 포즈로 복원")
        print("  Ctrl+C       → 종료")
        print()
        header = f"  {'키(+)':6s}  {'키(-)':6s}  {'dim':4s}  {'의미'}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for lower, dim in self.KEY_TO_DIM.items():
            if dim < self._action_dim:
                upper = lower.upper()
                label = self.DIM_LABELS.get(dim, f"action[{dim}]")
                print(f"  {lower:6s}  {upper:6s}  [{dim:2d}]  {label}")
        print("=" * 70 + "\n")

    def reset(self):
        with self._lock:
            self._actions = self._initial_actions.copy()

    def add_callback(self, key: str, func):
        self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor:
        with self._lock:
            return torch.tensor(self._actions.copy(), dtype=torch.float32, device=self._sim_device)


# ─────────────────────────────────────────────────────
# IK 초기 액션 헬퍼
# ─────────────────────────────────────────────────────
def _get_initial_ik_action(env, action_dim: int) -> np.ndarray:
    """리셋 직후 로봇의 실제 손목 포즈를 읽어 IK 초기 액션을 계산합니다.

    IK 액션은 절대 포즈(위치+쿼터니언)를 pelvis(골반) 기준 좌표계로 받습니다.
    영벡터(0,0,0,0,0,0,0)를 그대로 주면 IK 솔버가 로봇 몸통 내부를 목표로
    삼아 물리 연산이 폭주(BroadPhaseUpdateData)하므로, 반드시 현재 포즈로
    초기화해야 합니다.
    """
    from isaaclab.utils.math import subtract_frame_transforms

    robot = env.scene["robot"]

    try:
        pelvis_ids, _ = robot.find_bodies(".*pelvis")
        lw_ids, _     = robot.find_bodies(".*left_wrist_yaw_link")
        rw_ids, _     = robot.find_bodies(".*right_wrist_yaw_link")

        # world frame poses — shape (num_envs, 3) or (3,)
        p_p  = robot.data.body_pos_w[0, pelvis_ids[0]]
        q_p  = robot.data.body_quat_w[0, pelvis_ids[0]]   # wxyz
        p_lw = robot.data.body_pos_w[0, lw_ids[0]]
        q_lw = robot.data.body_quat_w[0, lw_ids[0]]
        p_rw = robot.data.body_pos_w[0, rw_ids[0]]
        q_rw = robot.data.body_quat_w[0, rw_ids[0]]

        # pelvis 기준 좌표계로 변환 (배치 차원 추가)
        p_lw_loc, q_lw_loc = subtract_frame_transforms(
            p_p.unsqueeze(0), q_p.unsqueeze(0),
            p_lw.unsqueeze(0), q_lw.unsqueeze(0),
        )
        p_rw_loc, q_rw_loc = subtract_frame_transforms(
            p_p.unsqueeze(0), q_p.unsqueeze(0),
            p_rw.unsqueeze(0), q_rw.unsqueeze(0),
        )

        action = np.zeros(action_dim, dtype=np.float32)
        action[0:3]   = p_lw_loc[0].cpu().numpy()
        action[3:7]   = q_lw_loc[0].cpu().numpy()   # wxyz
        action[7:10]  = p_rw_loc[0].cpu().numpy()
        action[10:14] = q_rw_loc[0].cpu().numpy()   # wxyz

        print("\n[JointKeyboard] ✅ 초기 IK 액션을 로봇 현재 포즈에서 읽었습니다.")
        print(f"  Left  Wrist (pelvis 기준): pos={action[0:3].round(3)}, quat={action[3:7].round(3)}")
        print(f"  Right Wrist (pelvis 기준): pos={action[7:10].round(3)}, quat={action[10:14].round(3)}")
        return action

    except Exception as e:
        print(f"\n[WARN] 초기 IK 액션 계산 실패: {e}")
        print("[WARN] 폴백: 쿼터니언 w=1 로 초기화 (위치는 원점 — 약간의 경고 발생 가능)")
        action = np.zeros(action_dim, dtype=np.float32)
        if action_dim > 3:
            action[3] = 1.0   # left wrist quat_w
        if action_dim > 10:
            action[10] = 1.0  # right wrist quat_w
        return action


# ─────────────────────────────────────────────────────
# 공통 유틸리티
# ─────────────────────────────────────────────────────
class RateLimiter:
    def __init__(self, hz: int):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env):
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def setup_output_directories() -> tuple[str, str]:
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir, output_file_name


def create_environment_config(output_dir: str, output_file_name: str):
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = args_cli.task.split(":")[-1]
    except Exception as e:
        logger.error(f"Failed to parse environment configuration: {e}")
        exit(1)

    success_term = None
    if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        logger.warning("No success termination term found. Cannot mark demos as successful.")

    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    # 관절 테깅 확인 목적이므로 EXPORT_ALL 사용
    # → Enter 키로 수동 저장 또는 에피소드 종료 시 자동 저장
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL

    return env_cfg, success_term


def process_success_condition(env, success_term, success_step_count: int):
    if success_term is None:
        return success_step_count, False

    if bool(success_term.func(env, **success_term.params)[0]):
        success_step_count += 1
        if success_step_count >= args_cli.num_success_steps:
            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            env.recorder_manager.set_success_to_episodes(
                [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
            )
            env.recorder_manager.export_episodes([0])
            print("Success condition met! Recording completed.")
            return success_step_count, True
    else:
        success_step_count = 0

    return success_step_count, False


def handle_reset(env, success_step_count: int, instruction_display, label_text: str) -> int:
    print("Resetting environment...")
    env.sim.reset()
    env.recorder_manager.reset()
    env.reset()
    success_step_count = 0
    if _USE_GUI and instruction_display is not None:
        instruction_display.show_demo(label_text)
    return success_step_count


# ─────────────────────────────────────────────────────
# 메인 시뮬레이션 루프
# ─────────────────────────────────────────────────────
def run_simulation_loop(env, success_term, rate_limiter) -> int:
    current_recorded_demo_count = 0
    success_step_count = 0
    should_reset_recording_instance = False
    should_manual_save = False
    running_recording_instance = True  # XR 없이는 즉시 시작

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Recording instance reset requested")

    def manual_save_and_reset():
        """Enter 키: 현재 에피소드를 성공으로 표시하고 즉시 저장 후 리셋."""
        nonlocal should_manual_save
        should_manual_save = True
        print("\n[JointKeyboard] ✅ Enter: 현재 에피소드를 저장합니다...")

    def start_recording_instance():
        nonlocal running_recording_instance
        running_recording_instance = True
        print("Recording started")

    def stop_recording_instance():
        nonlocal running_recording_instance
        running_recording_instance = False
        print("Recording paused")

    # 환경 리셋 후 액션 차원 확인
    env.sim.reset()
    env.reset()

    # action_space.shape[0]은 배치 차원을 반환할 수 있으므로
    # action_manager.total_action_dim 을 직접 사용
    action_dim = env.action_manager.total_action_dim
    print(f"\n[JointKeyboard] 환경 액션 차원: {action_dim}D")

    # ── 핵심: 로봇 현재 포즈에서 IK 초기 액션을 읽어옴 ──
    # 영벡터(0,0,0,0)로 IK를 구동하면 쿼터니언이 유효하지 않아
    # PhysX BroadPhaseUpdateData 오류가 연쇄 발생함
    init_actions = _get_initial_ik_action(env, action_dim)

    # 키보드 컨트롤러 생성 (초기 액션을 현재 로봇 포즈로 설정)
    teleop_interface = JointKeyboardController(
        action_dim=action_dim,
        delta=args_cli.joint_delta,
        device=args_cli.device,
        initial_actions=init_actions,
    )
    teleop_interface.add_callback("R", reset_recording_instance)
    teleop_interface.add_callback("ENTER", manual_save_and_reset)

    print("\n" + "=" * 55)
    print("  📌 저장 방법")
    print("  Enter  : 현재 에피소드 즉시 저장 후 리셋")
    print("  R      : 저장 없이 에피소드 리셋")
    print("  Backspace : 액션값 초기 포즈로 리셋")
    print("=" * 55 + "\n")

    # UI 설정 (headless 모드에서는 omni.ui 없음)
    label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
    instruction_display = InstructionDisplay(xr=False) if _USE_GUI else None
    if _USE_GUI and instruction_display is not None:
        window = EmptyWindow(env, "Instruction")
        with window.ui_window_elements["main_vstack"]:
            demo_label = ui.Label(label_text)
            subtask_label = ui.Label("")
            instruction_display.set_labels(subtask_label, demo_label)
    else:
        print("[Headless] UI 창 없이 실행합니다. 터미널 출력으로 상태를 확인하세요.")

    subtasks = {}

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running():
            # 키보드에서 액션 읽기
            action = teleop_interface.advance()
            # expand()는 비연속 뷰를 만들어 에러 날 수 있으므로 repeat() 사용
            actions = action.unsqueeze(0).repeat(env.num_envs, 1)

            if running_recording_instance:
                obv = env.step(actions)
                if _USE_GUI and subtasks is not None:
                    if subtasks == {}:
                        subtasks = obv[0].get("subtask_terms")
                    elif subtasks:
                        show_subtask_instructions(instruction_display, subtasks, obv, env.cfg)
            else:
                env.sim.render()

            # 성공 조건 확인
            success_step_count, success_reset_needed = process_success_condition(
                env, success_term, success_step_count
            )
            if success_reset_needed:
                should_reset_recording_instance = True

            # 녹화 카운트 업데이트
            if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
                print(label_text)

            # 목표 데모 수 달성 확인
            if args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                label_text = f"All {current_recorded_demo_count} demonstrations recorded.\nExiting the app."
                if _USE_GUI:
                    instruction_display.show_demo(label_text)
                print(label_text)
                target_time = time.time() + 0.8
                while time.time() < target_time:
                    if rate_limiter:
                        rate_limiter.sleep(env)
                    else:
                        env.sim.render()
                break

            # Enter 키: 수동 저장 처리
            if should_manual_save:
                should_manual_save = False
                env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                env.recorder_manager.set_success_to_episodes(
                    [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                )
                env.recorder_manager.export_episodes([0])
                print(f"[JointKeyboard] 💾 에피소드 저장 완료! (총 {env.recorder_manager.exported_successful_episode_count}개)")
                should_reset_recording_instance = True  # 저장 후 자동 리셋

            # 리셋 처리
            if should_reset_recording_instance:
                success_step_count = handle_reset(env, success_step_count, instruction_display, label_text)
                should_reset_recording_instance = False
                # 리셋 후 로봇 포즈가 바뀌므로 IK 초기 액션도 다시 읽어서 컨트롤러에 반영
                new_init = _get_initial_ik_action(env, action_dim)
                teleop_interface._initial_actions = new_init
                teleop_interface.reset()

            if env.sim.is_stopped():
                break

            if rate_limiter:
                rate_limiter.sleep(env)

    return current_recorded_demo_count


# ─────────────────────────────────────────────────────
# 엔트리포인트
# ─────────────────────────────────────────────────────
def main() -> None:
    rate_limiter = RateLimiter(args_cli.step_hz)

    output_dir, output_file_name = setup_output_directories()

    global env_cfg
    env_cfg, success_term = create_environment_config(output_dir, output_file_name)

    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        exit(1)

    current_recorded_demo_count = run_simulation_loop(env, success_term, rate_limiter)

    # ── 관절 테깅 (revision 클래스 사용) ──
    IsaacLabHdf5Tagger.tag_all(
        env,
        args_cli.dataset_file,
        robot_name="robot",
        auto_align=args_cli.auto_align,
    )

    env.close()
    print(f"Recording session completed with {current_recorded_demo_count} successful demonstrations")
    print(f"Demonstrations saved to: {args_cli.dataset_file}")


if __name__ == "__main__":
    main()
    simulation_app.close()
