# IsaacLabHdf5Tagger

Isaac Lab 환경에서 수집된 HDF5 데이터셋에 관절/액션 메타데이터를 자동으로 태깅하고,
state-action 표현 방식이 불일치할 경우 관절 공간으로 자동 변환하는 유틸리티입니다.

---

## 파일 구조

```
scripts/tools/
  isaaclab_hdf5_tagger.py       ← 메인 모듈 (import해서 사용)
  record_demos_keyboard_joint.py← 키보드 녹화 스크립트 (태거 사용)
  IsaacLabHdf5TaggerREADME.md   ← 이 문서
```

---

## 사용법

### 기본 import

```python
from isaaclab_hdf5_tagger import IsaacLabHdf5Tagge
```

### 1. 한 줄로 모든 메타데이터 태깅 (권장)

```python
# 녹화 종료 후, env.close() 전에 호출
IsaacLabHdf5Tagger.tag_all(
    env,                          # Isaac Lab 환경 객체
    dataset_path="./datasets/data.hdf5",
    robot_name="robot",           # 씬 내 로봇 에셋 이름
    auto_align=False,             # True: state/action 불일치 시 자동 변환
)
```

### 2. 개별 메서드 호출

```python
# 관절 이름만 태깅
IsaacLabHdf5Tagger.tag_joint_names(env, dataset_path)

# 액션 이름 + 표현 방식 태깅
IsaacLabHdf5Tagger.tag_action_info(env, dataset_path)

# 기존 HDF5의 action을 관절 공간으로 변환 (Isaac Lab 없이도 실행 가능)
IsaacLabHdf5Tagger.convert_actions_to_joint_space("./datasets/data.hdf5")
```

### 3. CLI에서 auto_align 사용

```bash
./isaaclab.sh -p scripts/tools/record_demos_keyboard_joint.py \
  --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 \
  --dataset_file ./datasets/data.hdf5 \
  --num_demos 5 --enable_pinocchio \
  --auto_align    # ← state/action 불일치 시 자동 변환
```

---

## 전체 실행 흐름

```
tag_all(env, dataset_path, auto_align)
│
├── [Step 1] tag_joint_names()
│   │
│   ├── env.scene["robot"].joint_names 에서 동적 추출
│   │   (실패 시 env.unwrapped.scene → env.robot 순으로 탐색)
│   │
│   └── HDF5 root attrs에 저장:
│       └── robot_joint_names = ["left_hip_pitch_joint", ...]  (예: 43개)
│
├── [Step 2] tag_action_info()
│   │
│   ├── env.action_manager._terms 에서 모든 액션 텀 순회
│   │   │
│   │   └── 각 텀을 _analyze_single_term()으로 분류 (아래 분기점 참조)
│   │
│   ├── space_type 종합 결정
│   │   ├── 텀이 전부 같은 타입 → 해당 타입
│   │   └── 섞여 있음           → "mixed"
│   │
│   ├── state와 일치 여부 체크 (_check_state_action_consistency)
│   │
│   └── HDF5 root attrs에 저장:
│       ├── action_names      = ["left_wrist_pos_x", ...]
│       ├── action_space_type = "eef_pose" | "joint_position" | "mixed" | ...
│       └── action_term_info  = JSON 문자열 (텀별 상세)
│
└── [Step 3] auto_align 분기
    │
    ├── auto_align=False → 종료
    │
    └── auto_align=True
        │
        ├── action_space_type이 "joint_position" → "변환 불필요" 출력, 종료
        │
        └── action_space_type이 "eef_pose" / "mixed" / "velocity_command"
            │
            └── convert_actions_to_joint_space() 실행
                │
                ├── 각 에피소드에 대해:
                │   ├── action[t] = obs/robot_joint_pos[t+1]  (t=0..T-2)
                │   ├── action[T-1] = obs/robot_joint_pos[T-1] (마지막 유지)
                │   └── actions_original에 원본 백업 (backup=True)
                │
                └── HDF5 메타데이터 업데이트:
                    ├── action_names       → robot_joint_names와 동일 (43개)
                    ├── action_space_type  → "joint_position"
                    ├── original_action_dim → 변환 전 차원 (예: 32)
                    └── conversion_source   → "obs/robot_joint_pos[t+1]"
```

---

## 액션 텀 분류 분기점 (_analyze_single_term)

각 액션 텀은 IO descriptor의 `action_type` 필드를 기준으로 7단계로 분류됩니다.
하드코딩된 로봇 이름이나 차원 수는 사용하지 않습니다.

```
term.IO_descriptor.action_type 확인
│
├── Case 1: "JointAction"
│   조건: io_desc.joint_names 존재 & len == action_dim
│   결과: space_type = "joint_position"
│   이름: io_desc.joint_names 그대로 사용
│   예시: JointPositionAction, JointVelocityAction
│
├── Case 2: "TaskSpaceAction"
│   조건: action_type == "TaskSpaceAction"
│   결과: space_type = "eef_pose"
│   이름: body_name + dim에 따라 자동 생성
│         3D → pos_x/y/z
│         6D → pos + rot_r/p/y
│         7D → pos + quat_w/x/y/z
│   예시: DifferentialInverseKinematicsAction
│
├── Case 3: "PinkInverseKinematicsAction"
│   조건: action_type == "PinkInverseKinematicsAction"
│   결과: space_type = "eef_pose"
│   이름: cfg.target_eef_link_names 키별 pos+quat (7D씩)
│         + cfg.hand_joint_names (실제 관절 이름)
│   예시: G1 상체 IK (14D EEF + 14D hand = 28D)
│
├── Case 4: "non holonomic actions" 포함
│   조건: action_type에 "non holonomic" 문자열 포함
│   결과: space_type = "velocity_command"
│   이름: body_name_vx, _vy, _wz
│   예시: NonHolonomicAction
│
├── Case 5: io_desc.joint_names 존재 (커스텀 텀)
│   조건: io_desc에 joint_names 속성 존재 & len == action_dim
│   결과: space_type = "joint_position"
│   이름: joint_names 그대로 사용
│
├── Case 6: term._joint_names 존재
│   조건: term 객체에 _joint_names 속성 존재 & len == action_dim
│   결과: space_type = "joint_position"
│   이름: _joint_names 그대로 사용
│   예시: AgileBasedLowerBodyAction 등 커스텀 텀
│
└── Case 7: 모두 실패 (fallback)
    결과: space_type = "unknown"
    이름: term_name_0, term_name_1, ... (제네릭)
    경고: "[Tagger] IO descriptor로 분류 불가" 출력
```

---

## 입력값 상세

### tag_all()

| 인자 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `env` | Isaac Lab Env | (필수) | `gym.make(task, cfg=env_cfg).unwrapped` 로 생성된 환경 |
| `dataset_path` | str | (필수) | 녹화된 HDF5 파일 경로 (예: `./datasets/data.hdf5`) |
| `robot_name` | str | `"robot"` | `env.scene[robot_name]`으로 접근할 로봇 에셋 이름 |
| `auto_align` | bool | `False` | `True`: action이 관절 공간이 아니면 자동 변환 |

### convert_actions_to_joint_space()

| 인자 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `dataset_path` | str | (필수) | HDF5 파일 경로 |
| `robot_name` | str | `"robot"` | 메타데이터 기록용 |
| `obs_key` | str | `"robot_joint_pos"` | 관절 위치 observation 키 |
| `backup` | bool | `True` | 원본 action을 `actions_original`에 백업 |

---

## HDF5에 저장되는 메타데이터

### 기본 태깅 (auto_align=False)

```
root attrs:
  robot_joint_names   = [b"left_hip_pitch_joint", ...]     # state 관절 이름
  action_names        = [b"left_wrist_pos_x", ...]         # action 차원별 이름
  action_space_type   = "eef_pose"                          # 표현 방식
  action_term_info    = '[ {"term_name": "upper_body_ik",   # JSON 문자열
                            "space_type": "eef_pose",
                            "action_dim": 28, ...}, ... ]'
```

### auto_align 변환 후

```
root attrs:
  robot_joint_names   = [b"left_hip_pitch_joint", ...]     # 변경 없음
  action_names        = [b"left_hip_pitch_joint", ...]     # robot_joint_names와 동일!
  action_space_type   = "joint_position"                    # 변환됨
  original_action_dim = 32                                  # 변환 전 차원 기록
  conversion_source   = "obs/robot_joint_pos[t+1]"         # 변환 소스 기록

data/demo_X:
  actions             = shape (T, 43)                       # 변환됨
  actions_original    = shape (T, 32)                       # 원본 백업
```

---

## auto_align 변환 원리

```
시간축:  t=0        t=1        t=2        ...  t=T-1

원본:
  action:  IK포즈₀    IK포즈₁    IK포즈₂         IK포즈ₜ₋₁    (32D, EEF 공간)
             ↓ IK풀기   ↓ IK풀기   ↓ IK풀기
  joint:   q₀    →    q₁    →    q₂    →  ...  → qₜ₋₁       (43D, 관절 공간)

변환 후:
  action:  q₁         q₂         q₃         ...  qₜ₋₁       (43D, 관절 공간)
           ↑          ↑          ↑                ↑
     joint_pos[t+1]  joint_pos[t+2]         joint_pos[T-1]

  의미: "이 시점에 명령을 내렸더니, 다음 스텝에 이 관절 위치가 됐다"
  마지막 프레임: action[T-1] = joint_pos[T-1] (다음 스텝 없으므로 현재 유지)
```

---

## state/action 일치 검사

`tag_action_info()` 실행 시 자동으로 체크됩니다:

| action_space_type | 검사 방법 | 메시지 |
|---|---|---|
| `joint_position` | action 이름과 state 관절 이름의 겹침 비율 | >80%: "관절 공간 일치", 그 외: "불일치" |
| `eef_pose` | 무조건 불일치 | "action=EEF포즈, state=관절각도 -> 공간 불일치" |
| `mixed` | 무조건 부분 불일치 | "action=혼합, state=관절각도 -> 부분 불일치" |
| `unknown` | 판별 불가 | "자동 판별 불가" |

---

## hdf5_to_lerobot_claude.py와의 호환성

태거가 저장한 속성을 LeRobot 변환기가 읽어 `info.json`에 반영합니다:

| HDF5 속성 | 컨버터 사용 여부 | info.json 반영 |
|---|---|---|
| `robot_joint_names` | ✅ 읽음 | `observation.robot_joint_pos.names` |
| `action_names` | ✅ 읽음 | `action.names` |
| `action_space_type` | 미활용 | (향후 확장 가능) |
| `action_term_info` | 미활용 | (향후 확장 가능) |

### auto_align 변환 후 호환성

```
변환 전: action_names 32개, actions.shape[1] = 32 → 32 == 32 ✅
변환 후: action_names 43개, actions.shape[1] = 43 → 43 == 43 ✅
```

action_names의 길이와 실제 action 차원이 항상 일치하므로 추가 수정 없이 호환됩니다.

---

## 시나리오별 결과 요약

| 시나리오 | action_dim | action_space_type | state 일치 | GR00T 학습 |
|---|---|---|---|---|
| IK teleop, auto_align=False | 32 | eef_pose | ❌ | ⚠ 불일치 주의 |
| IK teleop, auto_align=True | 43 | joint_position | ✅ | ✅ RELATIVE 정상 |
| Joint teleop (direct) | 43 | joint_position | ✅ | ✅ RELATIVE 정상 |
| 단일 팔 (Franka 등) | 7 | joint_position | ✅ | ✅ |
| DifferentialIK | 6~7 | eef_pose | ❌ | ⚠ 불일치 주의 |
