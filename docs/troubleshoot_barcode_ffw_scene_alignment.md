# Barcode FFW SG2 씬/카메라 얼라인 트러블슈팅 기록

이 문서는 `Isaac-BarcodePress-FFW-SG2-Abs-v0` 텔레옵에서 `custom_assets/scene/reference.usd` 기반 씬을 사용하면서, 손 카메라 모델 위치와 가상 카메라 위치가 어긋나던 문제를 정리한 기록입니다.

## 증상

- 텔레옵 UI에서 손목 카메라 영상/프러스텀이 실제 로봇 모델에 붙은 카메라 위치와 다르게 보임.
- 시작 직후 로봇 팔이 의도하지 않게 앞으로 벌어진 상태에서 내려오는 움직임이 발생함.
- 손 추적 텔레옵 시작 시 손 좌표와 로봇 손목 좌표가 바로 맞물리지 않아, 카메라가 뒤틀린 것처럼 보임.
- 기존 환경은 일부 오브젝트를 동적으로 배치하고 있어, authored USD 씬 기준 좌표와 런타임 좌표가 달라질 여지가 있었음.

## 원인

### 1. 동적 씬 생성과 authored 씬 기준 좌표가 섞여 있었음

기존 태스크는 서버랙/타겟/로봇 구성을 코드에서 동적으로 구성하는 흐름이 있었고, 새로 만든 씬은 `custom_assets/scene/reference.usd` 안에 서버랙과 `FFW_SG2`가 이미 배치된 상태였습니다.

이 상태에서 일부 좌표만 기존 방식으로 계산하면, USD 안에서 눈으로 맞춰둔 로봇/카메라/타겟의 상대 위치가 런타임에서 다시 변형될 수 있습니다.

해결 방향은 태스크의 기준을 `reference.usd` 하나로 고정하는 것이었습니다.

관련 코드:

- `scripts/custom_tasks/teleop_barcode_press/barcode_press_env_cfg.py`
- `REFERENCE_SCENE_USD = custom_assets/scene/reference.usd`
- 로봇 prim path: `{ENV_REGEX_NS}/ReferenceScene/FFW_SG2`
- 카메라 prim path: `{ENV_REGEX_NS}/ReferenceScene/FFW_SG2/.../camera_*_link/hand_rgb`

### 2. 로봇 초기 joint pose가 USD authored pose와 달랐음

`ffw_sg2_cfg.py`에 팔 joint 초기값이 별도로 들어가 있었습니다. 이 값 때문에 환경 reset 이후 로봇이 `reference.usd`에 저장된 자세가 아니라, 코드에 정의된 초기 자세로 움직였습니다.

그 결과 손목 링크에 붙은 카메라 역시 authored scene에서 기대한 위치/각도와 달라졌고, "초기에 팔이 앞으로 벌어진 상태에서 내려오는" 현상으로 보였습니다.

해결:

- `scripts/custom_tasks/teleop_barcode_press/ffw_sg2_cfg.py`의 arm joint 초기값을 authored articulation pose와 맞게 `0.0` 기준으로 정리함.
- `reference.usd`에 저장된 로봇 배치와 Isaac Lab 초기 상태가 서로 싸우지 않도록 함.

### 3. START 전 handtracking 입력이 바로 로봇 액션처럼 쓰였음

handtracking 모드에서는 START를 누르기 전에도 XR hand pose가 계속 들어옵니다. 이 pose를 바로 로봇 목표 pose로 쓰면, 사용자의 현재 손 위치와 로봇 손목의 현재 위치 차이가 그대로 점프/비틀림으로 나타납니다.

특히 손목 카메라는 end-effector 링크를 따라가기 때문에, START 시점의 기준 보정 없이 tracking pose가 적용되면 카메라가 모델에 붙은 위치와 다르게 움직이는 것처럼 보일 수 있습니다.

해결:

- START 전에는 현재 로봇 pose를 `hold_action`으로 유지.
- START를 누르는 순간의 XR hand pose와 현재 로봇 wrist pose 사이의 transform offset을 계산.
- 이후 handtracking action에 이 offset을 적용해서, START 순간에는 로봇이 튀지 않고 현재 위치에서 자연스럽게 이어지도록 함.

관련 코드:

- `scripts/teleop_barcode_ffw.py`
- `current_robot_action(env)`
- `calibrate_tracking_action(tracking_action, robot_action)`
- `apply_tracking_calibration(tracking_action, offsets)`
- `teleoperation_active`
- `hold_action`
- `start_calibration_pending`

## 적용한 수정 요약

### Reference USD를 태스크의 단일 기준 씬으로 사용

`barcode_press_env_cfg.py`에서 로봇 spawn이 `reference.usd`를 기준으로 동작하도록 정리했습니다.

```python
REFERENCE_SCENE_USD = os.path.join(CUSTOM_ASSETS_DIR, "scene/reference.usd")
```

로봇 reset pose는 별도 하드코딩 값을 우선하지 않고, `reference.usd` 내부의 로봇 prim transform에서 자동으로 읽습니다.

```python
REFERENCE_ROBOT_PRIM_PATH = "/World/FFW_SG2"
```

같은 pose를 `XrCfg.anchor_pos`, `XrCfg.anchor_rot`에도 사용합니다. 이 값이 `(0, 0, 0)`으로 남아 있으면 로봇은 `reference.usd` 위치에 있어도 handtracking/XR 기준점은 월드 중앙이 되어, 사용자가 보는 텔레옵 시작점이 로봇 기준이 아니라 중앙 기준으로 보일 수 있습니다.

단, `anchor_rot`은 로봇 rotation을 그대로 쓰지 않고 내부 yaw 보정 `-90도`를 곱합니다. 현재 reference 씬에서 로봇은 서버랙 방향인 월드 `-X`를 보고 있고, XR 사용자 기준 정면은 `+Y`로 쓰는 것이 자연스러워서 이 보정이 필요합니다. 이 값은 태스크 코드의 자동 정렬 규칙으로 관리하고 `.env` 관리 항목으로 노출하지 않습니다.

로봇은 authored scene 내부의 prim을 그대로 바라보도록 설정했습니다.

```python
prim_path="{ENV_REGEX_NS}/ReferenceScene/FFW_SG2"
```

카메라도 로봇 모델 안의 카메라 링크 하위에 생성되도록 맞췄습니다.

```python
.../arm_l_link7/camera_l_bottom_screw_frame/camera_l_link/hand_rgb
.../arm_r_link7/camera_r_bottom_screw_frame/camera_r_link/hand_rgb
```

### 초기 팔 자세를 authored pose와 일치

`ffw_sg2_cfg.py`에서 arm joint 초기값을 별도 벌림 자세가 아니라 `0.0` 기준으로 맞췄습니다.

이 변경으로 reset 직후 로봇이 USD에서 의도한 기본 자세와 다르게 움직이는 문제가 사라졌습니다.

### START 기반 handtracking 보정 추가

`teleop_barcode_ffw.py`에서 handtracking 모드는 START 전까지 로봇을 유지하고, START 순간에 현재 손 pose와 현재 로봇 pose 사이의 offset을 캘리브레이션합니다.

이후 입력은 다음 흐름으로 처리됩니다.

```text
XR raw hand pose
  -> START 시점 offset 적용
  -> calibrated hand pose
  -> robot action
```

이 방식 덕분에 START 직후 로봇 팔과 손목 카메라가 갑자기 다른 좌표계로 끌려가지 않습니다.

## 확인 방법

### 환경 설정

`.env`에서 BarcodePress FFW 태스크와 handtracking을 사용합니다.

```bash
TELEOP_TASK=Isaac-BarcodePress-FFW-SG2-Abs-v0
TELEOP_DEVICE=handtracking
XR_HEADLESS=false
```

### 씬 기준 파일

태스크가 아래 파일을 기준으로 실행되는지 확인합니다.

```bash
custom_assets/scene/reference.usd
```

### 실행 중 확인 포인트

- reset 직후 로봇 팔이 앞으로 벌어지며 내려오지 않는지 확인.
- START 전에는 로봇이 현재 자세를 유지하는지 확인.
- START를 누른 직후 손목 카메라 프러스텀/영상이 모델의 카메라 링크와 함께 움직이는지 확인.
- 손을 움직였을 때 카메라가 end-effector에 고정된 것처럼 따라오는지 확인.

## 재발 시 체크리스트

- `reference.usd` 내부의 `FFW_SG2` prim 경로가 바뀌지 않았는지 확인.
- `barcode_press_env_cfg.py`의 카메라 prim path가 실제 USD의 `camera_l_link`, `camera_r_link` 경로와 일치하는지 확인.
- `ffw_sg2_cfg.py`에 팔 joint 초기값 override가 다시 들어가지 않았는지 확인.
- handtracking START 전에 `teleoperation_active`가 켜져 있지 않은지 확인.
- START 순간 `start_calibration_pending`이 한 번 실행되는지 확인.
- USD asset reference가 깨진 경우 `reference.usd`가 `../env/...`, `../robot/...`처럼 현재 위치 기준으로 올바르게 참조하는지 확인.

## 핵심 결론

이번 문제는 카메라 하나의 offset 문제라기보다, 씬 기준 좌표, 로봇 초기 joint pose, handtracking 시작 기준점이 동시에 어긋난 문제였습니다.

최종적으로는 `reference.usd`를 단일 기준으로 삼고, 로봇 초기 자세를 authored pose와 맞춘 뒤, START 시점에 XR hand pose를 현재 로봇 wrist pose로 캘리브레이션하면서 손 카메라 모델과 가상 카메라 위치가 정상적으로 얼라인되었습니다.
