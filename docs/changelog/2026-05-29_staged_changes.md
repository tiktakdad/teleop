# Staged Changes Changelog

- 기준: staged diff (`git diff --cached`)
- 범위: teleop launcher, barcode FFW task, recording utilities, docs, runtime env, submodule pins

## 한줄 요약
바코드 FFW 텔레옵을 중심으로 실행기, 리타게팅, 녹화/데이터 수집 도구, 씬/카메라 배치, 문서화가 함께 정리된 staged 변경이다.

## 주요 변경점

- `scripts/run.sh`
  - 새 실행 런처를 추가해 `XR_HEADLESS`, `RUN_MODE`, `TELEOP_TASK`를 대화식으로 고를 수 있게 했다.
  - 기본 태스크를 바코드 FFW와 G1 Pick&Place 중에서 선택하도록 정리했다.
  - `.env`와 현재 환경 값을 읽어와 실행 요약을 보여주고, 최종적으로 `docker compose up` 진입점을 단순화한다.

- `scripts/isaac_lab_entrypoint.sh`
  - X11 `DISPLAY`를 자동 감지하고, 유효하지 않으면 `XR_HEADLESS=true`로 전환하는 안전장치를 넣었다.
  - barcode FFW record 모드에서는 전용 teleop 스크립트로 수집하도록 분기한다.

- `scripts/teleop_barcode_ffw.py`
  - record 모드, 데모 경로, 성공 판정 스텝 수 같은 녹화 옵션을 추가했다.
  - 녹화 시 HDF5 exporter를 사용하고, 성공 후 자동 저장/리셋 흐름을 처리한다.
  - 바코드 타겟 가시성, 손 마커, 프러스텀 시각화를 숨기거나 조정해 녹화 영상의 오염을 줄인다.
  - 카메라 프리뷰 하단에 네비게이션/카운트 패널을 연동한다.

- `scripts/custom_tasks/teleop_barcode_press/barcode_press_env_cfg.py`
  - reference USD에서 바코드 prim과 로봇 pose를 자동 추출하도록 강화했다.
  - IK 가중치와 damping, gain 값을 환경변수로 튜닝할 수 있게 했다.
  - 바코드 타겟 좌표와 헤드 카메라 기본 해상도도 새 기본값으로 정리했다.

- `scripts/custom_tasks/teleop_barcode_press/retargeters/ffw_sg2_retargeter.py`
  - 손 위치 변위에 스케일과 clutch 기준점을 적용해 팔 도달 거리를 더 유연하게 조정할 수 있게 했다.

- `scripts/custom_tasks/teleop_barcode_press/ffw_sg2_cfg.py`
  - `head_joint1` 초기값을 아래보기 방향으로 바꾸는 환경변수를 추가했다.

- `scripts/custom_tasks/teleop_barcode_press/utils/camera_preview_displays.py`
  - 카메라 프리뷰 묶음을 옆으로 이동시키는 배치 옵션을 추가했다.
  - 오른손 프리뷰 아래에 네비게이션 패널을 새로 그려 목표 방향, 거리, 성공 진행률을 표시한다.
  - 프리뷰 전체 표시와 네비 패널 표시를 독립적으로 켜고 끌 수 있다.

- `scripts/joint_targeter/isaaclab_hdf5_tagger.py`
  - Isaac Lab HDF5 데이터셋에 관절 이름과 액션 메타데이터를 태깅하는 새 유틸리티를 추가했다.
  - state/action 불일치 시 관절 공간으로 자동 변환하는 경로를 지원한다.

- `scripts/joint_targeter/record_demos_keyboard_joint.py`
  - 키보드 기반 관절 녹화용 새 스크립트를 추가했다.
  - HDF5 녹화와 태거 연동 흐름을 문서화/자동화하는 쪽으로 확장됐다.

- `scripts/joint_targeter/IsaacLabHdf5TaggerREADME.md`
  - HDF5 태거와 auto-align 변환 흐름을 설명하는 문서를 추가했다.

- `README.md`
  - 기본 태스크를 AI Worker 바코드 프레스 태스크로 바꾸고, 지원 태스크와 record/convert 사용법을 정리했다.
  - HDF5 → LeRobot 변환, 씬/카메라 얼라인, record 환경변수 안내를 보강했다.

- `docker-compose.yml`
  - DISPLAY와 GPU runtime 설정을 조정하고, head/hand 카메라 해상도 환경변수를 추가했다.
  - XR/graphics/display 권한을 명시해 실행 안정성을 높였다.

- `scripts/setup.sh`
  - X11 권한과 실행 준비 절차를 보완하는 방향으로 수정됐다.

- `.gitmodules`, `sub_modules/robotis_applications`, `sub_modules/robotis_lab`
  - 외부 서브모듈 참조가 갱신됐다.

## 운영 영향

- 기본 실행은 이제 바코드 FFW teleop과 record 흐름을 중심으로 안내된다.
- 데이터 수집은 새 HDF5 태거와 keyboard recording 스크립트로 확장된다.
- 카메라/프리뷰/리타게팅/환경 설정이 함께 정리돼 씬 변경과 수집 작업이 더 일관되게 이어진다.
