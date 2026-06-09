# Isaac Teleop – Docker Compose 통합 환경

NVIDIA Isaac Lab + Isaac Teleop (CloudXR) + WebXR Client를 Docker Compose로 통합 실행하는 환경입니다.
Meta Quest 3의 광학 핸드트래킹으로 시뮬레이션 로봇을 원격 제어합니다.

기본 태스크는 **AI Worker(FFW_SG2) 서버랙 바코드 프레스**(`Isaac-BarcodePress-FFW-SG2-Abs-v0`)이며, 기존 **G1 / GR1T2 Pick&Place** 태스크도 그대로 사용할 수 있습니다.

## 아키텍처

Quest 브라우저는 서버와 **서로 다른 역할의 연결 3개**를 맺습니다. VR 영상·핸드 입력(UDP)과 세션 협상(WSS)을 분리하는 구조입니다.

```
 Meta Quest 3 (WebXR 브라우저)
      │
      │  ① HTTPS (:8453)      ② WSS (:48322)         ③ UDP (:47998)
      │  WebXR 웹앱 로드       CloudXR 시그널링        VR 미디어 스트림
      ▼                      ▼                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Host Machine (GPU)                                                 │
│                                                                     │
│  ┌─────────────────┐   ┌──────────────────────────────────────────┐ │
│  │  webxr-client   │   │  isaac-teleop (network_mode: host)       │ │
│  │  (nginx)        │   │  python -m isaacteleop.cloudxr           │ │
│  │  호스트 :8453   │   │    ├─ WSS Proxy :48322 (TLS, 외부 노출)  │ │
│  │  (→ :8443)      │   │    ├─ Runtime   :49100 (ws, 내부 전용)   │ │
│  └─────────────────┘   │    └─ Media     :47998 (UDP)             │ │
│                        └──────────────┬───────────────────────────┘ │
│                                       │ /openxr (공유 볼륨, OpenXR) │
│                         ┌─────────────┴───────────────────────┐     │
│                         │  isaac-lab                          │     │
│                         │  Isaac Lab 시뮬레이션 + 텔레옵       │     │
│                         │  :8211 (Livestream, 선택)           │     │
│                         └─────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

### Quest ↔ 서버 연결 3종

| # | 포트 | 프로토콜 | 서비스 | 역할 |
|---|------|----------|--------|------|
| ① | **8453** | HTTPS | `webxr-client` | CloudXR WebXR **웹 페이지**(HTML/JS) 제공. Quest가 접속하는 URL |
| ② | **48322** | WSS (TLS WebSocket) | `isaac-teleop` | CloudXR **시그널링** — Connect, 세션 협상, 스트림 설정 등 제어 채널 |
| ③ | **47998** | UDP | `isaac-teleop` | **미디어 스트리밍** — VR 렌더 영상, 오디오, 핸드트래킹 입력 데이터 |

외부(인터넷)에서 Quest로 접속할 때는 ①②③ 모두 라우터 포트포워딩·방화벽 개방이 필요합니다.

### WSS (:48322)가 필요한 이유

CloudXR Runtime은 호스트에서 **49100** 포트로 평문 WebSocket(`ws://`) 시그널링을 띄웁니다.

그런데 Quest는 **HTTPS**(`https://…:8453`)로 페이지를 엽니다. 브라우저는 보안 정책상 **HTTPS 페이지에서 비암호 `ws://`(49100)로 직접 연결하는 것을 차단**합니다(mixed content).

그래서 `isaacteleop.cloudxr`가 **48322에서 TLS를 종료하는 WSS 프록시**를 함께 실행하고, 내부적으로 `ws://127.0.0.1:49100`으로 넘깁니다.

```
Quest  wss://<IP>:48322  ──►  WSS Proxy (:48322)  ──►  CloudXR Runtime ws://:49100
       (브라우저 허용)          (TLS 종료)              (호스트 내부, 포워딩 불필요)
```

WebXR 클라이언트 설정과의 대응:

| WebXR 설정 항목 | 연결 |
|-----------------|------|
| (브라우저 URL) `https://<IP>:8453` | ① 웹앱 |
| **Proxy URL** `wss://<IP>:48322` | ② 시그널링 |
| **Media Address** / **Media Port** | ③ `47998` UDP |

Connect 전에 `https://<IP>:48322/` 링크로 들어가 **자체 서명 SSL 인증서를 한 번 수락**해야 WSS 연결이 막히지 않습니다.

> **49100**은 WSS 프록시가 로컬에서만 쓰므로 외부 포트포워딩 대상이 아닙니다. 예전 `docker/wss-proxy` nginx는 사용하지 않으며, WSS 프록시는 `isaac-teleop` 컨테이너의 `isaacteleop.cloudxr`에 포함되어 있습니다.

## 서비스 구성

| 서비스 | 이미지 | 역할 |
|--------|--------|------|
| `isaac-teleop` | 로컬 빌드 | CloudXR Runtime + WSS Proxy (GPU, host 네트워크) |
| `isaac-lab` | `nvcr.io/nvidia/isaac-lab:2.3.0` | 로봇 시뮬레이션 + 텔레옵 스크립트 (GPU) |
| `webxr-client` | 로컬 빌드 (nginx) | CloudXR WebXR 클라이언트 서빙 |

## 지원 태스크

| 태스크 ID | 로봇 / 씬 | 설명 |
|-----------|-----------|------|
| `Isaac-BarcodePress-FFW-SG2-Abs-v0` | Robotis AI Worker `FFW_SG2` + 서버랙 | **기본 태스크.** 양손 핸드트래킹으로 서버랙의 바코드를 누르는 작업. 전용 스크립트 `scripts/teleop_barcode_ffw.py`로 실행 |
| `Isaac-PickPlace-Locomanipulation-G1-Abs-v0` | Unitree G1 | G1 Pick&Place Locomanipulation |
| `Isaac-PickPlace-GR1T2-Abs-v0` | Fourier GR1T2 | GR1T2 Pick&Place |

### AI Worker 바코드 프레스 태스크

- 씬·로봇·바코드 배치는 `custom_assets/scene/reference.usd`(서버랙 + `FFW_SG2`가 배치된 authored scene)를 **단일 기준**으로 사용합니다. 로봇 reset pose / XR·handtracking anchor도 이 USD에서 자동 추출·정렬됩니다.
- 양손 카메라(Intel RealSense D405 모사)와 헤드(POV) 카메라 영상을 화면에 프리뷰하고, 바코드까지의 방향·거리·성공 카운트를 표시하는 네비 패널을 제공합니다.
- 리프트(`lift_joint`) 제어는 주먹 제스처로 동작합니다 (`TELEOP_LIFT_MODE`): `manual`이면 왼손 주먹 → 하강, 오른손 주먹 → 상승, `auto`이면 오른 손목 높이를 추종합니다.
- handtracking은 **START 시점**에 현재 손 pose와 로봇 손목 pose 간 offset을 캘리브레이션하여 시작 직후 로봇이 튀지 않도록 합니다.
- 성공 판정은 오른손 카메라 중심 광선이 바코드 타겟에 **3초간 연속 접촉**하면 트리거됩니다.
- record 모드에서는 양손 카메라 RGB가 HDF5 `obs/left_hand_cam`, `obs/right_hand_cam`, `obs/head_cam`에 기록되며, 녹화 영상에는 손 마커·프러스텀 등 시각화 인디케이터가 제거됩니다.
- 씬/카메라 얼라인 관련 상세 기록: [docs/troubleshoot_barcode_ffw_scene_alignment.md](docs/troubleshoot_barcode_ffw_scene_alignment.md)

## 검증 환경

| 항목 | 값 |
|------|-----|
| Isaac Lab | 2.3.0 |
| Isaac Teleop | 1.0.193 |
| CloudXR Runtime | 6.1.0 |
| 디바이스 프로필 | `auto-webrtc` |
| 기본 태스크 | `Isaac-BarcodePress-FFW-SG2-Abs-v0` (AI Worker FFW_SG2) |
| 입력 장치 | Quest 3 광학 핸드트래킹 |
| 서버 GPU | NVIDIA GeForce RTX 3090 |

## 시스템 요구사항

**워크스테이션:**
- Ubuntu 22.04 / 24.04
- NVIDIA GPU (RTX 3090 이상, RTX 5090 / RTX PRO 6000 권장)
- Docker 26.0+ / Docker Compose 2.25+
- NVIDIA Container Toolkit
- RAM 64GB 이상 권장

**XR 디바이스:**
- Meta Quest 3 (핸드트래킹 지원)
- Wi-Fi 6 라우터 (5GHz, 워크스테이션과 동일 네트워크)

## 실행 방법

### 1. 사전 준비

```bash
# NGC 로그인 (최초 1회)
docker login nvcr.io
# Username: $oauthtoken
# Password: <NGC API Key from https://ngc.nvidia.com>
```

### 2. 환경 변수 설정

`.env` 파일에서 **반드시** 서버 공인 IP를 수정합니다 (미설정 시 실행 불가):

```bash
# .env 에서 아래 값을 서버 공인 IP로 변경
CXR_PUBLIC_IP=<서버 공인 IP>

# 공인 IP 확인 명령
curl -s ifconfig.me
```

### 3. 라우터 포트포워딩

공유기 관리 페이지에서 아래 포트를 서버 PC의 내부 IP로 포워딩합니다:

| 포트 | 프로토콜 | 용도 |
|------|----------|------|
| 8453 | TCP | WebXR 클라이언트 (HTTPS) |
| 48322 | TCP | CloudXR WSS 시그널링 프록시 |
| 47998 | UDP | CloudXR 미디어 스트리밍 |

### 4. 초기 셋업 (최초 1회)

```bash
# 자동 셋업: 요구사항 확인 + 방화벽 + 이미지 빌드
./scripts/setup.sh

# 또는 수동으로:
docker login nvcr.io          # NGC 로그인
docker compose build           # 이미지 빌드
```

### 5. 실행

```bash
docker compose up -d           # 전체 실행
docker compose logs -f         # 로그 확인
```

### 6. Start AR 자동화 (원격 PC)

Isaac Lab 기본 GUI 모드에서는 시뮬 로딩 후 **AR 패널 → Start AR**을 호스트 PC에서 눌러야 CloudXR 세션이 붙습니다. 원격으로만 서버를 돌릴 때는 `.env`에 아래를 설정하세요:

```bash
XR_HEADLESS=true
```

이때 `isaac-lab`은 `--headless --xr`로 기동하며, [Isaac Lab CloudXR 문서](https://isaac-sim.github.io/IsaacLab/develop/source/how-to/cloudxr_teleoperation.html)와 같이 **AR(OpenXR) 세션이 자동 시작**됩니다. 호스트에 Isaac Sim 창이 없어도 Quest WebXR **Connect**만 하면 됩니다.

로그에서 확인:

```bash
docker compose logs isaac-lab | grep -E 'XR headless|openxr.headless'
# 예: [isaac-lab] ✓ XR headless — Start AR 자동 시작
#     [INFO][AppLauncher]: Loading experience file: .../isaaclab.python.xr.openxr.headless.kit
```

| `XR_HEADLESS` | 동작 |
|---------------|------|
| `true` (권장, 원격) | Start AR 클릭 불필요, X11/VNC 불필요 |
| `false` | `DISPLAY`로 GUI 접속 후 Start AR 수동 클릭 |

설정 변경 후: `docker compose up -d --force-recreate isaac-lab`

### 7. Meta Quest 3 연결

Quest 3 브라우저에서 아래 순서대로 진행합니다:

1. `https://<서버_공인_IP>:8453` 접속
   - "연결이 안전하지 않습니다" → Advanced → Proceed 클릭
2. 설정 입력:
   - **Proxy URL**: `wss://<서버_공인_IP>:48322`
   - **Media Address**: `<서버_공인_IP>`
   - **Media Port**: `47998`
3. SSL 인증서 수락:
   - 페이지에 표시되는 `https://<서버_공인_IP>:48322/` 링크 클릭
   - "Your connection is not private" → Advanced → Proceed to (unsafe)
   - 인증서 수락 확인 후 이전 탭으로 돌아감
4. **Connect** 클릭
5. VR 모드 진입 후 **컨트롤러를 내려놓고** 손으로 조작

## 환경 변수 (.env)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `CXR_PUBLIC_IP` | `220.74.41.204` | 서버 공인 IP (Quest에서 접속하는 주소) |
| `NV_DEVICE_PROFILE` | `auto-webrtc` | 디바이스 프로필 (`auto-webrtc`: Quest/Pico WebXR, `auto-native`: Apple Vision Pro) |
| `NV_CXR_ENABLE_PUSH_DEVICES` | `false` | `false`: 헤드셋 광학 핸드트래킹, `true`: 외부 디바이스 (Manus 글러브) |
| `RUN_MODE` | `teleop` | 실행 모드 (`teleop`: 조작, `record`: 데이터 수집) |
| | | 바코드 FFW 태스크는 `teleop`/`record` 모두 전용 스크립트 `scripts/teleop_barcode_ffw.py`로 실행 (record 시 양손·헤드 카메라 → HDF5 `obs/`) |
| | | G1 Locomanipulation + `record` 시 **로봇 POV** (`robot_pov_cam`)가 HDF5 `obs/`에 자동 기록 |
| `TELEOP_TASK` | `Isaac-BarcodePress-FFW-SG2-Abs-v0` | 텔레옵 시뮬레이션 태스크 (위 [지원 태스크](#지원-태스크) 참조) |
| `TELEOP_DEVICE` | `handtracking` | 입력 장치 (`handtracking` / `keyboard` / `spacemouse`) |
| `XR_HEADLESS` | `false` | `true`: `--headless --xr`로 Start AR 자동 시작 (원격 서버 권장) |
| `HEAD_CAM_WIDTH` / `HEAD_CAM_HEIGHT` | (비움) | [수집 모드] FFW 헤드(POV) 카메라 해상도. 미설정 시 `ROBOT_CAM_*` → 기본 1280×720 폴백 |
| `HAND_CAM_WIDTH` / `HAND_CAM_HEIGHT` | `256` / `160` | [수집 모드] FFW 양손 카메라(D405) 해상도. D405 최대 RGB 1280×720 |
| `RECORD_FPS` | `15` | [수집 모드] HDF5 저장 주파수. 비워두면 태스크 기본값. Robotis AI Worker dataset/policy 권장값: `15` |
| `NUM_DEMOS` | `20` | [수집 모드] 기록할 데모 수 |
| `DATASET_FILE` | `/workspace/user/datasets/dataset_barcode.hdf5` | [수집 모드] 저장될 HDF5 파일 경로 |
| `ISAAC_LAB_VERSION` | `2.3.0` | Isaac Lab 이미지 버전 |
| `WEBXR_HTTPS_PORT` | `8453` | WebXR 클라이언트 HTTPS 포트 |
| `DISPLAY` | `:1` | X11 디스플레이 번호 |

## HDF5 → LeRobot 변환

Isaac Lab(`record_demos.py` 또는 `teleop_barcode_ffw.py --record`)으로 수집한 HDF5를 LeRobot v3 데이터셋으로 변환합니다. conda 환경(`isaaclab310`, LeRobot 0.4+)에서 동작합니다. 상세: [docs/convert_to_lerobot.md](docs/convert_to_lerobot.md)

```bash
# 출력: <hdf5와 같은 폴더>/<파일명>_lerobot_v3/ (카메라 자동 탐지)
# G1 (기본 robot-type unitree_g1)
./scripts/convert_hdf5_to_lerobot.sh workspace/datasets/dataset_g1_260520_0652.hdf5

# FFW 바코드 태스크 (head_cam / left_hand_cam / right_hand_cam)
./scripts/convert_hdf5_to_lerobot.sh workspace/datasets/dataset_barcode_260529.hdf5 --robot-type ffw_sg2
```

## 자주 쓰는 명령어

```bash
# 전체 실행 / 종료
docker compose up -d
docker compose down

# 특정 서비스 재시작 (설정 변경 후)
docker compose up -d --force-recreate isaac-teleop
docker compose restart isaac-lab

# 로그 확인
docker compose logs -f                  # 전체
docker compose logs -f isaac-teleop     # CloudXR 로그
docker compose logs -f isaac-lab        # 시뮬레이션 로그

# CloudXR 상세 로그
docker exec isaac-teleop ls /openxr/logs/
docker exec isaac-teleop tail -50 /openxr/logs/cxr_streamsdk.*.log

# WebXR 클라이언트만 재빌드
docker compose build --no-cache webxr-client
docker compose up -d webxr-client

# 컨테이너 진입
docker compose exec isaac-lab bash
docker compose exec isaac-teleop bash

# Scene editor만 실행 (CloudXR/teleop 없이 Isaac Sim GUI, custom_assets 저장 가능)
./scripts/run_scene_editor.sh

# custom_assets의 기존 USD를 열고 직접 저장
./scripts/run_scene_editor.sh \
  --open custom_assets/env/server_rack_v6.1/server_rack_teleop.usd

# 서버랙 USD 내부 reference 수리 (defaultPrim instancing 시 메시가 안 보일 때)
#   /visuals, /meshes geometry를 /network_rack 아래로 inline 복사해 teleop용 USD 생성
python scripts/repair_server_rack_usd.py

# 포트 리스닝 확인
ss -tulnp | grep -E '(48322|49100|47998|8453)'

# 볼륨 포함 완전 정리
docker compose down -v
```

## 네트워크 포트 요약

| 포트 | 프로토콜 | 서비스 | 용도 | 포트포워딩 필요 |
|------|----------|--------|------|:---:|
| 8453 | TCP | webxr-client | HTTPS WebXR 클라이언트 | ✅ |
| 48322 | TCP | isaac-teleop | WSS 시그널링 프록시 (TLS) | ✅ |
| 47998 | UDP | isaac-teleop | CloudXR 미디어 스트리밍 | ✅ |
| 49100 | TCP | isaac-teleop | CloudXR Runtime 시그널링 (내부) | - |
| 8211 | TCP | isaac-lab | Isaac Sim Livestream | - |

## 트러블슈팅

### WebXR 클라이언트 페이지가 안 뜨거나 깨짐

**증상**: Quest 브라우저에서 접속 시 빈 화면, JS 에러, 또는 SVG 아이콘 누락

**원인**: `webxr-client` Dockerfile에서 NVIDIA 호스팅 사이트의 에셋(JS 청크, SVG)을 모두 다운로드하지 않음

**해결**:
```bash
# Dockerfile에 누락된 에셋이 있는지 확인
docker exec webxr-client ls -la /usr/share/nginx/html/

# 필요한 파일: index.html, bundle.js, favicon.ico,
#   168.bundle.js, 372.bundle.js, 427.bundle.js,
#   play-circle.svg, arrow-uturn-left.svg, arrow-left-start-on-rectangle.svg

# 재빌드
docker compose build --no-cache webxr-client
docker compose up -d webxr-client
```

### "no response Media Server" 에러

**증상**: WebXR 클라이언트에서 Connect 후 "no response Media Server" 표시

**원인**: UDP 47998 포트가 라우터에서 포워딩되지 않았거나, ICE 설정이 NAT 환경과 충돌

**해결**:
1. 라우터에서 UDP 47998 포트포워딩 추가
2. `.env`에 아래 설정 확인:
   ```
   CXR_PUBLIC_IP=<서버 공인 IP>
   ```
3. `docker-compose.yml`에 `NV_CXR_STREAMSDK_ENABLE_ICE=0` 확인
4. Quest WebXR 클라이언트에서 Media Address/Port 명시적 입력:
   - **Media Address**: `<서버 공인 IP>`
   - **Media Port**: `47998`

### VR 화면은 보이지만 로봇이 안 움직임 (핸드트래킹 미작동)

**증상**: VR 스트리밍은 되지만 손 움직임이 로봇에 반영되지 않음

**원인**: 디바이스 프로필 또는 Push Device 설정 오류

**해결**:
```bash
# 환경변수 확인
docker exec isaac-teleop env | grep -E '(NV_DEVICE|NV_CXR)'

# 올바른 값:
#   NV_DEVICE_PROFILE=auto-webrtc       (Quest3 ❌ → auto-webrtc ✅)
#   NV_CXR_ENABLE_PUSH_DEVICES=false    (true ❌ → false ✅)
#   NV_CXR_STREAMSDK_ENABLE_ICE=0
```

- `NV_DEVICE_PROFILE=Quest3`는 네이티브 CloudXR 클라이언트용. **WebXR 브라우저에서는 `auto-webrtc` 사용**
- `NV_CXR_ENABLE_PUSH_DEVICES=true`는 Manus 글러브 등 외부 디바이스용. **Quest 3 광학 핸드트래킹은 `false`**

추가 체크:
- Quest 컨트롤러를 내려놓았는지 확인 (컨트롤러 감지 시 핸드트래킹 비활성화)
- `isaac-teleop` 설정 변경 후 반드시 `isaac-lab`도 재시작 (OpenXR IPC 소켓 재생성)

### CloudXR 로그에 "No packets seen on control channel"

**증상**: 스트리밍 연결 후 입력 데이터가 서버에 도달하지 않음

**원인**: NAT 환경에서 ICE 협상 실패

**해결**:
```bash
# docker-compose.yml의 isaac-teleop 환경변수에 추가
NV_CXR_STREAMSDK_ENABLE_ICE=0
```
NAT 뒤 서버에서는 ICE를 비활성화하고 직접 미디어 주소 지정 방식을 사용해야 합니다.
([CloudXR NAT Configuration 문서](https://docs.nvidia.com/cloudxr-sdk/latest/requirement/network_setup.html) 참고)

### isaac-teleop 재시작 후 isaac-lab이 동작하지 않음

**원인**: `isaac-teleop`이 재시작되면 `/openxr` 공유 볼륨의 IPC 소켓이 재생성됨.
`isaac-lab`은 이전 소켓을 참조하고 있어 연결이 끊김.

**해결**:
```bash
docker compose up -d --force-recreate isaac-teleop
docker compose restart isaac-lab
```

### GPU 인식 실패

```bash
# NVIDIA Container Toolkit 테스트
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

### NGC 이미지 풀 실패

```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <NGC API Key>
```

## 참고 문서

- [Isaac Teleop 공식 문서](https://nvidia.github.io/IsaacTeleop/main/index.html)
- [Isaac Teleop Quick Start](https://nvidia.github.io/IsaacTeleop/main/getting_started/quick_start.html)
- [CloudXR 네트워크 설정](https://docs.nvidia.com/cloudxr-sdk/latest/requirement/network_setup.html)
- [Isaac Lab CloudXR Teleoperation](https://isaac-sim.github.io/IsaacLab/develop/source/how-to/cloudxr_teleoperation.html)
- [Isaac Lab Mimic (모방학습)](https://isaac-sim.github.io/IsaacLab/develop/source/overview/imitation-learning/teleop_imitation.html)
- [HDF5 → LeRobot 변환 가이드](docs/convert_to_lerobot.md)
- [Barcode FFW 씬/카메라 얼라인 트러블슈팅](docs/troubleshoot_barcode_ffw_scene_alignment.md)
