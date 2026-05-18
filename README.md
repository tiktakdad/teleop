# Isaac Teleop – Docker Compose 통합 환경

NVIDIA Isaac Lab + Isaac Teleop (CloudXR) + WebXR Client를 Docker Compose로 통합 실행하는 환경입니다.
Meta Quest 3의 광학 핸드트래킹으로 시뮬레이션 로봇을 원격 제어합니다.

## 아키텍처

```
 Meta Quest 3 (WebXR 브라우저)
      │
      │  HTTPS (:8453)         WSS (:48322)          UDP (:47998)
      │  WebXR 클라이언트       시그널링               미디어 스트리밍
      ▼                        ▼                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Host Machine (GPU)                                                 │
│                                                                     │
│  ┌─────────────────┐   ┌──────────────────────────────────────────┐ │
│  │  webxr-client   │   │  isaac-teleop (network_mode: host)       │ │
│  │  (nginx)        │   │  CloudXR Runtime (:49100) + WSS Proxy    │ │
│  │  :8080 / :8443  │   │  :48322 (TCP) / :47998 (UDP)            │ │
│  └─────────────────┘   └──────────────┬───────────────────────────┘ │
│                                       │ /openxr (공유 볼륨)         │
│                         ┌─────────────┴───────────────────────┐     │
│                         │  isaac-lab                          │     │
│                         │  Isaac Lab 시뮬레이션 + 텔레옵       │     │
│                         │  :8211 (Livestream)                 │     │
│                         └─────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

## 서비스 구성

| 서비스 | 이미지 | 역할 |
|--------|--------|------|
| `isaac-teleop` | 로컬 빌드 | CloudXR Runtime + WSS Proxy (GPU, host 네트워크) |
| `isaac-lab` | `nvcr.io/nvidia/isaac-lab:2.3.0` | 로봇 시뮬레이션 + 텔레옵 스크립트 (GPU) |
| `webxr-client` | 로컬 빌드 (nginx) | CloudXR WebXR 클라이언트 서빙 |

## 검증 환경

| 항목 | 환경 A | 환경 B |
|------|--------|--------|
| OS | Ubuntu 22.04 | Ubuntu 24.04 |
| GPU | NVIDIA GeForce RTX 3090 | NVIDIA RTX 5090 |
| NVIDIA Driver | 550.x | 580.126.09 |
| CUDA | 12.4 | 13.0 |
| Container Toolkit | 1.17.3 | 1.17.5 |
| Isaac Lab | 2.3.0 | 2.3.0 |
| Isaac Teleop | 1.0.193 | 1.0.193 |
| CloudXR Runtime | 6.1.0 | 6.1.0 |
| 디바이스 프로필 | `auto-webrtc` | `auto-webrtc` |
| 태스크 | `Isaac-PickPlace-GR1T2-Abs-v0` | `Isaac-PickPlace-GR1T2-Abs-v0` |
| 입력 장치 | Quest 3 광학 핸드트래킹 | Quest 3 광학 핸드트래킹 |

## 시스템 요구사항

### 최소 사양 (워크스테이션)

| 항목 | 최소 | 권장 |
|------|------|------|
| OS | Ubuntu 22.04 LTS | Ubuntu 24.04 LTS |
| CPU | 8코어 (Intel i7 / AMD Ryzen 7) | 16코어+ |
| RAM | 32GB | 64GB+ |
| GPU | NVIDIA RTX 3090 (24GB VRAM) | RTX 4090 / RTX 5090 / RTX PRO 6000 |
| NVIDIA Driver | 535+ | 550+ |
| Docker | 26.0+ | 27.0+ |
| Docker Compose | v2.25+ | v2.30+ |
| NVIDIA Container Toolkit | **1.17+** (필수) | 최신 |
| 디스크 | 50GB 여유 공간 | 100GB+ (에셋 캐시 포함) |
| 네트워크 | 유선 1Gbps | 유선 1Gbps |

> **주의**: NVIDIA Container Toolkit **1.13.x 이하**에서는 최신 드라이버(580.x 등)의 Vulkan/EGL 라이브러리를 
> 컨테이너에 마운트하지 못해 `CloudXR runtime failed to start` 오류가 발생합니다.
> 반드시 **1.17 이상**을 사용하세요.

### Docker nvidia 런타임 (필수)

`isaac-teleop` 서비스는 `runtime: nvidia`를 사용합니다. Docker에 nvidia 런타임이 등록되어 있어야 합니다:

```bash
# 등록 확인
docker info 2>/dev/null | grep -i runtime
# 출력에 "nvidia"가 포함되어야 함

# 미등록 시 설정
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### XR 디바이스

- Meta Quest 3 (핸드트래킹 지원)
- Wi-Fi 6 라우터 (5GHz, 워크스테이션과 동일 네트워크)

## 실행 방법

### 1. 환경 변수 설정

`.env.example`을 복사하고 서버 IP를 설정합니다 (미설정 시 실행 불가):

```bash
cp .env.example .env

# .env 에서 CXR_PUBLIC_IP를 서버 IP로 변경
# 로컬 LAN: hostname -I | awk '{print $1}'   (예: 192.168.0.2)
# 외부 접속: curl -s ifconfig.me              (예: 220.74.x.x)
```

### 2. 초기 셋업 (최초 1회)

```bash
# 자동 셋업: 요구사항 확인 + nvidia 런타임 설정 + 방화벽 + 이미지 빌드
./scripts/setup.sh
```

수동 설정 시:
```bash
# nvidia 런타임 등록 (필수)
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 이미지 빌드
docker compose build
```

### 3. 네트워크 설정

**로컬 LAN** (Quest와 서버가 같은 공유기): 추가 설정 불필요

**외부 접속** (Quest가 다른 네트워크): 공유기 관리 페이지에서 포트포워딩 필요

| 포트 | 프로토콜 | 용도 |
|------|----------|------|
| 8453 | TCP | WebXR 클라이언트 (HTTPS) |
| 48322 | TCP | CloudXR WSS 시그널링 프록시 |
| 47998 | UDP | CloudXR 미디어 스트리밍 |

### 4. 실행

```bash
docker compose up -d           # 전체 실행
docker compose logs -f         # 로그 확인
```

### 5. Meta Quest 3 연결

Quest 3 브라우저에서 아래 순서대로 진행합니다:

1. `https://<서버_IP>:8453` 접속
   - "연결이 안전하지 않습니다" → Advanced → Proceed 클릭
2. 설정 입력:
   - **Proxy URL**: `wss://<서버_IP>:48322`
   - **Media Address**: `<서버_IP>` (로컬 LAN에서는 비워둬도 됨)
   - **Media Port**: `47998`
3. SSL 인증서 수락:
   - 페이지에 표시되는 `https://<서버_IP>:48322/` 링크 클릭
   - "Your connection is not private" → Advanced → Proceed to (unsafe)
   - 인증서 수락 확인 후 이전 탭으로 돌아감
4. **Connect** 클릭
5. VR 모드 진입 후 **컨트롤러를 내려놓고** 손으로 조작

## 환경 변수 (.env)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `CXR_PUBLIC_IP` | (필수 설정) | 서버 IP – 외부: 공인 IP, 로컬 LAN: 내부 IP (192.168.x.x) |
| `NV_DEVICE_PROFILE` | `auto-webrtc` | 디바이스 프로필 (`auto-webrtc`: Quest/Pico WebXR, `auto-native`: Apple Vision Pro) |
| `NV_CXR_ENABLE_PUSH_DEVICES` | `false` | `false`: 헤드셋 광학 핸드트래킹, `true`: 외부 디바이스 (Manus 글러브) |
| `TELEOP_TASK` | `Isaac-PickPlace-GR1T2-Abs-v0` | 텔레옵 시뮬레이션 태스크 |
| `TELEOP_DEVICE` | `handtracking` | 입력 장치 (`handtracking` / `keyboard` / `spacemouse`) |
| `ISAAC_LAB_VERSION` | `2.3.0` | Isaac Lab 이미지 버전 |
| `WEBXR_HTTPS_PORT` | `8453` | WebXR 클라이언트 HTTPS 포트 |
| `DISPLAY` | `:1` | X11 디스플레이 번호 |

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

# 포트 리스닝 확인
ss -tulnp | grep -E '(48322|49100|47998|8453)'

# 볼륨 포함 완전 정리
docker compose down -v
```

## 네트워크 포트 요약

| 포트 | 프로토콜 | 서비스 | 용도 | 외부 접속 시 포트포워딩 |
|------|----------|--------|------|:---:|
| 8453 | TCP | webxr-client | HTTPS WebXR 클라이언트 | ✅ |
| 48322 | TCP | isaac-teleop | WSS 시그널링 프록시 (TLS) | ✅ |
| 47998 | UDP | isaac-teleop | CloudXR 미디어 스트리밍 | ✅ |
| 49100 | TCP | isaac-teleop | CloudXR Runtime 시그널링 (내부) | - |
| 8211 | TCP | isaac-lab | Isaac Sim Livestream | - |

> 로컬 LAN에서는 포트포워딩 불필요. `isaac-teleop`이 `network_mode: host`이므로 LAN 내에서 직접 접근 가능.

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

### CloudXR runtime failed to start

**증상**: `CloudXR runtime failed to start, terminating...` 로그 출력 후 컨테이너 종료

**원인 1 – NVIDIA Container Toolkit 버전 불일치**:

Toolkit 1.13.x 이하에서는 최신 드라이버의 `libnvidia-gpucomp.so` 등 Vulkan/EGL 라이브러리를 컨테이너에 마운트하지 못합니다.

```bash
# 현재 버전 확인
nvidia-ctk --version

# 1.17 미만이면 업데이트 필요
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**원인 2 – Docker nvidia 런타임 미등록**:

```bash
# 확인
docker info 2>/dev/null | grep -i runtime
# "nvidia"가 없으면:
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**원인 3 – 컨테이너 내부 Vulkan 드라이버 미감지**:

```bash
# 컨테이너 안에서 직접 확인
docker compose run --rm isaac-teleop vulkaninfo 2>&1 | head -20
# "Found no drivers!" → 위 원인 1, 2 해결 후 재시도
# "VULKANINFO" + "Vulkan Instance Version: 1.x.x" 출력 → 정상
```

### GPU 인식 실패

```bash
# NVIDIA Container Toolkit 테스트
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

### NGC 이미지 풀 실패

`nvcr.io/nvidia/isaac-lab:2.3.0`은 퍼블릭 이미지로 로그인 없이 풀 가능합니다.
만약 다른 NGC 이미지가 필요한 경우:

```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <NGC API Key from https://ngc.nvidia.com>
```

## 참고 문서

- [Isaac Teleop 공식 문서](https://nvidia.github.io/IsaacTeleop/main/index.html)
- [Isaac Teleop Quick Start](https://nvidia.github.io/IsaacTeleop/main/getting_started/quick_start.html)
- [CloudXR 네트워크 설정](https://docs.nvidia.com/cloudxr-sdk/latest/requirement/network_setup.html)
- [Isaac Lab CloudXR Teleoperation](https://isaac-sim.github.io/IsaacLab/develop/source/how-to/cloudxr_teleoperation.html)
- [Isaac Lab Mimic (모방학습)](https://isaac-sim.github.io/IsaacLab/develop/source/overview/imitation-learning/teleop_imitation.html)
