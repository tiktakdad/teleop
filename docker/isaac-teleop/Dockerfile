FROM python:3.11-slim

ARG ISAACTELEOP_VERSION=1.0.0
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libx11-dev \
        patchelf \
        git \
        ca-certificates \
        openssl \
        libvulkan1 \
        libvulkan-dev \
        vulkan-tools \
        libegl1 \
        libgl1 \
        libgles2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    --timeout=300 \
    --retries=5 \
    "isaacteleop[cloudxr,retargeters]~=${ISAACTELEOP_VERSION}" \
    --extra-index-url https://pypi.nvidia.com

RUN git clone --depth 1 https://github.com/NVIDIA/IsaacTeleop.git /opt/IsaacTeleop

WORKDIR /opt/IsaacTeleop

EXPOSE 48322 49100 47998/udp

CMD ["bash"]
