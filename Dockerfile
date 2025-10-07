FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# System dependencies required for building and running Python packages used by the project
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    curl \
    ca-certificates \
    cmake \
    pkg-config \
    unzip \
    xz-utils \
    libffi-dev \
    libgl1 \
    libglib2.0-0 \
    libglvnd0 \
    libegl1 \
    libglu1-mesa \
    libglew-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxi6 \
    libxrandr2 \
    libxxf86vm1 \
    libxmu6 \
    libxmu-dev \
    libx11-xcb1 \
    libxcb-randr0-dev \
    libxkbcommon-dev \
    libxkbcommon-x11-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libusb-1.0-0 \
    libudev1 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && python3 -m pip install --upgrade pip setuptools wheel \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install CoppeliaSim required by PyRep
ARG COPPELIA_SIM_URL=https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
ENV COPPELIASIM_ROOT=/opt/CoppeliaSim \
    LD_LIBRARY_PATH=/opt/CoppeliaSim:$LD_LIBRARY_PATH \
    QT_QPA_PLATFORM_PLUGIN_PATH=/opt/CoppeliaSim

RUN curl -L ${COPPELIA_SIM_URL} -o /tmp/coppeliaSim.tar.xz \
    && tar -xf /tmp/coppeliaSim.tar.xz -C /opt \
    && mv /opt/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 ${COPPELIASIM_ROOT} \
    && rm /tmp/coppeliaSim.tar.xz

WORKDIR /workspace

COPY requirements.txt ./

# Pre-install build-time Python dependencies that PyRep expects via setup_requires
RUN python -m pip install --no-cache-dir \
    "pycparser==2.22" \
    "cffi==1.14.2"

RUN python -m pip install --no-cache-dir -r requirements.txt

CMD ["bash"]
