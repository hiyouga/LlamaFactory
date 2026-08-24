# LlamaFactory — Intel XPU (Arc / BMG)
#
# Base: Intel Deep Learning Essentials (DLE) 2026.1.0
#   - oneAPI 2026.1 toolkit, oneCCL 2022.1, MPI 2021.18
#   - torch 2.13.0+xpu
#   - LlamaFactory installed on top
#
# IMPORTANT — DLE version must match the host Intel GPU driver:
#   DLE 2026.1 → libze-intel-gpu 26.18.x → requires host driver ≥ 26.18
#   DLE 2025.3 → libze-intel-gpu 25.18.x → only works with host driver ≤ 26.09
#   Verify host: dpkg -l libze-intel-gpu1 | grep -oP '\d+\.\d+\.\d+'
#
# Build (from repo root):
#   docker build -t llamafactory:xpu \
#     -f docker/docker-xpu/Dockerfile .
#
# Or via docker compose (recommended):
#   cd docker/docker-xpu && docker compose up -d
#   docker compose exec llamafactory bash
#
# Run (manual):
#   docker run -it --rm \
#     --device /dev/dri \
#     -v /dev/dri/by-path:/dev/dri/by-path \
#     --group-add $(getent group render | cut -d: -f3) \
#     --group-add $(getent group video  | cut -d: -f3) \
#     --ipc=host \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     llamafactory:xpu bash

ARG BASE_IMAGE=intel/deep-learning-essentials:2026.1.0-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_ROOT_USER_ACTION=ignore
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Disable the LlamaFactory launcher's DDP "optimization" injection.
# With OPTIM_TORCH=1 (the launcher default), launcher.py sets
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True on every rank. On Intel XPU
# that expandable-segments (VMM) allocator path trips an L0 driver bug: a 2-GPU
# DDP run SIGSEGVs, and a 2-GPU FSDP2 SFT run hangs in the step-1 forward — same
# root trigger, two signatures. Throughput impact of turning it off is negligible
# on XPU (measured ±<1% noise). Force it off so the default CLI is stable on 2 GPUs.
ENV OPTIM_TORCH=0

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /app

# Python dev headers + a C/C++ toolchain.
# The Intel Triton backend JIT-compiles a C driver (driver.c) at first XPU
# kernel launch, which #include <Python.h> — without the matching -dev headers
# it fails with "fatal error: 'Python.h' file not found". This breaks every
# Triton-compiled path on XPU, including bitsandbytes 4-bit (QLoRA) quant.
# Derive the exact dev package from the interpreter so it stays correct if the
# base image bumps the Python minor version (e.g. python3.12 -> python3.13).
RUN PY_MM="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        "python${PY_MM}-dev" build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Bootstrap pip (DLE has python3.12 but no pip)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3 - --break-system-packages && \
    python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel editables "hatchling>=1.18.0"

# Install PyTorch XPU (torch 2.13.0 targets the oneAPI 2026.1 runtime in DLE 2026.1.0).
# Pin the matching +xpu torchvision/torchaudio explicitly: if left unpinned, pip may
# resolve the PLAIN (CPU) torchvision/torchaudio wheels, which are built against a
# different torch ABI -> `RuntimeError: operator torchvision::nms does not exist`,
# cascading through transformers (imports torchvision.io) -> peft -> llamafactory-cli.
# torchvision 0.28.0 / torchaudio 2.11.0 are the versions paired with torch 2.13.0.
RUN python3 -m pip install --no-cache-dir \
        torch==2.13.0 torchvision==0.28.0+xpu torchaudio==2.11.0+xpu \
        --index-url https://download.pytorch.org/whl/xpu

# Lock the torch XPU stack for EVERY subsequent pip install via a constraint file.
# pyproject asks for plain ranges (torch>=2.4.0, torchvision>=0.19.0, ...) resolved
# against the default PyPI index, and deepspeed/bitsandbytes declare torch as a dep —
# any of these could otherwise swap our +xpu wheels for a plain CPU/CUDA build while
# resolving. PIP_CONSTRAINT pins them so no later install can move torch off +xpu,
# which removes the need for a second --force-reinstall repair layer (~1.6GB saved:
# --force-reinstall duplicates the torch .so files into a new layer without reclaiming
# the originals). The wheels installed above already satisfy the constraint, so no
# extra download happens.
RUN printf 'torch==2.13.0\ntorchvision==0.28.0+xpu\ntorchaudio==2.11.0+xpu\n' \
        > /etc/pip-xpu-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-xpu-constraints.txt

# Install LlamaFactory + metrics deps (required).
# NOTE: this project migrated to a flat pyproject.toml with NO
# [project.optional-dependencies], so there is no `metrics` extra — pip would
# silently drop it (WARNING: does not provide the extra 'metrics') and skip
# nltk/jieba/rouge-chinese, breaking BLEU/ROUGE eval later. Install the metrics
# deps explicitly via requirements/metrics.txt, matching docker/docker-cuda.
COPY . /app
RUN python3 -m pip install --no-cache-dir -e . --no-build-isolation && \
    python3 -m pip install --no-cache-dir -r requirements/metrics.txt

# Optional accelerators (deepspeed / bitsandbytes). `|| true` so a build on a
# host without the toolchain for these still produces a usable image.
RUN DS_BUILD_OPS=0 python3 -m pip install --no-cache-dir -r requirements/deepspeed.txt || true && \
    python3 -m pip install --no-cache-dir -r requirements/bitsandbytes.txt || true

# Source the oneAPI environment in every interactive shell
RUN echo "source /opt/intel/oneapi/setvars.sh --force" >> /root/.bashrc


EXPOSE 7860
EXPOSE 8000
CMD ["bash", "-c", "source /opt/intel/oneapi/setvars.sh --force && exec llamafactory-cli webui"]
