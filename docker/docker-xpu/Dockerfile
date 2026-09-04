# https://hub.docker.com/r/intel/deep-learning-essentials/tags
# Unlike the CUDA/ROCm bases this image ships no PyTorch, and its version must match the
# host Intel GPU driver — see README.md.

ARG BASE_IMAGE=intel/deep-learning-essentials:2026.1.0-devel-ubuntu24.04
FROM ${BASE_IMAGE}

# Installation arguments
ARG PIP_INDEX=https://pypi.org/simple
ARG PYTORCH_INDEX=https://download.pytorch.org/whl/xpu
# Must stay in the same compute-runtime series as the base image — see README.md ("ocloc").
ARG OCLOC_VERSION=26.18.38308.1

# Define environments
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_ROOT_USER_ACTION=ignore
# The base image's Python is distro-managed (PEP 668); nothing to protect in a container.
ENV PIP_BREAK_SYSTEM_PACKAGES=1
# expandable_segments trips a Level-Zero bug on XPU — see README.md ("Multi-GPU and OPTIM_TORCH").
ENV OPTIM_TORCH=0

# Use Bash instead of default /bin/sh
SHELL ["/bin/bash", "-c"]

# Set the working directory
WORKDIR /app

# Install pip, Python dev headers and a C/C++ toolchain — see README.md ("Python.h: No such file or directory").
RUN PY_MM="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        "python${PY_MM}-dev" python3-pip build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install ocloc — pinned .deb rather than apt, see README.md ("ocloc and torch.compile").
RUN wget -q "https://github.com/intel/compute-runtime/releases/download/${OCLOC_VERSION}/intel-ocloc_${OCLOC_VERSION}-0_amd64.deb" \
        -O /tmp/intel-ocloc.deb && \
    dpkg -i /tmp/intel-ocloc.deb && \
    rm -f /tmp/intel-ocloc.deb

# Change pip source — pip is deliberately not upgraded, see README.md ("How pip is installed").
RUN pip config set global.index-url "${PIP_INDEX}" && \
    pip config set global.extra-index-url "${PIP_INDEX}" && \
    pip install --no-cache-dir packaging wheel setuptools editables "hatchling>=1.18.0"

# Copy the application into the image
COPY . /app

# Install PyTorch for XPU — all three wheels pinned as a set, see README.md ("Pinned torch stack")
RUN pip install --no-cache-dir -r requirements/xpu.txt --index-url "${PYTORCH_INDEX}"

# Install LLaMA Factory
# metrics.txt is explicit because this project defines no extras, so `.[metrics]` would no-op.
RUN pip install --no-cache-dir -e . --no-build-isolation && \
    pip install --no-cache-dir -r requirements/metrics.txt

# Optional accelerators — see README.md ("Optional accelerators").
# py-cpuinfo must land first: deepspeed's own setup.py imports deepspeed/ops/adam/cpu_adam.py
# during metadata generation (before pip installs deepspeed's declared deps), which needs it.
# deepspeed is pinned separately from requirements/deepspeed.txt's <=0.18.4: that range's XPU
# accelerator imports IPEX's DpcppBuildExtension unconditionally, which no longer exists on a
# post-IPEX torch. Fixed upstream in 0.18.7 (plain torch.utils.cpp_extension.BuildExtension).
RUN pip install --no-cache-dir py-cpuinfo && \
    (DS_BUILD_OPS=0 pip install --no-cache-dir "deepspeed==0.19.6" \
        || echo "WARNING: deepspeed install failed - deepspeed training unavailable in this image")
RUN pip install --no-cache-dir -r requirements/bitsandbytes.txt \
        || echo "WARNING: bitsandbytes install failed - 4-bit quantization (QLoRA) unavailable in this image"

# Source the oneAPI environment in every interactive shell
RUN echo "source /opt/intel/oneapi/setvars.sh --force" >> /root/.bashrc

# Set up volumes
# VOLUME [ "/root/.cache/huggingface", "/app/shared_data", "/app/output" ]

# Expose port 7860 for LLaMA Board
ENV GRADIO_SERVER_PORT=7860
EXPOSE 7860

# Expose port 8000 for API service
ENV API_PORT=8000
EXPOSE 8000

# Reset pip config
RUN pip config unset global.index-url && \
    pip config unset global.extra-index-url

# oneAPI must be on the loader path before torch can open its XPU backend
CMD ["bash", "-c", "source /opt/intel/oneapi/setvars.sh --force && exec llamafactory-cli webui"]
