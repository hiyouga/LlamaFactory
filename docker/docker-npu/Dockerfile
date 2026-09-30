# https://hub.docker.com/r/ascendai/cann/tags

ARG BASE_IMAGE=quay.io/ascend/cann:9.1.0-910b-ubuntu22.04-py3.12
FROM ${BASE_IMAGE}

# Installation arguments
ARG PIP_INDEX=https://pypi.org/simple
ARG HTTP_PROXY=""
ARG PYTORCH_INDEX=https://download.pytorch.org/whl/cpu

# Define environments
ENV MAX_JOBS=16
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV DEBIAN_FRONTEND=noninteractive
ENV NODE_OPTIONS=""
ENV PIP_ROOT_USER_ACTION=ignore
ENV http_proxy="${HTTP_PROXY}"
ENV https_proxy="${HTTP_PROXY}"

# Use Bash instead of default /bin/sh
SHELL ["/bin/bash", "-c"]

# Set the working directory
WORKDIR /app

# Change pip source
RUN pip config set global.index-url "${PIP_INDEX}" && \
    pip config set global.extra-index-url "${PIP_INDEX}" && \
    pip install --no-cache-dir --upgrade pip packaging wheel setuptools editables "hatchling>=1.18.0"

# Copy the application into the image
COPY . /app

# Install torch-npu
RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh
RUN pip uninstall -y torch torchvision torchaudio
RUN pip install --no-cache-dir -r requirements/npu.txt --index-url "${PYTORCH_INDEX}"
RUN pip install --no-cache-dir -r requirements/triton_ascend.txt
RUN pip install --no-cache-dir -r requirements/deepspeed.txt
RUN pip install --no-cache-dir -e . --no-build-isolation && \
    pip install --no-cache-dir -r requirements/metrics.txt --no-build-isolation

# Set up volumes
# VOLUME [ "/root/.cache/huggingface", "/app/shared_data", "/app/output" ]

# Expose port 7860 for LLaMA Board
ENV GRADIO_SERVER_PORT=7860
EXPOSE 7860

# Expose port 8000 for API service
ENV API_PORT=8000
EXPOSE 8000

# unset proxy
ENV http_proxy=
ENV https_proxy=

# Reset pip config
RUN pip config unset global.index-url && \
    pip config unset global.extra-index-url
