# LlamaFactory Image for Ascend NPU

LlamaFactory Ascend NPU images are designed for Huawei Ascend Atlas NPUs and provide a ready-to-use LlamaFactory environment. Built on Ascend CANN container images, they include Python, PyTorch, TorchNPU, DeepSpeed, LlamaFactory, and other components.

For installation and troubleshooting details, see the [English NPU installation guide](https://llamafactory.readthedocs.io/en/latest/multibackend/npu/npu_installation.html).

## Quick Reference

- Image registries:
  - `docker.io/hiyouga/llamafactory`
  - `quay.io/ascend/llamafactory`
- Dockerfile: `docker/docker-npu/Dockerfile`
- Docker Compose file: `docker/docker-npu/docker-compose.yml`

The following `latest` NPU image tags are available:

| Hardware series | Operating system | Tag |
| --- | --- | --- |
| A2 | Ubuntu 22.04 | `latest-910b-ubuntu` |
| A3 | Ubuntu 22.04 | `latest-a3-ubuntu` |
| A2 | openEuler 24.03 | `latest-910b-openeuler` |
| A3 | openEuler 24.03 | `latest-a3-openeuler` |

For historical NPU image tags and links to their corresponding archived Dockerfiles, see [Supported Tags](./supported_tags.md).

## Image Overview

The image includes the following core components:

| Component | Version |
| --- | --- |
| CANN | `9.1.0` |
| Python | `3.12` |
| PyTorch | `2.10.0` |
| TorchNPU | `2.10.0.post2` |
| torchvision / torchaudio | `0.25.0` / `2.10.0` |
| Transformers | Latest compatible version at build time |
| Triton Ascend | `3.2.1` |
| DeepSpeed | Latest compatible version at build time |
| LlamaFactory | Installed from the repository build context |

The image does not include model weights or datasets. Mount or download them separately and comply with their respective licenses and acceptable-use requirements.

## Image Tags

NPU `latest` and release tags use different formats; the following rules do not apply to CUDA images.

Non-release builds reuse the following short tags. Each scheduled build updates the image referenced by the corresponding tag:

```text
latest-<chip>-<os>
```

| Field | Values | Description |
| --- | --- | --- |
| `chip` | `910b` or `a3` | Ascend chip model supported by the image |
| `os` | `ubuntu` or `openeuler` | Container operating system family |

Future NPU release images will use the full tag format below. Historical release images used earlier naming formats and do not follow this rule.

```text
<LlamaFactory-version>-cann<CANN-version>-torch_npu<TorchNPU-version>-<chip>-<os>-<Python-version>
```

| Field | Example | Description |
| --- | --- | --- |
| `LlamaFactory-version` | `0.9.5` | LlamaFactory release version |
| `CANN-version` | `9.1.0` | Parsed from the CANN base image tag |
| `TorchNPU-version` | `2.10.0.post2` | Full TorchNPU version used by the image, including suffixes such as `.postN` |
| `chip` | `910b` or `a3` | Ascend chip model supported by the image |
| `os` | `ubuntu22.04` or `openeuler24.03` | Container operating system family and version |
| `Python-version` | `py3.12` | Parsed from the CANN base image tag |

For example:

```text
0.9.5-cann9.1.0-torch_npu2.10.0.post2-a3-ubuntu22.04-py3.12
```

## Quick Start

### Prerequisites

Before starting a container:

1. Install an Ascend driver and firmware compatible with the CANN version in the image.
2. Verify that `npu-smi info` works on the host.
3. Install Docker with permission to access the required Ascend device nodes and driver files.

Driver, firmware, CANN, TorchNPU, and the target Ascend hardware must be mutually compatible.

### Pull and Run

The following example starts the latest A2 Ubuntu image with one NPU. Adjust `DOCKER_IMAGE` and the `--device` options for your environment.

```bash
CONTAINER_NAME=llamafactory-npu
DOCKER_IMAGE=hiyouga/llamafactory:latest-910b-ubuntu

docker run --rm -it \
  --net=host \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /data:/data \
  --name "$CONTAINER_NAME" \
  "$DOCKER_IMAGE" \
  /bin/bash
```

The host path for `npu-smi` may be `/usr/local/sbin/npu-smi` on some driver installations. Adjust the mount source when necessary. Add more `--device=/dev/davinci<N>` options to expose additional NPUs.

Verify the runtime inside the container:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available())"
llamafactory-cli help
```

### Build Locally

Run the build from the repository root. The following example builds the A2 Ubuntu variant:

```bash
docker build \
  -f ./docker/docker-npu/Dockerfile \
  --build-arg BASE_IMAGE=quay.io/ascend/cann:9.1.0-910b-ubuntu22.04-py3.12 \
  --build-arg PIP_INDEX=https://pypi.org/simple \
  -t llamafactory:npu-910b-ubuntu \
  .
```

Available build arguments:

| Argument | Default | Purpose |
| --- | --- | --- |
| `BASE_IMAGE` | `quay.io/ascend/cann:9.1.0-910b-ubuntu22.04-py3.12` | Selects the base image that matches the device model and container operating system |
| `PIP_INDEX` | `https://pypi.org/simple` | Selects the Python package index |
| `PYTORCH_INDEX` | `https://download.pytorch.org/whl/cpu` | Selects the PyTorch wheel index used with TorchNPU |
| `HTTP_PROXY` | Empty | Provides an optional HTTP/HTTPS proxy during the build |

### Start with Docker Compose

The preceding `docker build` command invokes the Dockerfile directly. It builds an image but does not start a container. Docker Compose does not use a separate build implementation: it reads the presets in `docker-compose.yml`, reuses the same Dockerfile, and selects a hardware-series and operating-system combination through a profile. Each `up -d` command below starts the selected container in the background. If the image is not available locally, Docker Compose builds it first:

```bash
cd docker/docker-npu

# A2 with Ubuntu
docker compose --profile a2-ubuntu up -d

# A3 with Ubuntu
docker compose --profile a3-ubuntu up -d

# A2 with openEuler
docker compose --profile a2-openeuler up -d

# A3 with openEuler
docker compose --profile a3-openeuler up -d
```

To build an image with Docker Compose without starting a container, use `docker compose --profile <profile> build`.

## Hardware Support and Compatibility Notes

- A2 images use the `910b` CANN base image; A3 images use the `a3` CANN base image.
- The image build targets both x86-64 (`linux/amd64`) and AArch64 (`linux/arm64`) hosts. The CPU architecture is independent of whether the hardware series is A2 or A3.
- Ubuntu 22.04 and openEuler 24.03 refer to the operating system inside the container.
- Legacy NPU tags are replaced by the `latest-<910b|a3>-<ubuntu|openeuler>` format.
- Validate the exact driver, firmware, CANN, and SoC combination before production deployment.

## License and Disclaimer

LlamaFactory is distributed under the [Apache License 2.0](../../LICENSE).

Ascend CANN, TorchNPU, Triton Ascend, DeepSpeed, base operating-system packages, model weights, datasets, and other third-party components are governed by their respective licenses and terms. The LlamaFactory license does not replace or override those terms.

The image is provided on an "AS IS" basis, without warranties or conditions of any kind. Users are responsible for validating hardware and software compatibility, securing the container and its runtime configuration, complying with applicable licenses and laws, and reviewing model and dataset terms before training, evaluation, or deployment.
