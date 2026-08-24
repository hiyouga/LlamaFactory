# Docker Setup for Intel GPUs

This directory contains Docker configuration files for running LLaMA Factory with Intel GPU (XPU) support.

## Image Details

| Component | Version |
|---|---|
| Base OS | Ubuntu 24.04 LTS (x86_64) |
| Intel DLE base | [intel/deep-learning-essentials:2026.1.0-devel-ubuntu24.04](https://hub.docker.com/r/intel/deep-learning-essentials) |
| Python | 3.12 |
| PyTorch | 2.13.0+xpu |
| Intel GPU runtime | Bundled inside DLE (libze-intel-gpu 26.18.x, oneAPI 2026.1) |

> **The Intel compute runtime (Level-Zero, OpenCL ICD) is fully bundled inside the DLE base image.**
> You do NOT need to install any Intel GPU compute packages on the host.
> The host only needs the i915/xe **kernel driver** to expose `/dev/dri` device nodes.
> The DLE runtime communicates with the kernel driver through those nodes.
>
> The only compatibility requirement: the DLE's bundled userspace runtime must support
> the host kernel driver's ABI. DLE 2026.1 works with kernel driver ≥ 26.18.
> Verify your host kernel driver: `dpkg -l libze-intel-gpu1 | grep -oP '\d+\.\d+\.\d+'`

## Prerequisites

### 1. Docker

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install docker.io

# Or from the official Docker repository:
# https://docs.docker.com/engine/install/
```

### 2. Docker Compose (recommended)

```bash
# Ubuntu/Debian
sudo apt-get install docker-compose-v2

# Or the latest version:
# https://docs.docker.com/compose/install/
```

### 3. Intel GPU Kernel Driver (host only)

The container bundles its own Intel compute runtime — you only need the **kernel driver**
on the host to expose `/dev/dri`.

```bash
# Add the Intel GPU PPA
sudo apt-get install -y gpg-agent wget
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
    sudo gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
    https://repositories.intel.com/gpu/ubuntu noble client" | \
    sudo tee /etc/apt/sources.list.d/intel-graphics.list
sudo apt-get update

# Kernel driver only — no compute runtime packages needed on the host
sudo apt-get install -y intel-i915-dkms intel-fw-gpu
sudo reboot
```

After reboot, verify `/dev/dri` is populated:

```bash
ls /dev/dri/
# Expected: card0  card1  renderD128  renderD129  by-path/
```

For the official step-by-step guide, see the [Intel GPU Installation Guide](https://dgpu-docs.intel.com/installation-guides/installing-packages-from-the-intel-ppa.html).

> [!IMPORTANT]
> Enable **Resizable BAR** in your system BIOS before proceeding. Without it you may see
> `Bus error (core dumped)` or degraded GPU performance. See [Intel's guide](https://www.intel.com/content/www/us/en/support/articles/000090831/graphics.html).

### 4. Add your user to the GPU groups

```bash
sudo usermod -aG render,video $USER
# Log out and back in for group membership to take effect
```

Verify GPU access on the host:

```bash
clinfo --list | grep Device
# Expected output (Arc/BMG example):
#  `-- Device #0: Intel(R) Arc(TM) B770 Graphics
#  `-- Device #0: Intel(R) Arc(TM) B580 Graphics
# `-- Device #0: Intel(R) Arc(TM) Pro B70 Graphics
# `-- Device #1: Intel(R) Arc(TM) Pro B70 Graphics
```

## Usage

### Using Docker Compose (Recommended)

```bash
cd docker/docker-xpu/
docker compose up -d
docker compose exec llamafactory bash
```

Inside the container, source the oneAPI environment and verify GPU access:

```bash
source /opt/intel/oneapi/setvars.sh --force
python3 -c "import torch; print(torch.xpu.device_count(), 'XPU device(s) found')"
```

### Using Docker Run

```bash
# Build the image (from the repo root)
docker build -t llamafactory:xpu \
    -f docker/docker-xpu/Dockerfile .

# Run the container
RENDER_GID=$(getent group render | cut -d: -f3)
VIDEO_GID=$(getent group video  | cut -d: -f3)
docker run -it --rm \
    --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    --group-add ${RENDER_GID} \
    --group-add ${VIDEO_GID} \
    --ipc=host \
    -p 7860:7860 \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --name llamafactory \
    llamafactory:xpu bash
```

## Troubleshooting

### GPU Not Detected (`torch.xpu.device_count()` returns 0)

1. **Kernel driver too old for the DLE runtime** — the most common cause.
   The DLE image bundles its own compute runtime; the host only needs the kernel driver.
   But the DLE runtime's ABI must be compatible with the host kernel driver version.
   Check both:
   ```bash
   # Host kernel driver version
   dpkg -l libze-intel-gpu1 | grep -oP '\d+\.\d+\.\d+'
   # DLE bundled runtime version (inside the container)
   docker run --rm llamafactory:xpu dpkg -l libze-intel-gpu1 | grep -oP '\d+\.\d+\.\d+'
   ```
   If the host driver is older than the DLE runtime requires, update the host kernel driver:
   ```bash
   sudo apt-get install -y intel-i915-dkms intel-fw-gpu && sudo reboot
   ```
   If you cannot update the host driver, use an older DLE base image (e.g. `2025.3.x`) that
   ships a matching runtime — see `ARG BASE_IMAGE` in `Dockerfile`.

2. **Missing `/dev/dri` device** — ensure `--device /dev/dri` is passed (done automatically by `docker compose`).

3. **Missing group membership** — the container process must be in the `render` (renderD*) and `video` (card*) groups. `docker compose` sets these automatically via `group_add`. For manual `docker run`, pass `--group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3)`.

4. **`by-path` mount missing** — required for multi-process Level-Zero IPC. Always mount `-v /dev/dri/by-path:/dev/dri/by-path` (included in `docker-compose.yml`).

### Permission Denied on `/dev/dri`

```bash
# Ensure your host user is in the render and video groups
sudo usermod -aG render,video $USER
newgrp render   # apply without logout
```

### `SYCL Backends mismatch` at runtime

This means two incompatible SYCL runtimes loaded in the same process. Common cause: the PyTorch XPU wheel bundles `libsycl.so.8` (oneAPI 2025.3), but the container's system oneAPI is 2026.1 (`libsycl.so.9`). Check:

```bash
# inside the container
ldd $(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")/lib/libtorch_xpu.so \
    | grep libsycl
ldconfig -p | grep libsycl
```

## Additional Notes

- The container automatically sources `/opt/intel/oneapi/setvars.sh` in every interactive shell (`~/.bashrc`). For non-interactive scripts, source it explicitly.
- For training, `llamafactory-cli train` dispatches automatically via `torchrun` for multi-GPU.
