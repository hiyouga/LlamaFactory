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

The Intel compute runtime is bundled inside the DLE base image — no GPU compute packages are needed on the host, only the kernel driver. The bundled runtime must be compatible with the host kernel driver; verify with `dpkg -l libze-intel-gpu1 | grep -oP '\d+\.\d+\.\d+'`.

## Prerequisites

### 1. Docker & Docker Compose

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install docker.io docker-compose-v2
```

See the [official Docker install docs](https://docs.docker.com/engine/install/) for other distros or newer versions.

### 2. Intel GPU Kernel Driver (host only)

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

After reboot, verify `/dev/dri` is populated: `ls /dev/dri/` (expect `card0`, `renderD128`, etc).

See the [Intel GPU Installation Guide](https://dgpu-docs.intel.com/installation-guides/installing-packages-from-the-intel-ppa.html) for details.

> [!IMPORTANT]
> Enable **Resizable BAR** in your system BIOS before proceeding, or you may see `Bus error (core dumped)` or degraded performance. See [Intel's guide](https://www.intel.com/content/www/us/en/support/articles/000090831/graphics.html).

### 3. Add your user to the GPU groups

```bash
sudo usermod -aG render,video $USER
# Log out and back in for group membership to take effect
```

(Optional) sanity-check the host side with `sudo apt-get install -y clinfo && clinfo --list | grep Device`. The check that actually matters — `torch.xpu.device_count()` — runs inside the container, in Usage below.

## Usage

### Using Docker Compose (Recommended)

```bash
cd docker/docker-xpu/
docker compose up -d
docker compose exec llamafactory bash
```

Verify GPU access inside the container:

```bash
python3 -c "import torch; print(torch.xpu.device_count(), 'XPU device(s) found')"
```

### Using Docker Run

```bash
# Build the image (from the repo root)
docker build -t llamafactory:xpu -f docker/docker-xpu/Dockerfile .

# Run the container
docker run -it --rm \
    --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    --group-add $(getent group render | cut -d: -f3) \
    --group-add $(getent group video  | cut -d: -f3) \
    --ipc=host \
    -p 7860:7860 \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --name llamafactory \
    llamafactory:xpu bash
```

## Build arguments

| Argument | Default | Purpose |
|---|---|---|
| `BASE_IMAGE` | `intel/deep-learning-essentials:2026.1.0-devel-ubuntu24.04` | oneAPI / GPU runtime version |
| `PIP_INDEX` | `https://pypi.org/simple` | PyPI mirror for everything except the torch wheels |
| `PYTORCH_INDEX` | `https://download.pytorch.org/whl/xpu` | where the `+xpu` torch wheels come from |
| `OCLOC_VERSION` | `26.18.38308.1` | pinned `intel-ocloc` build — must match `BASE_IMAGE`'s compute-runtime series |

```bash
docker build -t llamafactory:xpu -f docker/docker-xpu/Dockerfile . \
    --build-arg PIP_INDEX=https://pypi.org/simple
```

## Design notes

Why the image is built the way it is — skip to [Troubleshooting](#troubleshooting) if you just want to run it.

### Pinned torch stack

`torch`, `torchvision` and `torchaudio` install together from `requirements/xpu.txt` as `+xpu` builds, via `--index-url https://download.pytorch.org/whl/xpu` (not `--extra-index-url`, which would let a plain PyPI build win). Pinning all three avoids `RuntimeError: operator torchvision::nms does not exist` from an ABI-mismatched pair.

No pip constraint file is needed: every other `torch` requirement in the dependency graph is a lower bound that `2.13.0+xpu` already satisfies, so nothing later swaps it out.

If the XPU wheels ever get replaced with plain ones, `torch.xpu.device_count()` returns `0`. Restore with:

```bash
pip install --force-reinstall -r requirements/xpu.txt \
    --index-url https://download.pytorch.org/whl/xpu
```

### How pip is installed

The DLE base ships Python 3.12 with no pip and no `ensurepip` (Ubuntu strips it). pip comes from apt (`python3-pip`) and is **left at the distro version** rather than upgraded — apt's pip has no `RECORD` file, so `pip install --upgrade pip` fails with:

```
ERROR: Cannot uninstall pip 24.0, RECORD file not found. Hint: The package was installed by debian.
```

— taking `setuptools`/`wheel`/`hatchling` down with it in the same command.

### Multi-GPU and `OPTIM_TORCH`

The image sets `ENV OPTIM_TORCH=0`. The launcher's default (`OPTIM_TORCH=1`) sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, whose allocator path trips a Level-Zero driver bug on 2-GPU XPU runs (DDP segfault, FSDP2 hang). Disabling it costs <1% throughput. Re-enable per-run with `OPTIM_TORCH=1 llamafactory-cli train ...` to test it. It is tracked in intel, will be update or remove this ENV when it is resolve with PyTorch or level zero etc fixes.

### Optional accelerators

`deepspeed` and `bitsandbytes` install on a best-effort basis — neither is required to train on XPU, so a build failure only prints a warning instead of failing the image:

```
WARNING: deepspeed install failed - deepspeed training unavailable in this image
WARNING: bitsandbytes install failed - 4-bit quantization (QLoRA) unavailable in this image
```

Both install and work on XPU today (verified: `torch.compile`, LoRA + DeepSpeed ZeRO-2 SFT, and bitsandbytes 4-bit all pass on real Arc Pro B60 hardware). `deepspeed` is pinned to `==0.19.6` here, separately from `requirements/deepspeed.txt`'s shared `<=0.18.4` cap: any version in that range detects XPU as the active accelerator and imports `torch.utils.cpp_extension.DpcppBuildExtension` — a class that used to be provided by `intel_extension_for_pytorch` (IPEX). IPEX is no longer part of this stack, so that import fails with `ImportError: cannot import name 'DpcppBuildExtension'`. deepspeed dropped the IPEX dependency and switched to the stock `BuildExtension` in `0.18.7`, so anything from there onward works; `0.19.6` is the latest release at time of writing.

Check the build log, or verify inside the container with `python3 -c "import deepspeed"` / `import bitsandbytes`. Install manually if missing:

```bash
DS_BUILD_OPS=0 pip install "deepspeed==0.19.6"
pip install -r requirements/bitsandbytes.txt
```

### `ocloc` and `torch.compile`

`torch.compile` lowers to Intel Triton, which shells out to `ocloc` (the Intel offline GPU compiler) — not bundled in the DLE base, so compiled paths fail with `FileNotFoundError: 'ocloc'` without it. The Dockerfile installs a single pinned `.deb` from Intel's GitHub releases (`ARG OCLOC_VERSION`) rather than `apt-get install intel-ocloc`, because the Intel graphics PPA keeps only its newest revision and would drag `libze-intel-gpu1` (the host-driver-facing package) up with it.

> [!IMPORTANT]
> If you override `ARG BASE_IMAGE`, move `OCLOC_VERSION` to match — check `docker run --rm <base-image> dpkg -l libze-intel-gpu1 | tail -1` and pick the nearest [compute-runtime release](https://github.com/intel/compute-runtime/releases).

## Troubleshooting

### GPU Not Detected (`torch.xpu.device_count()` returns 0)

1. **Kernel driver too old for the DLE runtime** — compare `dpkg -l libze-intel-gpu1` on the host vs. `docker run --rm llamafactory:xpu dpkg -l libze-intel-gpu1`. Update the host driver (`sudo apt-get install -y intel-i915-dkms intel-fw-gpu && sudo reboot`) or use an older `BASE_IMAGE`.
2. **Missing `/dev/dri` device** — pass `--device /dev/dri` (done automatically by `docker compose`).
3. **Missing group membership** — the container process needs the `render` and `video` groups; `docker compose` sets these via `group_add`. For manual `docker run`, pass `--group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3)`.
4. **`by-path` mount missing** — required for multi-GPU Level-Zero IPC: `-v /dev/dri/by-path:/dev/dri/by-path` (included in `docker-compose.yml`).

### `fatal error: Python.h: No such file or directory`

Intel Triton JIT-compiles a C driver at the first XPU kernel launch, which needs matching `pythonX.Y-dev` headers and a C compiler present in the image — installed at build time even though the error would happen at runtime. If you've swapped the interpreter, reinstall matching headers:

```bash
PY_MM="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
apt-get update && apt-get install -y "python${PY_MM}-dev" build-essential
```

### Permission Denied on `/dev/dri`

```bash
sudo usermod -aG render,video $USER
newgrp render   # apply without logout
```

### `SYCL Backends mismatch` / `libsycl.so.N: cannot open shared object file`

Two SYCL runtimes exist in this image by design — torch's own `intel-sycl-rt` (pip) and the DLE base's oneAPI — and normally share the same `libsycl.so.9` SONAME, so both coexist fine. A mismatch shows up only after changing `torch` or `BASE_IMAGE` independently. Compare versions:

```bash
pip list | grep -E "intel-sycl-rt|dpcpp-cpp-rt"
ls /opt/intel/oneapi/*/lib/libsycl.so.*
```

The `libsycl.so.N` numbers on both sides must match. `ldconfig -p | grep libsycl` returning nothing is expected — neither runtime is in the ldconfig cache.

## Additional Notes

- The container automatically sources `/opt/intel/oneapi/setvars.sh` in every interactive shell (`~/.bashrc`). For non-interactive scripts, source it explicitly.
- For training, `llamafactory-cli train` dispatches automatically via `torchrun` for multi-GPU.
