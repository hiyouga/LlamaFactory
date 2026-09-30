# Docker Setup for NVIDIA GPUs

This directory contains Docker configuration files for running LLaMA Factory with NVIDIA GPU support.

## Prerequisites

### Linux-specific Requirements

Before running the Docker container with GPU support, you need to install the following packages:

1. **Docker**: The container runtime
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install docker.io

   # Or install Docker Engine from the official repository:
   # https://docs.docker.com/engine/install/
   ```

2. **Docker Compose** (if using the docker-compose method):
   ```bash
   # Ubuntu/Debian
   sudo apt-get install docker-compose

   # Or install the latest version:
   # https://docs.docker.com/compose/install/
   ```

3. **NVIDIA Container Toolkit** (required for GPU support):
   ```bash
   # Add the NVIDIA GPG key and repository
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

   # Install nvidia-container-toolkit
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit

   # Restart Docker to apply changes
   sudo systemctl restart docker
   ```

   **Note**: Without `nvidia-container-toolkit`, the Docker container will not be able to access your NVIDIA GPU.

### Verify GPU Access

After installation, verify that Docker can access your GPU:

```bash
sudo docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If successful, you should see your GPU information displayed.

## Usage

### Using Docker Compose (Recommended)

```bash
cd docker/docker-cuda/
docker compose up -d
docker compose exec llamafactory bash
```

### Using Docker Run

```bash
# Build the image
docker build -f ./docker/docker-cuda/Dockerfile \
    --build-arg PIP_INDEX=https://pypi.org/simple \
    --build-arg EXTRAS=metrics \
    -t llamafactory:latest .

# Run the container
docker run -dit --ipc=host --gpus=all \
    -p 7860:7860 \
    -p 8000:8000 \
    --name llamafactory \
    llamafactory:latest

# Enter the container
docker exec -it llamafactory bash
```

## Troubleshooting

### GPU Not Detected

If your GPU is not detected inside the container:

1. Ensure `nvidia-container-toolkit` is installed
2. Check that the Docker daemon has been restarted after installation
3. Verify your NVIDIA drivers are properly installed: `nvidia-smi`
4. Check Docker GPU support: `docker run --rm --gpus all ubuntu nvidia-smi`

### Permission Denied

If you get permission errors, ensure your user is in the docker group:

```bash
sudo usermod -aG docker $USER
# Log out and back in for changes to take effect
```

## Megatron Bridge Image

`Dockerfile.megatron` builds a CUDA runtime for LLaMA-Factory + [Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/):

| Component | Version |
| --- | --- |
| Base | `ubuntu:22.04` (Python 3.12) |
| PyTorch | 2.12.1+cu126 (CUDA libs from wheels) |
| TransformerEngine | 2.17.0 |
| megatron-core | 0.18.x (via megatron-bridge) |
| megatron-bridge | 0.5.0 |

### Build

From repo root:

```bash
docker build -f docker/docker-cuda/Dockerfile.megatron \
  -t llamafactory-megatron-bridge:latest .
```

### Run training

```bash
docker run --rm -it --gpus all --ipc=host --shm-size=16g \
  -e DISABLE_VERSION_CHECK=1 \
  -e USE_MEGATRON_BRIDGE=1 \
  -v "$PWD":/app -w /app \
  llamafactory-megatron-bridge:latest
```

## Additional Notes

- The default image is built on Ubuntu 22.04 (x86_64), CUDA 12.4, Python 3.11, PyTorch 2.6.0, and Flash-attn 2.7.4
- For different CUDA versions, you may need to adjust the base image in the Dockerfile
- Make sure your NVIDIA driver version is compatible with the CUDA version used in the Docker image
