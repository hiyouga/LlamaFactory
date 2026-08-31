# 面向昇腾 NPU 的 LlamaFactory 镜像

LlamaFactory 昇腾 NPU 镜像面向华为昇腾 Atlas NPU，提供可直接使用的 LlamaFactory 环境。镜像基于昇腾 CANN 容器镜像构建，预装 Python、PyTorch、TorchNPU、DeepSpeed、LlamaFactory 等组件。

安装方法和问题排查请参考 [LlamaFactory NPU 安装及配置文档](https://llamafactory.readthedocs.io/zh-cn/latest/multibackend/npu/npu_installation.html)。

## 快速参考

- 镜像仓库：
  - `docker.io/hiyouga/llamafactory`
  - `quay.io/ascend/llamafactory`
- Dockerfile：`docker/docker-npu/Dockerfile`
- Docker Compose 文件：`docker/docker-npu/docker-compose.yml`

当前提供以下 `latest` NPU 镜像 tag：

| 硬件系列 | 操作系统 | Tag |
| --- | --- | --- |
| A2 | Ubuntu 22.04 | `latest-910b-ubuntu` |
| A3 | Ubuntu 22.04 | `latest-a3-ubuntu` |
| A2 | openEuler 24.03 | `latest-910b-openeuler` |
| A3 | openEuler 24.03 | `latest-a3-openeuler` |

历史版本的 NPU 镜像 Tag 及其对应的 Dockerfile 归档链接，请参阅 [Supported Tags](./supported_tags.md)。

## 镜像介绍

镜像内预装以下主要组件：

| 组件 | 版本 |
| --- | --- |
| CANN | `9.1.0` |
| Python | `3.12` |
| PyTorch | `2.10.0` |
| TorchNPU | `2.10.0.post2` |
| torchvision / torchaudio | `0.25.0` / `2.10.0` |
| Transformers | 构建时的最新兼容版本 |
| Triton Ascend | `3.2.1` |
| DeepSpeed | 构建时的最新兼容版本 |
| LlamaFactory | 从构建上下文中的仓库源码安装 |

镜像不包含模型权重和数据集。请通过目录挂载或运行时下载的方式单独提供，并遵守对应的许可证和使用要求。

## 镜像 Tag 说明

NPU 镜像的 `latest` 和 release tag 使用不同格式；以下规则不适用于 CUDA 镜像。

非 release 构建复用以下简短 tag，每次定时构建会更新对应 tag 所指向的镜像：

```text
latest-<芯片信息>-<操作系统>
```

| 字段 | 可选值 | 说明 |
| --- | --- | --- |
| `芯片信息` | `910b` 或 `a3` | 镜像所适配的昇腾芯片型号 |
| `操作系统` | `ubuntu` 或 `openeuler` | 容器操作系统类型 |

后续发布的 NPU release 镜像将使用以下完整 tag 格式。历史 release 镜像使用旧的命名格式，不适用以下规则。

```text
<LlamaFactory版本>-cann<CANN版本>-torch_npu<TorchNPU版本>-<芯片信息>-<操作系统>-<Python版本>
```

| 字段 | 示例 | 说明 |
| --- | --- | --- |
| `LlamaFactory版本` | `0.9.5` | LlamaFactory release 版本号 |
| `CANN版本` | `9.1.0` | 从 CANN 基础镜像 tag 中提取 |
| `TorchNPU版本` | `2.10.0.post2` | 镜像使用的 TorchNPU 完整版本，包含 `.postN` 等后缀 |
| `芯片信息` | `910b` 或 `a3` | 镜像所适配的昇腾芯片型号 |
| `操作系统` | `ubuntu22.04` 或 `openeuler24.03` | 容器操作系统类型和版本 |
| `Python版本` | `py3.12` | 从 CANN 基础镜像 tag 中提取 |

例如：

```text
0.9.5-cann9.1.0-torch_npu2.10.0.post2-a3-ubuntu22.04-py3.12
```

## 快速开始

### 前置条件

启动容器前需要：

1. 在宿主机安装与镜像内 CANN 版本兼容的昇腾驱动和固件。
2. 确认宿主机执行 `npu-smi info` 可以正常识别 NPU。
3. 安装 Docker，并确保当前用户有权访问所需的昇腾设备节点和驱动文件。

驱动、固件、CANN、TorchNPU 与目标昇腾硬件需要保持兼容。

### 拉取并运行镜像

以下示例使用一张 NPU 启动最新的 A2 Ubuntu 镜像。请根据实际情况修改 ``DOCKER_IMAGE`` 和 ``device``。

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

部分驱动环境中的 `npu-smi` 位于 `/usr/local/sbin/npu-smi`，此时需要调整挂载源路径。使用多张 NPU 时，继续追加 `--device=/dev/davinci<N>` 参数。

进入容器后验证运行环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available())"
llamafactory-cli help
```

### 本地构建镜像

在仓库根目录执行构建。以下示例构建 A2 Ubuntu 镜像：

```bash
docker build \
  -f ./docker/docker-npu/Dockerfile \
  --build-arg BASE_IMAGE=quay.io/ascend/cann:9.1.0-910b-ubuntu22.04-py3.12 \
  --build-arg PIP_INDEX=https://pypi.org/simple \
  -t llamafactory:npu-910b-ubuntu \
  .
```

可用构建参数：

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `BASE_IMAGE` | `quay.io/ascend/cann:9.1.0-910b-ubuntu22.04-py3.12` | 根据设备型号和容器操作系统选择对应的基础镜像 |
| `PIP_INDEX` | `https://pypi.org/simple` | 指定 Python 软件包索引 |
| `PYTORCH_INDEX` | `https://download.pytorch.org/whl/cpu` | 指定配合 TorchNPU 使用的 PyTorch wheel 索引 |
| `HTTP_PROXY` | 空 | 构建期间可选的 HTTP/HTTPS 代理 |

### 通过 Docker Compose 启动

前面的 `docker build` 命令直接调用 Dockerfile，只构建镜像，不启动容器。Docker Compose 不使用另一套构建逻辑：它读取 `docker-compose.yml` 中的预设配置，复用同一个 Dockerfile，并通过 profile 选择硬件系列和操作系统组合。下面的 `up -d` 会在后台启动容器；若本地镜像不存在，Docker Compose 会先构建镜像：

```bash
cd docker/docker-npu

# A2 + Ubuntu
docker compose --profile a2-ubuntu up -d

# A3 + Ubuntu
docker compose --profile a3-ubuntu up -d

# A2 + openEuler
docker compose --profile a2-openeuler up -d

# A3 + openEuler
docker compose --profile a3-openeuler up -d
```

如果只想通过 Docker Compose 构建镜像而不启动容器，请使用 `docker compose --profile <profile> build`。

## 硬件支持与兼容性说明

- A2 镜像使用标记为 `910b` 的 CANN 基础镜像，A3 镜像使用标记为 `a3` 的 CANN 基础镜像。
- 镜像构建目标同时包含 x86-64（`linux/amd64`）和 AArch64（`linux/arm64`）宿主机。CPU 架构与硬件系列是 A2 还是 A3 无关。
- Ubuntu 22.04 和 openEuler 24.03 指容器内部的操作系统。
- 旧式 NPU tag 已由 `latest-<910b|a3>-<ubuntu|openeuler>` 格式取代。
- 正式部署前，请验证具体驱动、固件、CANN 和 SoC 组合的兼容性。

## 许可证与免责声明

LlamaFactory 基于 [Apache License 2.0](../../LICENSE) 发布。

昇腾 CANN、TorchNPU、Triton Ascend、DeepSpeed、基础操作系统软件包、模型权重、数据集和其他第三方组件分别受其自身许可证与条款约束。LlamaFactory 的许可证不会替代或覆盖这些条款。

本镜像按“原样”提供，不附带任何明示或暗示的保证。用户需要自行验证软硬件兼容性、保障容器及运行配置的安全、遵守适用的许可证和法律，并在训练、评测或部署前审查模型与数据集的使用条款。
