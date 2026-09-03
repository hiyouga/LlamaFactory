# Ascend NPU

本页介绍 Ascend NPU 的手动安装、功能范围和融合算子加速能力。

## 手动安装

手动安装需要依次准备 HDK 驱动与固件、CANN 和 `torch-npu`。安装包需要与 NPU 型号、CPU 架构和操作系统匹配。本分支使用以下软件组合：

| 组件 | 版本 |
|------|------|
| CANN | `9.1.0` |
| PyTorch | `2.10.0` |
| torch-npu | `2.10.0.post2` |
| torchvision | `0.25.0` |
| torchaudio | `2.10.0` |
| Triton Ascend | `3.2.1` |

安装前使用[Ascend 兼容性查询助手](https://www.hiascend.com/hardware/compatibility)确认硬件与操作系统组合，并从[CANN 9.1.0 社区版资源中心](https://www.hiascend.com/developer/download/community/result?cann=9.1.0&module=cann)选择对应 CPU 架构的软件包。

### 安装驱动与固件

从 Ascend 下载与设备匹配的 HDK 驱动和固件安装包。以下命令中的文件名需要替换为实际下载的包名：

```bash
chmod +x Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run
chmod +x Ascend-hdk-<chip_type>-npu-firmware_<version>.run

sudo ./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run \
  --full --install-for-all
sudo ./Ascend-hdk-<chip_type>-npu-firmware_<version>.run --full
```

根据安装程序提示重启系统，然后验证驱动：

```bash
npu-smi info
```

### 安装 CANN

先安装 Toolkit，再安装与设备匹配的 ops 算子包：

```bash
chmod +x Ascend-cann-toolkit_<version>_linux-<arch>.run
sudo ./Ascend-cann-toolkit_<version>_linux-<arch>.run --install

chmod +x Ascend-cann-<chip_type>-ops_<version>_linux-<arch>.run
sudo ./Ascend-cann-<chip_type>-ops_<version>_linux-<arch>.run --install
```

默认以 root 安装时，加载以下环境变量：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

普通用户安装时，环境脚本位于用户选择的 CANN 安装目录。需要在每个运行 LlamaFactory 的 shell 中加载该脚本，也可以将命令加入 shell 配置文件。

### 安装 LlamaFactory

在项目根目录创建 Python 环境，并安装 NPU 依赖和 LlamaFactory：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements/npu.txt
python -m pip install -r requirements/triton_ascend.txt
```

`requirements/npu.txt` 固定了相互匹配的 PyTorch、torchvision、torchaudio 和 `torch-npu` 版本；`requirements/triton_ascend.txt` 安装 NPU 对应的 Triton 实现。如果使用已有 Python 环境，应确认安装完成后的 PyTorch 与 `torch-npu` 版本仍与上述版本一致。

### 验证 PyTorch NPU

```bash
python -c "import torch, torch_npu; print(torch.npu.is_available())"
```

输出 `True` 表示 PyTorch 已识别 NPU。启用 v1 后即可运行 SFT：

```bash
export USE_V1=1
llamafactory-cli sft examples/v1/train_full/train_full_fsdp2.yaml
```

## 功能范围

| 功能 | 状态 | 说明 |
|------|:----:|------|
| [SFT 全参训练](../../feature-guide/sft.md) | 支持 | 支持 FSDP2 |
| [LoRA / Freeze](../../feature-guide/sft.md) | 支持 | 使用通用 PEFT 路径 |
| [QLoRA](../../feature-guide/sft.md#qlora) | 不支持 | 当前量化路径依赖 bitsandbytes |
| [DPO](../../feature-guide/dpo.md) | 支持 | 使用偏好对数据 |
| [RM](../../feature-guide/rm.md) | 支持 | `cp_size` 需要为 `1` |
| [FSDP2](../../feature-guide/distributed_training.md) | 支持 | 使用 NPU 设备和通信后端 |
| [FSDPTurbo](../../feature-guide/distributed_training.md) | 支持 | 提供 MoE 专家并行和专家参数分片 |
| [Ulysses CP](../../feature-guide/distributed_training.md#ulysses-context-parallel) | 支持 | 依赖适配的 attention 实现 |
| [DeepSpeed](../../feature-guide/distributed_training.md#deepspeed) | 依赖环境 | 由 NPU DeepSpeed 发行版和配置决定 |
| [HF CLI 推理](../../feature-guide/inference.md) | 支持 | `sample_backend: hf` |

## 量化支持

当前 v1 QLoRA 依赖 bitsandbytes，因此不适用于 NPU。

## 融合算子加速

`kernel_config.name: auto` 在 NPU 上依次尝试：

- `npu_fused_moe`
- `npu_fused_rmsnorm`
- `npu_fused_rope`
- `npu_fused_swiglu`

也可以显式选择一个或多个名称：

```yaml
kernel_config:
  name: npu_fused_rmsnorm,npu_fused_rope
```

每个实现会检查当前设备及 `torch_npu` 依赖。模型结构不匹配时，具体 Kernel 可能跳过替换或抛出明确错误。

## Liger Kernel

`liger_kernel` 接受 CUDA 或 NPU，但仍要求模型类型在当前 Liger 映射中，并且已安装兼容版本的 `liger_kernel`。

## 分布式训练

FSDP2、FSDPTurbo 与 Ulysses 使用 accelerator 抽象选择 NPU 通信设备。DeepSpeed 的可用性由 NPU DeepSpeed 发行版、依赖版本和配置共同决定。
