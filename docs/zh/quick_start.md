# 快速开始

v1 是 LlamaFactory 的实验性架构，通过 `USE_V1=1` 启用。稳定版（v0）的功能与完整文档见[LlamaFactory 文档](https://llamafactory.readthedocs.io/)。

本页是 v1 的快速指南，介绍安装、启用 v1 和运行 SFT。更完整的环境说明见仓库 README。

## 安装

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e .
```

## 启用 v1

```bash
export USE_V1=1
```

## 运行 SFT

仓库提供数据配置 `data/v1_sft_demo.yaml` 和可运行示例。以下训练配置已经通过 `train_dataset` 引用该数据配置，可以直接运行：

```bash
llamafactory-cli sft examples/v1/train_full/train_full_fsdp2.yaml
```

检测到多个设备时，CLI 会自动通过 `torchrun` 重启；单设备需要强制走该入口时可以设置 `FORCE_TORCHRUN=1`。

## 选择运行命令

v1 当前支持以下命令：

| 命令 | 用途 |
|------|------|
| `sft` | 监督微调 |
| `dpo` | 偏好优化 |
| `rm` | 奖励模型训练 |
| `chat` | 交互式推理 |
| `merge` | 合并 LoRA adapter 并导出模型 |

## 文档导航

- [功能指南](feature-guide/index.md)：训练、推理、分布式训练、模型保存与融合算子加速
- [参数说明](configuration/index.md)：数据、模型、训练和推理参数
- [开发者指南](developer-guide/index.md)：v1 架构、Core 与 Plugin
- [多后端](multi-backend/npu/index.md)：昇腾 NPU 的环境安装与功能说明
