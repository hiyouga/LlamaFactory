# 功能指南

功能指南面向使用 v1 完成训练与推理任务的用户。配置结构、默认值和可用选项统一放在[参数配置](../configuration/index.md)。

## 训练任务

| 页面 | 内容 |
|------|------|
| [数据准备](data_preparation.md) | Messages 格式、数据集 YAML 和 converter |
| [SFT](sft.md) | 全参、LoRA、Freeze、QLoRA |
| [DPO](dpo.md) | DPO、ORPO、SimPO 与偏好数据 |
| [RM](rm.md) | 奖励模型训练 |

## 训练效率与扩展

| 页面 | 内容 |
|------|------|
| [批处理](batching.md) | 四种 batching strategy |
| [分布式训练](distributed_training.md) | FSDP2、FSDPTurbo、DeepSpeed、Ulysses |
| [优化器](optimizer.md) | AdamW 与 Muon 配置 |
| [融合算子加速](kernel_acceleration.md) | Liger、融合算子和组合配置 |

## 模型保存与使用

| 页面 | 内容 |
|------|------|
| [模型保存与恢复](model_saving.md) | 最终模型、checkpoint、断点续训 |
| [模型导出](model_export.md) | LoRA 合并和 HF 格式导出 |
| [推理](inference.md) | CLI 对话与 adapter 加载 |
