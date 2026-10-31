# 融合算子加速

`kernel_config` 统一配置模型侧的融合算子加速。它可以替换单个算子，也可以像 Liger Kernel 一样同时应用多个融合实现和训练优化。

`kernel_config.name` 接受一个加速实现名称，也接受逗号分隔的多个名称。未设置 `kernel_config` 或设为 `null` 时，此入口不替换算子。以下为算子配置片段，完整训练配置见 [SFT](sft.md)。

## 自动配置

`name: auto` 根据当前设备应用默认组合。当前仅 [NPU](../multi-backend/npu/index.md) 配置了默认组合，CUDA 上不会自动启用 Liger 或 CUDA Fused MoE。

## Liger Kernel

Liger Kernel 依赖 `liger-kernel`，并要求模型具有对应适配。

```yaml
kernel_config:
  name: liger_kernel
```

Liger Kernel 根据模型类型调用 `liger_kernel.transformers` 中对应的应用函数，可融合 RMSNorm、RoPE、SwiGLU、Cross Entropy 等训练路径。具体启用项由模型支持范围和 Liger Kernel 版本决定。

## CUDA Fused MoE

```yaml
kernel_config:
  name: cuda_fused_moe
```

该方案依赖 CUDA 和 Triton，并要求模型结构匹配，使用融合实现替换 MoE 计算路径。模型架构不匹配时保留原模型。

## Flash Linear Attention

`flash-linear-attention` 通过 FSDPTurbo 的算子注册表替换模型中已有的 FLA 实现，支持 CUDA 和 NPU。它依赖 FLA 和 FSDPTurbo，不会将普通 attention 模型转换成线性注意力模型。

```yaml
kernel_config:
  name: flash-linear-attention
  include_kernels: chunk_gated_delta_rule,fused_recurrent_gated_delta_rule
  chunk_size: 64
```

`include_kernels` 可以设置为 `auto` 或逗号分隔的算子名称；`chunk_size` 支持 `16`、`32` 和 `64`。使用前需要安装 `requirements/fsdpturbo.txt`。

## 组合多个加速实现

多个名称以逗号分隔。以下为语法示意，`first_kernel` 和 `second_kernel` 是占位符，运行时替换为实际注册的实现名称：

```yaml
kernel_config:
  name: first_kernel,second_kernel
```

多个实现按书写顺序应用，后一个接收前一个处理后的模型；所有实现共享同一份 `kernel_config`。设备与依赖检查在应用前执行，模型匹配由各实现负责。组合入口不自动处理重复替换或顺序冲突。内部调用关系见[开发者指南](../developer-guide/plugins/kernel-acceleration/overview.md)，字段定义见[模型参数](../configuration/model.md#kernel_config)。
