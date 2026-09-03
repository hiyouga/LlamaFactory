# 融合算子加速

`kernel_config` 统一配置模型侧的融合算子加速。它可以替换单个算子，也可以像 Liger Kernel 一样同时应用多个融合实现和训练优化。

`kernel_config.name` 接受一个加速实现名称，也接受逗号分隔的多个名称。`auto` 根据当前设备选择默认实现。

## Liger Kernel

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

该方案使用 CUDA/Triton 实现替换受支持模型的 MoE 计算路径。

## Flash Linear Attention

`flash-linear-attention` 通过 FSDPTurbo 的算子注册表替换模型中的 FLA 实现，支持 CUDA 和 NPU：

```yaml
kernel_config:
  name: flash-linear-attention
  include_kernels: chunk_gated_delta_rule,fused_recurrent_gated_delta_rule
  chunk_size: 64
```

`include_kernels` 可以设置为 `auto` 或逗号分隔的算子名称；`chunk_size` 支持 `16`、`32` 和 `64`。使用前需要安装 `requirements/fsdpturbo.txt`。

## 组合多个加速实现

多个实现按配置顺序应用：

```yaml
kernel_config:
  name: first_kernel,second_kernel
```

每个实现会在应用前检查设备、依赖和模型结构。
