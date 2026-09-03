# 融合算子加速

Kernel 系统在模型加载后应用融合算子加速。一个实现既可以替换单个算子，也可以组合多个融合操作或接入外部加速库。

## Kernel 应用流程

```text
ModelEngine
  → apply_kernels(model, kernel_config)
  → 解析 kernel_config.name
  → auto 设备选择或 KernelPlugin(name)
  → BaseKernel.apply()
  → check_device()
  → check_deps()
  → _apply()
```

名称可以用逗号分隔，按顺序应用。`auto` 当前在 NPU 上应用一组内置融合算子，在其他设备上可能不做任何替换。

## 已注册实现

- `liger_kernel`
- `cuda_fused_moe`
- `flash-linear-attention`
- `npu_fused_moe`
- `npu_fused_rmsnorm`
- `npu_fused_rope`
- `npu_fused_swiglu`

用户配置见[融合算子加速](../../../feature-guide/kernel_acceleration.md)。
