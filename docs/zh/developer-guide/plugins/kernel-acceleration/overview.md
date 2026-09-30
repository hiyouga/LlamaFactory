# 融合算子加速

Kernel 系统在模型加载后应用融合算子加速。一个实现既可以替换单个算子，也可以组合多个融合操作或接入外部加速库。

入口位于 `plugins/model_plugins/kernels/interface.py`。它负责名称选择与调用顺序，`base.py` 中的 BaseKernel 负责执行公共检查，各实现的 `_apply` 负责识别模型并进行具体替换。注册机制、检查流程和模型适配由这三层分别承担。

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

接口模块显式导入内置实现，使装饰器在调用前完成注册。`apply_kernels` 将名称按逗号拆分，逐个调用，并将返回模型传给下一个实现；每个实现收到同一份配置与 `require_logits` 等上下文。

`auto` 是入口处的特殊分派：根据设备查询 `_AUTO_KERNELS`，再调用其中的已注册实现。它不是自动搜索最快实现的算法；当前映射只包含 NPU，其他设备不会因 `auto` 而执行替换。

## BaseKernel 与具体实现如何协作

注册表保存的是实现类。`KernelPlugin(name).apply(...)` 将方法调用转发到该类，继承的 `BaseKernel.apply` 再使用 `cls` 调用该实现的检查和替换方法。

`check_device` 检查设备，`check_deps` 检查可选依赖，随后公共入口确认存在模型对象，最后调用 `_apply`。模型是否受支持、需要替换哪个模块，由具体 `_apply` 决定。BaseKernel 不统一检测所有模型结构。

例如，CUDA Fused MoE 按模型架构和模块类名寻找目标并替换 forward；Liger 根据 model type 调用对应外部适配函数；FLA 通过算子注册表匹配模型属性。成功找到注册名称，只说明能进入实现，不能保证模型中存在可替换目标。

## 组合与扩展边界

组合时，后一个实现看到的是已经修改过的模型。入口不提供冲突检测、模型快照或失败回滚，也不会把各实现的专属配置拆成独立配置块。新增实现需要清楚界定自己修改哪些模块，并返回后续调用所需的模型对象。

实现类继承 BaseKernel，提供 `check_device` 和 `_apply`，需要额外依赖时覆盖 `check_deps`；子类定义时会检查必需方法是否实现。名称注册后，还需要确保实现模块在使用前被导入。通用路由与注册规则见[插件注册机制](../../baseplugin_mechanism.md)。

## 已注册实现

- `liger_kernel`
- `cuda_fused_moe`
- `flash-linear-attention`
- `npu_fused_moe`
- `npu_fused_rmsnorm`
- `npu_fused_rope`
- `npu_fused_swiglu`

用户配置见[融合算子加速](../../../feature-guide/kernel_acceleration.md)。
