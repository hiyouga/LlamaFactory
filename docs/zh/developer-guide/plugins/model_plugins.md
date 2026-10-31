# 模型插件

模型插件提供模型加载和处理过程中的可替换操作。初始化、量化、PEFT 和 Kernel 由 ModelEngine 依次调用；Sequence Parallel 需要训练拓扑，因此由 BaseTrainer 在训练初始化阶段调用。

## 每个插件接收和改变什么

| 插件 | 接收 | 返回或修改 |
|------|------|------------|
| InitPlugin | 当前进程的设备与 rank 信息 | 返回用于创建模型的 `torch.device` |
| QuantizationPlugin | 模型加载参数、量化配置与训练标记 | 返回补充量化选项后的加载参数 |
| PeftPlugin | 已构造模型、PEFT 配置与训练标记 | 设置可训练参数、加载或合并 adapter，返回处理后的模型 |
| KernelPlugin | PEFT 处理后的模型与 Kernel 配置 | 替换适配的计算路径，返回处理后的模型 |

它们的接口不同，不能仅通过替换 `name` 在不同插件族之间互换。具体调用次序及原因见 [ModelEngine](../core/model_engine.md)。

## InitPlugin

注册 `init_on_default`、`init_on_meta`、`init_on_rank0`，返回模型创建使用的 `torch.device`。

实现位于 `plugins/model_plugins/initialization.py`。这些函数不加载模型权重；ModelEngine 根据返回设备决定调用 `from_pretrained` 还是在 meta 上使用 `from_config`。Rank 0 加载与其他 rank 的权重同步需要后续分布式路径协作。

## PeftPlugin

注册 `lora` 与 `freeze`。两者分别通过 `LoraParams` 和 `FreezeParams` 严格解析配置。LoRA 还负责 adapter 加载、合并与导出。

实现位于 `plugins/model_plugins/peft.py`。训练时，LoRA 创建或加载 adapter，Freeze 按层和模块选择可训练权重；后续优化器只收集 `requires_grad` 的参数。推理与合并导出也复用 LoRA 插件，因此同一配置入口的行为还取决于调用时的 `is_train`。

## QuantizationPlugin

注册 `auto` 与 `bnb`。插件修改 `from_pretrained` 的 init kwargs，不直接替换已经加载的权重。

实现位于 `plugins/model_plugins/quantization.py`。调用发生在模型创建之前，使量化选项能参与权重加载；这也解释了它与加载完成后执行的 PEFT、Kernel 所处阶段不同。当前可配置字段见[模型参数](../../configuration/model.md#quant_config)。

## KernelPlugin

Kernel 在模型加载和 PEFT 处理后应用。调用流程见[融合算子加速](kernel-acceleration/overview.md)。

## Sequence Parallel Plugins

设置 `TrainingArguments.cp_size > 1` 后，BaseTrainer 使用 `cp_mode` 的值选择 `SequenceParallelModelPlugin`。因此，`cp_mode: ulysses` 会调用 `SequenceParallelModelPlugin("ulysses")` 修改模型 forward 所需的通信；训练循环再调用 `SequenceParallelLossPlugin("sequence_parallel_loss")` 处理 loss 聚合。用户配置见[分布式训练](../../feature-guide/distributed_training.md#ulysses-context-parallel)。

这两个插件共同完成序列并行：模型侧处理 attention 的通信与切分，损失侧处理分布后的输入与监督计算。只替换 forward 而沿用不匹配的损失路径，会破坏这组协作关系。实现位于 `plugins/model_plugins/parallelization/`。

## Chat Template 迁移

旧 `RenderingPlugin` 和 `plugins/model_plugins/templates/` 已删除。Chat
template 统一由 `core/rendering/` 调用 Hugging Face 模板。
