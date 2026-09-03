# 参数说明

v1 使用一份 YAML 同时描述数据、模型、训练和推理配置。顶层字段来自四个参数类。`peft_config`、`quant_config`、`kernel_config`、`dist_config` 等插件配置通过 `name` 选择具体实现，其余字段由对应实现解析；可用名称和字段统一列在所属的参数页面中。

```yaml
model: Qwen/Qwen3-0.6B
train_dataset: data/v1_sft_demo.yaml

output_dir: outputs/qwen3_sft
micro_batch_size: 1
learning_rate: 1.0e-4
max_steps: 10

dist_config:
  name: fsdp2

kernel_config:
  name: liger_kernel
```

## 参数分类

| 页面 | 参数 |
|------|------|
| [数据参数](data.md) | `DataArguments`、`DatasetInfo` |
| [模型参数](model.md) | `ModelArguments`、PEFT、量化、初始化和 Kernel |
| [训练参数](training.md) | `TrainingArguments`、分布式和优化器 |
| [推理参数](inference.md) | `SampleArguments` |

功能指南中的示例只列出完成对应任务所需的字段。字段类型、默认值和可用选项以本目录为准。
