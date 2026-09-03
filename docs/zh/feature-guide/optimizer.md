# 优化器

未设置 `optim_config` 时，v1 使用 AdamW，并从顶层 `learning_rate` 读取学习率。

## Muon

设置 `optim_config.name: muon` 启用 Muon。Muon 对适合正交化更新的二维权重使用 Muon，并将偏置、归一化参数、embedding、输出层和 LoRA 参数交给内部 AdamW。

```yaml
model: Qwen/Qwen3-0.6B
model_class: llm
train_dataset: data/v1_sft_demo.yaml

dist_config:
  name: fsdp2

optim_config:
  name: muon
  wd: 0.1
  momentum: 0.95
  nesterov: true
  ns_steps: 5
  adamw_betas: [0.9, 0.95]
  adamw_eps: 1.0e-8

output_dir: outputs/qwen3_muon
micro_batch_size: 1
cutoff_len: 2048
learning_rate: 1.0e-5
max_steps: 10
```

```bash
llamafactory-cli sft config.yaml
```

学习率统一由顶层 `learning_rate` 控制，不在 `optim_config` 中重复设置。仓库示例见 `examples/v1/train_full/train_full_muon.yaml`，完整字段见[训练参数](../configuration/training.md#optim_config)。
