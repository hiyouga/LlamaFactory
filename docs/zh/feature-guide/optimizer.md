# 优化器

`optim_config` 未设置或为 `null` 时，v1 使用 AdamW，并从顶层 `learning_rate` 读取学习率。

## Muon

设置 `optim_config.name: muon` 启用 Muon。Muon 对适合正交化更新的二维权重使用 Muon，并将偏置、归一化参数、embedding、输出层和 LoRA 参数交给内部 AdamW。

以下为完整的 SFT 配置（`config.yaml`）：

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

学习率统一由顶层 `learning_rate` 控制。仓库示例见 `examples/v1/train_full/train_full_muon.yaml`，完整字段见[训练参数](../configuration/training.md#optim_config)。

### 批量正交化

单卡训练可以在 `optim_config` 中设置 `ns_batch_size: 4`，将同一参数组中形状、设备和 dtype 相同的二维矩阵合批执行 Newton–Schulz 迭代。每个矩阵独立归一化，momentum 状态仍按参数保存；每批最多包含指定数量的矩阵。默认值为 `1`，保留串行路径。DTensor 参数仍走原有分布式路径。

批量计算可以减少 kernel launch，但需要更多临时显存。BF16 批量矩阵乘法与逐矩阵计算存在舍入差异，不保证参数轨迹逐位相同；应对比实际模型的训练曲线与完整 step 时间。该选项只作用于 Muon 参数，纯 LoRA 参数仍使用 AdamW。优化器 checkpoint 保存批量设置；恢复旧 checkpoint 时，缺失该字段的参数组使用串行路径。
