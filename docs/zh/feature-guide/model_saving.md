# 模型保存与恢复

训练结束时 `save_model()` 保存最终模型；训练过程中可以按 step 或 epoch 保存 checkpoint。

## 保存训练 Checkpoint

```yaml
save_steps: 500
save_epochs: null
save_total_limit: 3
save_ckpt_as_hf: false
```

`save_steps` 与 `save_epochs` 控制触发时机。`save_total_limit` 删除最旧的完整 checkpoint。

## 从 Checkpoint 恢复训练

```yaml
resume_from_checkpoint: auto
```

`auto` 在 `output_dir` 下寻找最新的完整 checkpoint，也可以直接填写 checkpoint 路径。未设置 `resume_from_checkpoint` 时，即使 `output_dir` 中存在 checkpoint，也不会触发续训。

## 不同后端的保存格式

| 训练后端 | 默认 checkpoint | `save_ckpt_as_hf: true` |
|----------|-------------------|--------------------------|
| FSDP2 / FSDPTurbo | 保存用于续训的分布式 checkpoint | 额外在 checkpoint 中生成 `hf_model` 目录 |
| DeepSpeed | 保存 DeepSpeed 训练状态 | 额外在 checkpoint 中生成 `hf_model` 目录 |
| 单设备 / DDP | 模型权重使用 HF 格式保存 | 不生成额外的 `hf_model` 目录 |

FSDP2、FSDPTurbo 和 DeepSpeed 启用 `save_ckpt_as_hf` 后，仍会保留用于恢复训练的原始 checkpoint，同时额外保存 HF 格式模型。聚合完整模型权重会提高保存时的内存占用。

除非需要在训练过程中直接获得 HF 格式的中间模型，否则保持 `save_ckpt_as_hf: false`，可以避免每次保存 checkpoint 时额外聚合完整模型权重。

## 初始化权重与恢复训练

| 配置 | 用途 | 加载时机 | 恢复内容 |
|------|------|----------|----------|
| `dist_config.dcp_path` | 使用 DCP 权重初始化模型 | FSDP2/FSDPTurbo 模型分片阶段 | 仅模型权重 |
| `resume_from_checkpoint` | 从训练 checkpoint 继续训练 | Trainer 初始化阶段 | 模型、优化器、学习率调度器、批次进度、训练步数及可用的随机数状态 |

例如，只加载已有 DCP 模型权重并开始一次新训练时使用：

```yaml
dist_config:
  name: fsdp2
  dcp_path: path/to/dcp_model
```

需要从 `output_dir` 中最新的完整 checkpoint 继续原训练时使用：

```yaml
resume_from_checkpoint: auto
```
