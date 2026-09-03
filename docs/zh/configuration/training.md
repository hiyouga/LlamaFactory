# 训练参数

## TrainingArguments

### 训练过程与精度

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `output_dir` | `str` | `outputs/<uuid>` | 输出目录 |
| `micro_batch_size` | `int` | `1` | 基础 micro-batch 大小 |
| `global_batch_size` | `int \| None` | `None` | 默认 `dp_size × micro_batch_size` |
| `cutoff_len` | `int` | `2048` | 最大序列长度 |
| `learning_rate` | `float` | `1e-4` | 训练及优化器插件使用的学习率 |
| `num_train_epochs` | `int` | `3` | 训练轮数 |
| `max_steps` | `int \| None` | `None` | 设置后使用 step 作为终止条件 |
| `max_grad_norm` | `float` | `1.0` | 梯度裁剪阈值 |
| `bf16` | `bool` | `true` | 是否使用 bf16 |
| `seed` | `int` | `42` | 随机种子 |
| `full_determinism` | `bool` | `false` | 是否启用完整确定性模式 |

### 批处理配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `batching_strategy` | `str` | `normal` | `normal`、`padding_free`、`dynamic_batching` 或 `dynamic_padding_free` |
| `batching_workers` | `int` | `16` | 数据加载 worker 数 |
| `enable_activation_checkpointing` | `bool` | `true` | 是否启用激活值重算 |

各策略的 token 预算和使用约束见[批处理策略](../feature-guide/batching.md)。

### 分布式与优化器

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dist_config` | `dict \| None` | `None` | FSDP2 或 DeepSpeed 配置 |
| `dp_size` | `int \| None` | `None` | 默认由 world size 和 `cp_size` 推导 |
| `cp_size` | `int` | `1` | Context Parallel 大小 |
| `cp_mode` | `str` | `ulysses` | Context Parallel 实现 |
| `mp_replicate_size` | `int` | `1` | FSDP 二维 Mesh 的参数复制维度大小 |
| `mp_shard_size` | `int \| None` | `None` | FSDP 二维 Mesh 的参数分片维度大小；默认由 world size 推导 |
| `dist_timeout` | `int` | `18000` | 进程组初始化超时，秒 |
| `optim_config` | `dict \| None` | `None` | 优化器插件 |
| `lr_scheduler_config` | `dict \| None` | `None` | 学习率调度插件；当前没有内置实现 |

### Checkpoint 与日志配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `resume_from_checkpoint` | `str \| None` | `None` | checkpoint 路径或 `auto` |
| `save_steps` | `int \| None` | `None` | 每 N 个全局 step 保存 |
| `save_epochs` | `float \| None` | `None` | 每 N 个 epoch 保存 |
| `save_ckpt_as_hf` | `bool` | `false` | 是否在中间 checkpoint 中额外保存 HF 格式模型 |
| `save_total_limit` | `int \| None` | `None` | 最多保留数量 |
| `logging_steps` | `int` | `1` | 日志间隔 |

### 偏好优化参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `pref_loss` | `str` | `sigmoid` | `sigmoid`、`orpo` 或 `simpo` |
| `pref_beta` | `float` | `0.1` | DPO beta |
| `pref_ftx` | `float` | `0.0` | SFT 损失系数 |
| `simpo_gamma` | `float` | `0.5` | SimPO reward margin |
| `dpo_label_smoothing` | `float` | `0.0` | cDPO label smoothing |
| `ld_alpha` | `float \| None` | `None` | LD-DPO 长度差异权重 |

## dist_config

### FSDP2

设置 `name: fsdp2`：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `reshard_after_forward` | `bool` | `true` | forward 后是否重新分片 |
| `offload_params` | `bool` | `false` | 是否 offload 参数 |
| `pin_memory` | `bool` | `true` | 是否使用 pinned memory |
| `dcp_path` | `str \| None` | `None` | 初始化 DCP 权重路径 |

### DeepSpeed

设置 `name: deepspeed`：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `config_file` | `str` | 必填 | DeepSpeed JSON 配置 |

### FSDPTurbo

设置 `name: fsdpturbo`：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `reshard_after_forward` | `bool` | `true` | forward 后是否重新分片 |
| `offload_params` | `bool` | `false` | 是否 offload 参数 |
| `pin_memory` | `bool` | `true` | 是否使用 pinned memory |
| `dcp_path` | `str \| None` | `None` | 初始化 DCP 权重路径 |
| `ep_size` | `int` | `1` | 专家并行组大小 |
| `ep_dispatcher` | `str` | `eager` | 专家 token dispatcher |
| `fsdp_ignored_modules` | `list[str]` | `[]` | 外层 FSDP2 忽略的额外模块 |
| `hook_modules` | `list[str]` | `[]` | EFSDP hook 的模块模式 |
| `fsdp_implementation` | `str` | `native` | `native` 或 `custom` |

`dp_size`、`cp_size`、`mp_replicate_size` 和 `mp_shard_size` 是 `TrainingArguments` 字段，不放在 `dist_config` 中。

## optim_config

设置 `name: muon`：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `wd` | `float` | `0.1` | weight decay |
| `momentum` | `float` | `0.95` | Muon momentum |
| `nesterov` | `bool` | `true` | 是否启用 Nesterov |
| `ns_steps` | `int` | `5` | Newton-Schulz 步数 |
| `adamw_betas` | `list[float]` | `[0.9, 0.95]` | 内部 AdamW betas |
| `adamw_eps` | `float` | `1e-8` | 内部 AdamW epsilon |

学习率统一使用顶层 `learning_rate`。

Muon 的完整配置示例见[优化器](../feature-guide/optimizer.md)。
