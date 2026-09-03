# BatchGenerator

`core/utils/batching.py` 负责 sampler、DataLoader、collate、梯度累积批次和状态恢复；`trainer_plugins/batching.py` 提供非 normal 策略。

## Normal Batching

固定取 `micro_batch_size` 个样本，Renderer 处理后按当前 batch 最长序列 padding。

## Padding-Free Batching

`BatchingPlugin("padding_free")` 将多个样本拼成无 padding 的序列，并维护能够隔离文档的 attention/position 信息。

## Dynamic Batching

动态策略根据样本 token 数决定每个 batch 包含多少样本。纯 `dynamic_batching` 仍执行 padding；`dynamic_padding_free` 同时拼接序列。

## 保存批次状态

BatchGenerator 保存 sampler/buffer 进度，以便 checkpoint 恢复后继续当前 epoch。恢复路径必须避免重新清空已还原的 buffer。

## 校验动态批处理参数

需要 `max_steps`、不能使用 `save_epochs` 等动态策略约束在 `TrainingArguments.__post_init__` 中校验，而不是只依赖训练循环。
