# 训练器插件

## DistributedPlugin

当前注册 `fsdp2`、`fsdpturbo` 和 `deepspeed`。每个实现类提供统一的模型切分、保存和 checkpoint 方法组：

- `shard_model`
- `save_model`
- `save_checkpoint`
- `load_checkpoint`

参数分别由 `FSDP2Params`、`FSDPTurboParams` 和 `DeepSpeedParams` 解析。FSDPTurbo 额外实现跨专家并行 Mesh 的梯度裁剪。公共 DeviceMesh 拓扑由 `TrainingArguments` 和 `DistributedInterface` 管理。

## BatchingPlugin

`normal` 是 BatchGenerator 默认路径。插件注册：

- `padding_free`
- `dynamic_batching`
- `dynamic_padding_free`

## OptimizerPlugin

当前注册 `muon`。未指定插件时 BaseTrainer 使用默认优化器。Muon 将适合正交化更新的二维权重和其余 AdamW 权重分组。用户配置见[优化器](../../feature-guide/optimizer.md)。

## LRSchedulerPlugin

插件族存在，但当前没有注册可选的 scheduler 名称。
