# 训练器插件

训练器插件替换训练过程中的具体操作。BaseTrainer 持有模型、优化器和训练计数，BatchGenerator 持有读取进度与缓冲区；插件通过调用参数访问这些对象。替换实现时需要维持调用方依赖的输入、输出和状态约定。

## DistributedPlugin

当前注册 `fsdp2`、`fsdpturbo` 和 `deepspeed`。每个实现类提供统一的模型切分、保存和 checkpoint 方法组：

- `shard_model`
- `save_model`
- `save_checkpoint`
- `load_checkpoint`

参数分别由 `FSDP2Params`、`FSDPTurboParams` 和 `DeepSpeedParams` 解析。FSDPTurbo 额外实现跨专家并行 Mesh 的梯度裁剪。公共 DeviceMesh 拓扑由 `TrainingArguments` 和 `DistributedInterface` 管理。

注册和入口位于 `plugins/trainer_plugins/distributed/interface.py`，具体引擎位于同目录下的 `fsdp2.py`、`fsdpturbo.py` 和 `deepspeed.py`。路由层把配置转换为对应 Params，再委托后端执行；参数配置本身不负责创建进程组。

FSDP2 / FSDPTurbo 的 `shard_model` 返回处理后的模型，BaseTrainer 再创建优化器。DeepSpeed 的该入口返回后端 engine，BaseTrainer 随后调用它的 `prepare`，共同准备模型、优化器和 scheduler。因此，公共入口背后的返回对象与初始化协议仍需结合 BaseTrainer 的后端分支理解。

保存时，Trainer 和 checkpoint 协调器决定时机与通用训练状态，分布式插件负责其模型和优化器状态格式。扩展新后端需要同时提供分片、最终模型保存、checkpoint 保存与恢复；仅实现分片无法覆盖完整训练生命周期。

## BatchingPlugin

`normal` 是 BatchGenerator 默认路径。插件注册：

- `padding_free`
- `dynamic_batching`
- `dynamic_padding_free`

接口与实现位于 `plugins/trainer_plugins/batching.py`。实现类继承 BaseBatcher，提供四个操作：

| 方法 | BatchGenerator 用它决定什么 |
|------|-----------------------------|
| `get_data_provider_batch_size` | 底层 DataLoader 每次读多少条样本 |
| `compute_length` | 批次生成器报告的长度 |
| `fill_buffer` | 何时继续读取样本、怎样填充缓冲区 |
| `generate_batch` | 如何取出样本并组织一个更新步的 micro-batch 列表 |

方法接收 `batch_info`、buffer 或读取函数，状态由调用方管理。`generate_batch` 返回 `None` 时，BatchGenerator 结束本次迭代。新策略需要同时考虑数据耗尽、剩余 buffer 和 checkpoint 恢复，不能只实现拼接张量。

## OptimizerPlugin

当前注册 `muon`。未指定插件时 BaseTrainer 使用默认优化器。Muon 将适合正交化更新的二维权重和其余 AdamW 权重分组。用户配置见[优化器](../../feature-guide/optimizer.md)。

入口位于 `plugins/trainer_plugins/optimizers/optimizer.py`，接收处理后的模型与 `optim_config`，返回 optimizer 实例。当前 Muon 直接读取配置字段；顶层 `learning_rate` 在 TrainingArguments 初始化时写入配置的 `lr`。插件负责参数分组与优化器构造，反向传播和更新时机仍由 Trainer 或 DeepSpeed engine 控制。

## LRSchedulerPlugin

插件族存在，但当前没有注册可选的 scheduler 名称。

接口位于 `plugins/trainer_plugins/lr_scheduler.py`。BaseTrainer 未收到配置时使用固定倍率的 LambdaLR；收到配置时按名称调用插件，传入 optimizer、训练总步数与配置，并保存返回的 scheduler。扩展实现应返回兼容训练循环和 checkpoint 状态保存的 scheduler 对象。
