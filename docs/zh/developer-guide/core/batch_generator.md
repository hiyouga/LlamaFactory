# BatchGenerator

`core/utils/batching.py` 负责 sampler、DataLoader、collate、梯度累积批次和状态恢复；`trainer_plugins/batching.py` 提供非 normal 策略。

## 输入、输出与状态

BatchGenerator 接收 DataEngine、Renderer 和批处理参数。内部的 StatefulDataLoader 按 DP 维度分配样本，并用 `Renderer.process_samples` 把读出的消息转换为 ModelInput 列表。此时还没有完成训练 batch 的组装。

这些 ModelInput 进入 StatefulBuffer，再由选定策略取出、截断并组织为 BatchInput。`__next__()` 返回的是一个更新步所需的 micro-batch 列表，BaseTrainer 逐个计算损失和累积梯度。

```text
DataEngine + StatefulDistributedSampler
  → StatefulDataLoader（Renderer.process_samples）
  → StatefulBuffer
  → normal collate 或 BatchingPlugin.generate_batch
  → list[BatchInput]
```

BatchGenerator 持有 DataLoader、迭代器和缓冲区；策略插件接收缓冲区和 `batch_info`，决定如何填充与取样。读取进度因此不保存在插件注册表中。

## Normal Batching

固定取 `micro_batch_size` 个样本，Renderer 处理后按当前 batch 最长序列 padding。

## Padding-Free Batching

`BatchingPlugin("padding_free")` 将多个样本拼成无 padding 的序列，并维护能够隔离文档的 attention/position 信息。

## Dynamic Batching

动态策略根据样本 token 数决定每个 batch 包含多少样本。纯 `dynamic_batching` 仍执行 padding；`dynamic_padding_free` 同时拼接序列。

## 保存批次状态

`state_dict()` 同时保存 DataLoader 的读取状态与 buffer 中已读取、尚未消费的样本。只恢复前者会丢掉预读样本，只恢复后者则无法确定接下来应该读哪里。

`load_state_dict()` 恢复两部分状态并设置 `_is_resuming`。普通 `__iter__()` 会清空 buffer；恢复后的首次迭代保留 buffer，随后清除恢复标记。新增策略必须保持这个状态约定，不能将剩余样本藏在无法随 checkpoint 保存的全局变量中。

## 校验动态批处理参数

纯 `dynamic_batching` 需要正数 `max_steps` 且不能使用 `save_epochs`，这些条件在 `TrainingArguments.__post_init__` 中校验。多模态输入是否能由当前策略处理，则在 BatchGenerator 生成 batch 时检查。配置条件与使用选择见[批处理策略](../../feature-guide/batching.md)。

## 扩展边界

改变 token 预算或样本拼接方式时实现 BatchingPlugin 的方法组，接口见[训练器插件](../plugins/trainer_plugins.md#batchingplugin)。改变消息如何变成 token 时修改 Renderer，改变反向传播和更新边界时检查 BaseTrainer。
