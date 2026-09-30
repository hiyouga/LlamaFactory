# BaseTrainer

`BaseTrainer` 提供 SFT、DPO 和 RM 共用的训练生命周期。子类主要实现 `compute_loss`，必要时覆盖模型分片或输入处理。

实现位于 `core/base_trainer.py`。任务入口负责加载模型与数据，再把它们传入 Trainer；BaseTrainer 从这些对象开始组织批次生成、优化器更新和训练状态。这样，各训练目标可以共用同一套运行循环。

## 初始化训练组件

先保存参数、模型、Renderer 与数据集，创建 BatchGenerator，再确定训练总步数和 checkpoint 间隔。后续组件的创建顺序取决于后端：

- FSDP2 / FSDPTurbo 先分片模型，再基于处理后的参数创建 optimizer 和 scheduler；没有显式分布式后端时，使用普通模型或 DDP。
- DeepSpeed 先构造后端 engine，再创建 optimizer 和 scheduler，最后由 engine 的 `prepare` 统一准备训练对象。

模型和优化器就绪后，`TrainingCheckpointCoordinator` 才能恢复它们的状态以及批次进度。之后创建 CallbackHandler 和 TrainerState，并把已恢复的 step、epoch 同步给回调状态。CP 模型处理在初始化的最后阶段接入。

## 训练循环

```text
epoch / global step
  → BatchGenerator
  → forward + compute_loss
  → gradient accumulation
  → gradient clipping
  → optimizer.step
  → scheduler.step
  → callback / logging
  → 可选 checkpoint
```

`global_batch_size / (dp_size × micro_batch_size)` 决定梯度累积所需的 micro-batch 数。

BatchGenerator 每次迭代返回一组 micro-batch，对应一次更新步。普通路径对每个 micro-batch 调用 `compute_loss` 并反向传播，累积完成后执行梯度裁剪、参数更新和清零。`compute_loss` 返回标量损失，BaseTrainer 还会根据有效监督 token 数和 DP 规模缩放损失，因此子类不应自行重复执行参数更新。

DeepSpeed 路径把反向传播和同步边界交给后端 engine；SFT 开启 CP 时，循环改为调用 SequenceParallelLossPlugin。扩展损失时需要同时考虑目标任务是否支持这两条路径。

## 训练器实现

- `SFTTrainer`：带 `loss_weights` 的语言模型损失
- `DPOTrainer`：policy/reference 偏好损失
- `RMTrainer`：chosen/rejected reward 排序损失

## 保存模型与 Checkpoint

训练循环在保存 step 调用 `core/utils/checkpoint.py` 中的 TrainingCheckpointCoordinator。它协调模型、优化器、scheduler、批次进度与训练计数等状态；其中依赖分片格式的模型和优化器读写交给 DistributedPlugin。这样，训练器决定何时保存，后端决定如何保存其分布式状态。

`fit()` 结束后，任务入口另外调用 `save_model()` 保存最终模型。最终模型与用于续训的 checkpoint 用途不同，用户配置见[模型保存与恢复](../../feature-guide/model_saving.md)。

## 扩展边界

新增训练目标时，在 Trainer 子类中实现损失和必要的任务准备。替换优化器、批处理或分布式实现时使用[训练器插件](../plugins/trainer_plugins.md)，避免在每个任务的训练循环中复制后端分支。只观察生命周期事件时使用 [Callback](callback.md)。
