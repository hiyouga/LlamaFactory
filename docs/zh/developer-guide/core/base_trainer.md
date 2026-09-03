# BaseTrainer

`BaseTrainer` 提供 SFT、DPO 和 RM 共用的训练生命周期。子类主要实现 `compute_loss`，必要时覆盖模型分片或输入处理。

## 初始化训练组件

- 保存训练参数、模型、Renderer 和数据集
- 根据分布式拓扑推导 micro-batch 数
- 创建 BatchGenerator
- 应用分布式后端或 DDP
- 创建 optimizer 和 scheduler
- 注册 callback
- 恢复 checkpoint

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

## 训练器实现

- `SFTTrainer`：带 `loss_weights` 的语言模型损失
- `DPOTrainer`：policy/reference 偏好损失
- `RMTrainer`：chosen/rejected reward 排序损失

## 保存模型与 Checkpoint

分布式模型、checkpoint 的保存和恢复通过 `DistributedPlugin` 对应方法完成；无显式后端时走普通模型/DDP 路径。
