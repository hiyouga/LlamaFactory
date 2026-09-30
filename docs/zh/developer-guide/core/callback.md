# Callback

Callback 系统位于 `utils/callbacks/`，用于把日志和生命周期通知从训练循环中分离。

## Callback 组件

- `TrainerCallback`：事件接口
- `CallbackHandler`：按注册顺序广播事件
- `LoggingCallback`：输出 loss、learning rate、grad norm 等指标

## 事件如何传递

BaseTrainer 创建 CallbackHandler，并先加入默认 LoggingCallback，再加入构造参数 `callbacks` 中的实例。训练循环在固定位置调用 handler；handler 按列表顺序同步调用每个实例的同名方法，不经过 BasePlugin 注册表。

```text
on_train_begin
  → on_epoch_begin
    → on_step_begin
    → 前向、反向与更新
    → on_step_end
    → on_log（到达 logging_steps 时）
    → on_save（checkpoint 写入后）
  → on_epoch_end
on_train_end
```

事件位置由 Trainer 和 checkpoint 协调器决定。Callback 无需自行判断什么时候保存模型或推进优化器，只响应已经发生的生命周期事件。

## Callback 能读取什么

每个事件接收训练参数 `args`、进度对象 `state` 和额外关键字参数。TrainerState 保存 step、epoch、最近一次损失、梯度范数和学习率等；Handler 还通过关键字参数传入当前 model、optimizer、lr_scheduler 和批次生成器。

在 `on_step_begin` 时，本步计算尚未发生，损失等指标仍来自此前的更新；本步指标在 `on_step_end` 前写入 state。`on_log` 额外接收本次日志字典。

## 扩展边界

自定义 Callback 继承 TrainerCallback，仅覆盖关心的事件，并将实例传给 Trainer 的 `callbacks` 参数。它适合记录指标、连接外部观察工具等附加行为。需要改变损失、更新规则或 checkpoint 格式时，使用 Trainer 或对应插件。

接口约定将 state 作为供观察的进度信息，不通过修改它来控制训练流程；代码没有把这个 dataclass 冻结。事件也不会统一限制到 rank 0，涉及单次输出的 Callback 需要自行判断 rank，默认 LoggingCallback 就在自身内部处理这个条件。
