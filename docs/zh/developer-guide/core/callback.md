# Callback

Callback 系统位于 `utils/callbacks/`，用于把日志和生命周期通知从训练循环中分离。

## Callback 组件

- `TrainerCallback`：事件接口
- `CallbackHandler`：按注册顺序广播事件
- `LoggingCallback`：输出 loss、learning rate、grad norm 等指标

BaseTrainer 在训练开始、step、日志、保存和训练结束等阶段调用 handler。Callback 通过这些生命周期事件扩展行为，optimizer 和 checkpoint 流程仍由 BaseTrainer 管理。
