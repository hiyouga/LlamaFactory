# Core（核心模块）

Core 负责连接配置、数据、模型和训练流程。各组件由 Trainer 或 Sampler 组合，不通过插件注册表选择。

| 组件 | 职责 | 源码 |
|------|------|------|
| [DataEngine](data_engine.md) | 加载、转换和索引数据集 | `core/data_engine.py` |
| [ModelEngine](model_engine.md) | 加载 Processor、Renderer 和模型 | `core/model_engine.py` |
| [Renderer](renderer.md) | 将 Messages 转换为模型输入 | `core/rendering/` |
| [BaseTrainer](base_trainer.md) | 管理通用训练生命周期 | `core/base_trainer.py` |
| [BatchGenerator](batch_generator.md) | 生成批次并恢复批次状态 | `core/utils/batching.py` |
| [Callback](callback.md) | 分发训练生命周期事件 | `utils/callbacks/` |

```{toctree}
:maxdepth: 1
:hidden:

data_engine
model_engine
renderer
base_trainer
batch_generator
callback
```
