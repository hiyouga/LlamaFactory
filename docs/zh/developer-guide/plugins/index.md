# 插件实现

v1 按数据、模型和训练流程组织内置插件。配置文件通过实现名称选择插件，新增实现时使用对应的插件类注册。

| 分类 | 内容 |
|------|------|
| [数据插件](data_plugins.md) | DataLoader 与 DataConverter |
| [模型插件](model_plugins.md) | 初始化、PEFT、量化、Kernel 和 Sequence Parallel |
| [训练器插件](trainer_plugins.md) | 分布式后端、批处理和优化器 |
| [融合算子加速](kernel-acceleration/overview.md) | Kernel 选择与调用流程 |

插件注册和参数解析的通用机制见[插件注册机制](../baseplugin_mechanism.md)。

```{toctree}
:maxdepth: 2
:hidden:

data_plugins
model_plugins
trainer_plugins
kernel-acceleration/overview
```
