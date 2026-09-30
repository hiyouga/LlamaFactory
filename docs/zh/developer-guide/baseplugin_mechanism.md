# 插件注册机制

`BasePlugin` 位于 `utils/plugin.py`，负责按名称注册和查找实现。每个 `BasePlugin` 子类表示一个独立的插件类型，并拥有自己的 `_registry`，例如 `OptimizerPlugin` 和 `DistributedPlugin` 的注册项互不共享。

## 插件类型、路由对象与实现

以 `DistributedPlugin("fsdp2").shard_model(...)` 为例，这里有三个不同对象：

| 对象 | 职责 |
|------|------|
| `DistributedPlugin` | 插件类型，持有这个类型的名称注册表 |
| `DistributedPlugin("fsdp2")` | 路由对象，仅保存要查找的名称 |
| 注册在 `fsdp2` 下的实现类 | 提供 `shard_model` 等实际操作 |

创建路由对象只保存名称，不会立即加载或创建后端 engine。第一次调用函数或访问实现方法时，`_resolve` 才从该类型的注册表中查找对象。模型、优化器、buffer 等运行状态由调用方持有并作为参数传入。

## 注册与调用流程

```text
定义插件类型
  → PluginType(BasePlugin)
  → PluginType("name").register()
  → 将函数或类对象写入 PluginType._registry
  → PluginType("name") 按名称查找实现
  → 通过 __call__ 或 __getattr__ 调用
```

装饰器在模块导入时完成注册。注册名称必须存在；查找未注册的名称时会抛出 `ValueError`。同一插件类型重复注册相同名称时会记录警告，并使用后注册的实现。

注册表不会自动扫描文件或按 YAML 名称导入 Python 模块。内置实现通过其入口模块的正常导入完成注册，例如 Kernel 入口显式导入各实现。新增实现也必须在被调用前执行所在模块的导入，仅添加文件或写入配置名称不会触发注册。

## 函数实现

一个插件名称只对应一个操作时，直接注册函数。调用 `PluginType("name")(...)` 时，`BasePlugin.__call__` 查找并调用该函数。

```python
class OptimizerPlugin(BasePlugin):
    pass


@OptimizerPlugin("example").register()
def create_optimizer(model, config):
    return ExampleOptimizer(model.parameters(), lr=config["learning_rate"])


optimizer = OptimizerPlugin("example")(model, config)
```

数据加载、数据转换、模型初始化、PEFT 和优化器等插件使用这种形式。插件类型也可以提供语义化方法，在方法内部调用 `super().__call__`，例如 `DataLoaderPlugin.load(...)`。

## 类实现（静态方法组）

同一插件名称需要提供多个相关操作时，注册一个类对象。`BasePlugin.__getattr__` 先查找该类，再把方法访问转发给它。

```python
@DistributedPlugin("example").register()
class ExampleDistributed(BaseDistributed):
    @staticmethod
    def shard_model(model, dist_config, **kwargs):
        ...

    @staticmethod
    def save_checkpoint(model, optimizer, checkpoint_dir, **kwargs):
        ...


model = DistributedPlugin("example").shard_model(model, dist_config)
DistributedPlugin("example").save_checkpoint(model, optimizer, checkpoint_dir)
```

通过 `.method(...)` 访问时，路由会取出注册类上的方法，不创建该类的实例。因此，这类实现使用 `staticmethod` 或 `classmethod`，不在 `self` 中保存运行状态：

| 方法形式 | 调用时接收 | 用途 |
|----------|------------|------|
| `staticmethod` | 只接收显式参数 | 实现不依赖类本身的独立操作，例如保存 checkpoint |
| `classmethod` | 首个参数为 `cls` | 需要调用子类方法的公共流程，例如 `BaseKernel.apply` |
| 普通实例方法 | 首个参数为 `self` | 需要先创建实例，不适用于当前类对象路由 |

Python 没有单独的“静态类”类型；这里使用的是包含静态方法或类方法的普通类。分布式后端、批处理策略和 Kernel 使用这种方法组形式。对应抽象基类声明的也是静态方法或类方法，`ensure_methods_implemented` 在具体子类定义时检查这些方法是否完整。

## 两种实现形式的选择

| 实现形式 | 注册对象 | 调用入口 | 适用情况 |
|----------|----------|----------|----------|
| 函数 | 函数对象 | `PluginType("name")(...)` | 一个名称对应一个操作 |
| 类方法组 | 类对象 | `PluginType("name").method(...)` | 一个名称需要提供多个相关操作 |

两种形式使用相同的名称注册表。区别只在于注册对象和调用方式，不影响配置文件通过 `name` 选择实现。

## 参数解析

注册过程只完成名称到实现的映射，不会自动解析参数。参数配置经过两个不同阶段：

1. `config/arg_utils.py` 的 `get_plugin_config` 将顶层参数中的插件配置包装成 PluginConfig，并检查是否有 `name`。这时并未根据名称验证插件专属字段。
2. 调用进入具体实现后，由实现解析自己的字段。PEFT 和分布式入口使用参数 dataclass 与 `parse_params`；Muon 等实现直接读取配置。插件类型本身没有为所有实现统一选择参数类。

使用参数 dataclass 的入口可以显式调用 `parse_params(config, ParamsClass)`：

```python
from dataclasses import dataclass
from typing import Literal


class ExamplePlugin(BasePlugin):
    pass


@dataclass
class ExampleParams:
    name: Literal["example"] = "example"
    enabled: bool = True


@ExamplePlugin("example").register()
def apply_example(model, config):
    params = ExamplePlugin.parse_params(config, ExampleParams)
    if params.enabled:
        ...
```

`parse_params` 的处理规则如下：

- `config` 为目标 dataclass 实例时直接返回；
- `config` 为字典时检查字段名，再创建参数实例；
- `config` 为 `None` 时使用 dataclass 默认值创建实例；
- 未声明的字段会抛出 `ValueError`；
- 参数类不是 dataclass，或配置不是字典、`None`、目标 dataclass 实例时，会抛出 `TypeError`；
- 取值范围和字段组合可以在参数 dataclass 的 `__post_init__` 中继续校验。

这里的严格检查主要针对未知字段和配置对象的形式；Python dataclass 不会自动强制执行每个类型注解或 `Literal` 的可选值，额外约束需要实现显式校验。

参数 dataclass 由具体插件入口选择，因此不同实现可以使用不同字段。用户可配置字段记录在对应的[参数说明](../configuration/index.md)中。
