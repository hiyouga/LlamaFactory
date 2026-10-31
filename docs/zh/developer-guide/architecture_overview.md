# 整体架构

v1 将命令入口、运行流程和可替换实现分开组织。Core 负责连接配置、数据、模型和训练状态，Plugin 为 Core 提供数据加载、模型处理、批处理和分布式训练等具体实现。

这样划分后，更换数据格式不需要改训练循环，更换分布式后端也不需要重新实现 SFT 损失。任务入口负责把组件组装成一次运行，Core 决定何时执行操作，Plugin 提供被选择的具体操作。

下文源码路径均相对于 `src/llamafactory/v1/`。

## 模块分层

| 模块 | 目录 | 职责 |
|------|------|------|
| 命令入口 | `launcher.py` | 路由 `sft`、`dpo`、`rm`、`chat` 和 `merge`，并在多设备训练时通过 `torchrun` 重启 |
| 参数配置 | `config/` | 解析数据、模型、训练和推理参数 |
| 任务入口 | `trainers/`、`samplers/` | 组装 SFT、DPO、RM 或推理流程 |
| Core | `core/` | 管理数据索引、模型加载、样本渲染、批处理、训练循环和推理引擎 |
| Plugin | `plugins/` | 提供数据、模型和训练器相关的可替换实现 |
| 设备抽象 | `accelerator/` | 管理设备、进程组和 DeviceMesh |

## 调用关系

```text
llamafactory-cli <command> config.yaml
  → launcher
  → get_args
  → 任务入口（run_sft / run_dpo / run_rm / run_chat 等）
      ├── DataEngine
      │   ├── DataLoaderPlugin
      │   └── DataConverterPlugin
      ├── ModelEngine
      │   ├── InitPlugin
      │   ├── QuantizationPlugin
      │   ├── PeftPlugin
      │   └── KernelPlugin
      ├── BaseTrainer
      │   ├── BatchGenerator → BatchingPlugin
      │   ├── DistributedPlugin
      │   └── OptimizerPlugin
      └── BaseSampler
```

上图表示组件依赖，实际初始化次序由对应任务入口决定。以 `trainers/sft_trainer.py` 中的 `run_sft` 为例：先解析参数并初始化 `DistributedInterface`，再构造 DataEngine 和 ModelEngine，将数据集、模型与 Renderer 传给 SFTTrainer，最后调用 `fit()`、`save_model()` 并销毁分布式环境。SFTTrainer 继承 BaseTrainer 的运行循环，并提供 SFT 的损失计算。

## 数据如何到达训练循环

```text
DataEngine[index]                  → 标准 Sample（仍是消息结构）
Renderer.process_samples(samples)  → ModelInput 列表（已分词，尚未组 batch）
BatchGenerator                    → 一个更新步所需的 micro-batch 列表
Trainer.compute_loss(batch)       → 标量 loss
BaseTrainer.fit()                 → 反向传播、参数更新与生命周期事件
```

ModelEngine 同时提供模型和与模型匹配的 Renderer，BatchGenerator 持有这个 Renderer。这样，数据源的读取与字段转换可以独立于模型的 tokenizer、chat template 和多模态 processor。

## 状态与实现分别由谁管理

DataEngine 保存数据集和索引，ModelEngine 保存加载结果，BatchGenerator 保存读取进度与缓冲区，BaseTrainer 保存模型、优化器和训练进度。它们都是一次运行中的对象。

插件注册表保存函数或类对象。调用方每次把模型、配置或缓冲区传给插件，并接收处理结果；插件路由对象本身只记录实现名称。例如，分布式插件可以返回有状态的后端 engine，由 BaseTrainer 持有，但注册表本身不保存某次训练的 engine。

判断改动位置时，先区分是在改变任务目标、组件流程，还是替换某个操作：新损失属于 Trainer，统一的数据索引逻辑属于 DataEngine，新的原始字段转换属于 DataConverterPlugin。具体扩展边界分别在下方页面说明。

Core 各组件见[Core（核心模块）](core/index.md)，插件的注册方式见[插件注册机制](baseplugin_mechanism.md)，内置实现见[插件实现](plugins/index.md)。
