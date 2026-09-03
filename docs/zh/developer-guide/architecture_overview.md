# 整体架构

v1 将命令入口、运行流程和可替换实现分开组织。Core 负责连接配置、数据、模型和训练状态，Plugin 为 Core 提供数据加载、模型处理、批处理和分布式训练等具体实现。

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
  → Trainer / Sampler
      ├── DataEngine
      │   ├── DataLoaderPlugin
      │   └── DataConverterPlugin
      ├── ModelEngine
      │   ├── InitPlugin
      │   ├── QuantizationPlugin
      │   ├── PeftPlugin
      │   └── KernelPlugin
      ├── BaseTrainer
      │   ├── BatchingPlugin
      │   ├── DistributedPlugin
      │   └── OptimizerPlugin
      └── BaseSampler
```

Trainer 和 Sampler 决定任务流程，Core 组件维护运行状态，Plugin 只实现由 Core 调用的可替换操作。例如，`ModelEngine` 决定模型加载顺序，具体的量化、PEFT 和融合算子处理分别由对应插件完成。

Core 各组件见[Core（核心模块）](core/index.md)，插件的注册方式见[插件注册机制](baseplugin_mechanism.md)，内置实现见[插件实现](plugins/index.md)。
