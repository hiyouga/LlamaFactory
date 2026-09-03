# 开发者指南

开发者指南说明 v1 的运行路径和扩展点。v1 由 Core（核心模块）组织主要运行流程，并通过 Plugin（插件系统）提供可替换实现。

## 架构

| 页面 | 内容 |
|------|------|
| [整体架构](architecture_overview.md) | 模块分层以及 Core 与 Plugin 的调用关系 |

## Core（核心模块）

[Core（核心模块）](core/index.md)说明数据加载、模型加载、样本渲染、批处理和训练循环。

## Plugin（插件系统）

| 页面 | 内容 |
|------|------|
| [插件注册机制](baseplugin_mechanism.md) | `BasePlugin` 的注册、路由和参数解析 |
| [插件实现](plugins/index.md) | 数据、模型和训练器插件，以及融合算子加速 |
