# BaseSampler

`BaseSampler` 位于 `core/base_sampler.py`，为推理调用方提供生成接口，并持有具体的推理 engine。模型和 Renderer 由 [ModelEngine](model_engine.md) 创建后传入；Sampler 负责选择后端和转发调用。

## 组件与状态

| 组件 | 职责 | 源码 |
|------|------|------|
| BaseSampler | 根据 `sample_backend` 选择 engine，转发生成请求 | `core/base_sampler.py` |
| BaseEngine / HuggingFaceEngine | 定义后端接口，使用模型和 Renderer 执行生成 | `core/utils/inference_engine.py` |
| SyncSampler | 将异步生成接口适配为 CLI 使用的同步迭代器 | `samplers/cli_sampler.py` |
| `run_chat` | 创建组件，读取用户输入，保存对话历史并输出回复 | `samplers/cli_sampler.py` |

当前 BaseSampler 直接构造 HuggingFaceEngine，后端选择不经过 BasePlugin 注册表。HuggingFaceEngine 保存模型、Renderer 和推理参数；对话历史由 `run_chat` 中的消息列表持有，每次请求传入完整历史。

## 单次生成路径

```text
run_chat：追加 user 消息
  → SyncSampler.generate(messages)
  → BaseSampler.generate(messages, tools)
  → HuggingFaceEngine.generate(messages, tools)
      → Renderer.render_messages(..., is_generate=True)
      → input_ids / attention_mask 转为设备上的张量
      → 后台线程调用 model.generate
      → AsyncTextIteratorStreamer 异步返回文本片段
  → CLI 逐段输出，拼成 assistant 消息并加入历史
```

HuggingFaceEngine 用信号量限制同时进入生成过程的请求数。它将 `max_new_tokens` 和 streamer 传给 `model.generate`，streamer 跳过 prompt 和特殊 token；返回值是解码后的文本片段，不是 token ID，也不保证每个片段恰好对应一个 token。

SyncSampler 另行维护后台事件循环，逐次等待异步生成器的下一段文本，再交给同步调用方。这个适配层改变调用方式，不改变消息渲染或模型生成逻辑。

## 接口与当前支持范围

| 接口 | 输入与输出 | 当前行为 |
|------|------------|----------|
| `generate(messages, tools=None)` | 标准 Message 列表、可选 tools 字符串；异步产出 `str` | Hugging Face 流式生成 |
| `batch_infer(dataset)` | Dataset；约定返回 Sample 列表 | HuggingFaceEngine 抛出 `NotImplementedError` |

BaseSampler 的非 HF 后端分支会抛出 `ValueError`。CLI 中设置 `train_dataset` 会进入批量推理分支，但当前 engine 尚未实现该接口。因此，已实现的用户路径是 HF 交互式对话，启动方式见[推理](../../feature-guide/inference.md)，配置字段见[推理参数](../../configuration/inference.md)。

## 扩展边界

新增推理后端时，实现 `BaseEngine` 的接口，并接入 BaseSampler 的后端选择；仅添加文件或写入新的 `sample_backend` 名称不会完成接入。后端需要保持异步文本流接口，并明确批量推理是否可用，SyncSampler 才能沿用现有适配方式。

改变模板和消息到模型输入的转换时，修改 [Renderer](renderer.md)；改变生成参数的传递、模型调用或输出流时，修改 engine；改变命令行输入和对话历史管理时，修改 `run_chat`。当前 HF 生成调用只传递 Renderer 结果中的 `input_ids` 和 `attention_mask`，扩展其他模型输入时还需要补齐 engine 向模型传参的路径。
