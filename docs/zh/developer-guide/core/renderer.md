# Renderer

当前 Renderer 位于 `core/rendering/`，把 v1 `Sample` 转成 tokenized `ModelInput`。它不再依赖 `RenderingPlugin` 或模型名称模板表。

## Chat Template 的来源

1. 使用 processor/tokenizer 自带 `chat_template`。
2. `custom_chat_template` 可以在 ModelEngine 中覆盖它。
3. 模型完全没有模板时使用内置 ChatML fallback。

## 生成训练标签

多轮对话由 DataEngine 按每个受监督 assistant turn 切分为训练样本（见[DataEngine](data_engine.md)）。Renderer 对样本最后一个 assistant turn 分别渲染 prompt 和完整序列，通过两者前缀差恢复监督 token 区间，不维护模型专属 role marker 表。

```text
messages before assistant + generation prompt → prompt ids
messages including assistant response          → full ids
full ids 中超出 prompt 的尾部                 → supervised span
```

这种方式兼容模型在历史消息中处理 reasoning 内容的差异。

## 拼接 Chosen/Rejected 序列

chosen 与 rejected 分别渲染后拼接，并使用 `token_type_ids` 标记两个文档。DPO/RM 根据标记构造 block-diagonal attention 和各自 position ids，避免 rejected 序列读取 chosen 序列。

## 转义特殊 Token

`core/rendering/escape.py` 在渲染用户控制的文本和 tools 前处理中和 tokenizer 的特殊 token 字符串，避免输入直接注入特殊 token。

## 自定义 Chat Template

新模型通常应在模型仓库的 tokenizer 配置中提供标准 HF chat template。仅在运行时覆盖时使用 `custom_chat_template`，无需注册 Python 插件。
