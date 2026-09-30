# Renderer

当前 Renderer 位于 `core/rendering/`，把 v1 `Sample` 转成 tokenized `ModelInput`。它不再依赖 `RenderingPlugin` 或模型名称模板表。

## 在数据路径中的位置

DataEngine 负责返回标准消息结构，ModelEngine 提供与模型匹配的 processor 并创建 Renderer。BatchGenerator 的 DataLoader 调用 `Renderer.process_samples(samples)`，得到已分词的样本列表，再执行批次组装。

Renderer 的结果包含 `input_ids`、`attention_mask`、`labels`、`loss_weights` 和位置信息；多模态样本还携带 processor 产生的媒体特征。它负责表达模型输入和监督范围，batch 内的 padding、拼接以及长度截断由后续批处理承担。

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

该方法要求 prompt 的 token 序列与完整序列的前缀完全一致。Renderer 会实际比较两次编码的结果；如果模板在追加最后一轮后改写了前面的内容，会抛出错误，避免把上下文 token 当作回复标签。

前缀部分的 `labels` 为 `IGNORE_INDEX`、`loss_weights` 为零；最后回复按该轮的监督权重设置 `loss_weights`，有监督的位置使用实际 token ID 作为 label。生成模式则添加 generation prompt，并把所有位置设为不参与监督。

## 处理多模态内容

Renderer 先把标准内容块转换为 Hugging Face 消息结构，再通过 chat template 生成文本。存在媒体内容时，processor 接收文本与图片、视频或解码后的音频，并返回 token 与媒体特征。Renderer 把这些特征和媒体位置标记一并传给批处理，保持文本与媒体对应。

## 拼接 Chosen/Rejected 序列

chosen 与 rejected 分别渲染后拼接，并使用 `token_type_ids` 标记两个文档。DPO/RM 根据标记构造 block-diagonal attention 和各自 position ids，避免 rejected 序列读取 chosen 序列。

Renderer 为两段序列分别从起点生成 position IDs，并用 `token_type_ids` 的 `1`、`2` 保留段边界。物理拼接只改变存储布局；两段仍是独立回答，任务 Trainer 需要保留这个隔离语义。

## 转义特殊 Token

`core/rendering/escape.py` 在渲染用户控制的文本和 tools 前处理中和 tokenizer 的特殊 token 字符串，避免输入直接注入特殊 token。

## 自定义 Chat Template

新模型通常应在模型仓库的 tokenizer 配置中提供标准 HF chat template。仅在运行时覆盖时使用 `custom_chat_template`，无需注册 Python 插件。

若原始数据字段需要变化，扩展 converter；若标准消息如何映射到模型输入发生变化，再检查 `rendering/format.py` 和 `rendering/rendering.py`。转换原始字段与生成监督标签分属不同阶段，避免在 converter 中依赖模型的 token ID。
