# ModelEngine

`ModelEngine(model_args, is_train=False)` 拥有 processor、model config、Renderer 和最终 Hugging Face model。

## 模型加载流程

```text
AutoProcessor.from_pretrained
  → 同步或覆盖 chat_template
  → AutoConfig.from_pretrained
  → Renderer(processor)
  → 选择初始化设备
  → 应用量化加载参数
  → 选择 AutoModel 类
  → from_pretrained / from_config
  → PEFT
  → Kernel
```

## 同步 Chat Template

多模态 processor 没有模板时，会从其 tokenizer 同步。`custom_chat_template` 则覆盖 processor/tokenizer 模板。Renderer 最终调用 `apply_chat_template`，不再导入模型专属 Python 模板。

## 选择 Hugging Face 模型类

- `llm`：因果语言模型或 image-to-text 模型
- `cls`：单标签 token classification 模型，RM 使用
- `other`：`AutoModel`

## Meta 与 ZeRO-3 初始化

- DeepSpeed ZeRO-3 训练路径在加载模型前设置对应上下文。
- `init_on_meta` 通过 `init_empty_weights()` 从 config 构造模型。
- meta device 与量化互斥。
- 量化、PEFT、Kernel 按上述顺序应用，后续阶段基于前一阶段的模型结果。
