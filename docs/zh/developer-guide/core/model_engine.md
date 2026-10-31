# ModelEngine

`ModelEngine(model_args, is_train=False)` 拥有 processor、model config、Renderer 和最终 Hugging Face model。

实现位于 `core/model_engine.py`。它把同一份模型配置转为相互匹配的模型与输入处理组件；任务入口随后把 `model` 和 `renderer` 交给 Trainer 或 Sampler。训练循环、优化器创建和常规分布式分片在后续组件中完成。

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

这个顺序由 ModelEngine 固定：初始化插件先决定在哪里创建模型，量化插件再补充权重加载参数；模型对象创建后，PEFT 决定可训练部分或加载 adapter，Kernel 最后处理实际得到的模型。插件之间不互相调度，后一阶段接收前一阶段的结果。

`is_train` 会传给量化和 PEFT 路径，并影响未启用 PEFT 时的模型精度处理。它表达本次加载的用途，训练循环仍会在 `fit()` 中设置模型的训练模式。

## 同步 Chat Template

多模态 processor 没有模板时，会从其 tokenizer 同步。`custom_chat_template` 则覆盖 processor/tokenizer 模板。Renderer 最终调用 `apply_chat_template`，不再导入模型专属 Python 模板。

## 选择 Hugging Face 模型类

- `llm`：根据模型 config 选择 image-to-text、多模态语言模型或因果语言模型类；多模态语言模型类仅在当前 Transformers 提供时参与选择
- `cls`：单标签 token classification 模型，RM 使用
- `other`：`AutoModel`

## Meta 与 ZeRO-3 初始化

普通路径通过 `from_pretrained` 加载权重。选择 meta device 后，ModelEngine 使用 `init_empty_weights()` 和 `from_config` 构造模型结构，此时参数没有真实数据，需要后续分布式加载路径填充。`init_on_rank0` 则让 rank 0 在 CPU 加载，其余 rank 构造 meta 模型。

Meta 初始化不能与量化同时使用；当前 LoRA 路径也会拒绝 `init_on_meta`。这些检查发生在模型加载阶段，不能通过调整 YAML 中字段的书写顺序绕过。

DeepSpeed ZeRO-3 需要在 `from_pretrained` 创建参数时介入，因此 ModelEngine 在加载前建立 ZeRO-3 上下文，并在加载后清理；后续 BaseTrainer 再准备训练 engine。它与 FSDP2 在 Trainer 中对已构造模型进行分片的时机不同。

## 扩展边界

改变加载顺序或 AutoModel 类选择时，修改 ModelEngine；增加一种初始化、PEFT 或量化实现时，扩展相应[模型插件](../plugins/model_plugins.md)。只需改变消息格式时，使用 chat template 或修改 Renderer 的转换逻辑，无需增加一种模型加载插件。
