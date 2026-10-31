# 推理

v1 已实现基于 Hugging Face 后端的 `chat` 入口，可以在命令行中进行流式对话。模型对话格式来自 tokenizer 自带的 Hugging Face chat template；没有模板时回退到内置 ChatML。

当前 `chat` 入口支持 `sample_backend: hf` 的交互式单条推理。批量推理和 `vllm` 采样后端尚未接入该入口；交互式配置中不设置 `train_dataset`。

## 启动 CLI 对话

`model` 指定模型 Hub ID 或包含完整权重的 HF 模型目录。LoRA adapter 的加载方式见下文。

以下对话配置保存为 `chat.yaml`，使用与 [SFT 示例](sft.md)相同的基座：

```yaml
model: Qwen/Qwen3-0.6B
sample_backend: hf
max_new_tokens: 512
```

```bash
llamafactory-cli chat chat.yaml
```

## 覆盖模型 Chat Template

`custom_chat_template` 接收一段 Jinja2 模板字符串，并覆盖 tokenizer 自带模板：

```yaml
model: path/to/model
custom_chat_template: >-
  {% for message in messages %}
  {{ message['role'] + ': ' + message['content'] }}
  {% endfor %}
```

v1 使用模板字符串，不接受 `template: <name>` 字段。

## 使用 LoRA Adapter

LoRA 推理通过 `model` 指定训练时的基座，通过 `peft_config.adapter_name_or_path` 指定 adapter。以下片段加入 `chat.yaml`，加载 [SFT 的 LoRA 示例](sft.md#lora)产生的 `outputs/qwen3_lora`；对应基座为 `Qwen/Qwen3-0.6B`：

```yaml
peft_config:
  name: lora
  adapter_name_or_path: outputs/qwen3_lora
```

推理模式会依次合并 `adapter_name_or_path` 中的 adapter。[模型导出](model_export.md)说明合并结果的持久化保存。

## 使用训练或合并后的模型

完成 [SFT 全参示例](sft.md#全参训练)后，将 `chat.yaml` 中的 `model` 改为 `outputs/qwen3_full`；完成[模型导出示例](model_export.md#导出配置)后，改为 `outputs/qwen3_merged`。这两种目录都包含完整模型权重，加载时移除此前的 `peft_config`，保留采样参数并运行 `llamafactory-cli chat chat.yaml`。

完整字段见[推理参数](../configuration/inference.md)。
