# 模型导出

`merge` 命令将一个或多个 LoRA adapter 依次合并到基座模型，并保存为 Hugging Face 格式目录。

导出目录包含合并后的模型权重。基座与 adapter 的直接加载见[推理](inference.md)，训练状态的保存和续训见[模型保存与恢复](model_saving.md)。

## 导出配置

以下配置保存为 `merge.yaml`，接续 [SFT 的 LoRA 示例](sft.md#lora)：基座为 `Qwen/Qwen3-0.6B`，adapter 位于 `outputs/qwen3_lora`，合并结果保存到 `outputs/qwen3_merged`。`model` 必须与 adapter 训练时的基座一致。

```yaml
model: Qwen/Qwen3-0.6B
peft_config:
  name: lora
  adapter_name_or_path: outputs/qwen3_lora
  export_dir: outputs/qwen3_merged
  export_size: 5
  infer_dtype: auto
  export_legacy_format: false
```

```bash
llamafactory-cli merge merge.yaml
```

导出完成后，可以将推理配置的 `model` 设置为 `outputs/qwen3_merged`，按[推理指南](inference.md#使用训练或合并后的模型)加载。

`export_size` 的单位为 GB。`infer_dtype` 支持 `auto`、`float16`、`float32` 和 `bfloat16`。完整字段见[模型参数](../configuration/model.md#peft_config)。

## 合并多个 Adapter

`adapter_name_or_path` 可以使用列表。系统按照列表顺序将每个 LoRA adapter 合并到前一步得到的模型中：

```yaml
model: Qwen/Qwen3-0.6B
peft_config:
  name: lora
  adapter_name_or_path:
    - outputs/domain_adapter
    - outputs/task_adapter
  export_dir: outputs/qwen3_merged
```

上例先合并 `domain_adapter`，再合并 `task_adapter`。这两个目录是独立训练的 adapter 示例，均须与 `Qwen/Qwen3-0.6B` 基座匹配；使用时替换为实际目录。
