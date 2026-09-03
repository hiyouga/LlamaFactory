# 模型导出

`merge` 命令将一个或多个 LoRA adapter 依次合并到基座模型，并保存为 Hugging Face 格式目录。

## 导出配置

```yaml
model: Qwen/Qwen3-4B
peft_config:
  name: lora
  adapter_name_or_path: outputs/qwen3_lora
  export_dir: outputs/qwen3_merged
  export_size: 5
  infer_dtype: auto
  export_legacy_format: false
```

```bash
llamafactory-cli merge config.yaml
```

`export_size` 的单位为 GB。`infer_dtype` 支持 `auto`、`float16`、`float32` 和 `bfloat16`。完整字段见[模型参数](../configuration/model.md#peft_config)。

## 合并多个 Adapter

`adapter_name_or_path` 可以使用列表。系统按照列表顺序将每个 LoRA adapter 合并到前一步得到的模型中：

```yaml
model: Qwen/Qwen3-4B
peft_config:
  name: lora
  adapter_name_or_path:
    - outputs/domain_adapter
    - outputs/task_adapter
  export_dir: outputs/qwen3_merged
```

上例先合并 `domain_adapter`，再合并 `task_adapter`。列表中的每个目录都需要包含可加载的 LoRA adapter。
