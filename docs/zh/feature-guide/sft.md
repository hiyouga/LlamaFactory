# 监督微调（SFT）

使用 `llamafactory-cli sft` 启动监督微调。是否传入 `peft_config` 决定全参、LoRA 或 Freeze。

## 全参训练

```yaml
model: Qwen/Qwen3-0.6B
model_class: llm
train_dataset: data/v1_sft_demo.yaml

output_dir: outputs/qwen3_full
micro_batch_size: 1
cutoff_len: 2048
learning_rate: 1.0e-4
max_steps: 10

dist_config:
  name: fsdp2
```

```bash
llamafactory-cli sft config.yaml
```

## LoRA

```yaml
peft_config:
  name: lora
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: all
```

继续训练已有 adapter 时设置 `adapter_name_or_path`。训练只允许一个 adapter；LoRA 参数从 adapter 自身恢复。

## Freeze

```yaml
peft_config:
  name: freeze
  freeze_trainable_layers: 2
  freeze_trainable_modules: all
  freeze_extra_modules: null
  cast_trainable_params_to_fp32: true
```

正数表示最后 N 层，负数表示最前 N 层。

## QLoRA

```yaml
peft_config:
  name: lora
  r: 16
  target_modules: all

quant_config:
  name: bnb
  quantization_bit: 4

dist_config:
  name: fsdp2
```

QLoRA 使用 bitsandbytes 4-bit。字段详情见[模型参数](../configuration/model.md#peft_config)。

## 激活值重算

`enable_activation_checkpointing` 默认为 `true`。启用后，训练在反向传播时重新计算部分前向结果，以减少激活值占用的显存或设备内存，但会增加计算量。内存充足并希望减少重计算时可以设置：

```yaml
enable_activation_checkpointing: false
```
