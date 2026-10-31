# 监督微调（SFT）

`llamafactory-cli sft` 启动监督微调。下文依次说明全参训练、LoRA、Freeze 和量化 LoRA 的配置。

## 全参训练

全参训练更新全部模型参数，配置中不设置 `peft_config`。以下完整配置保存为 `config.yaml`，使用 `Qwen/Qwen3-0.6B` 和 FSDP2：

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

LoRA 冻结基座权重并训练 adapter，通过 `peft_config.name: lora` 配置。在上述训练配置中加入以下块，并将 `output_dir` 改为 `outputs/qwen3_lora`；基座仍为 `Qwen/Qwen3-0.6B`：

```yaml
peft_config:
  name: lora
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: all
```

训练结束后，adapter 保存到 `outputs/qwen3_lora`。[推理](inference.md#使用-lora-adapter)和[模型导出](model_export.md#导出配置)示例沿用这一基座与目录。

继续训练已有 adapter 时设置 `adapter_name_or_path`。训练只允许一个 adapter；LoRA 参数从 adapter 自身恢复。

## Freeze

Freeze 直接更新指定层或模块的权重，其余参数保持冻结。以下 `peft_config` 替换 LoRA 配置，模型、数据与训练字段沿用全参示例：

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

量化 LoRA 在量化后的基座上训练 adapter，分别由 `quant_config` 和 `peft_config` 控制。当前 v1 的量化插件入口为 `bnb` 和 `auto`：`bnb` 提供 bitsandbytes 的 4-bit、8-bit 加载分支；`auto` 在指定有效位宽时转交 `bnb` 处理。

下面是使用 bitsandbytes 4-bit 的 QLoRA 示例，运行环境需安装 bitsandbytes。将以下字段加入全参示例，模型、数据与 FSDP2 配置保持一致：

```yaml
output_dir: outputs/qwen3_qlora

peft_config:
  name: lora
  r: 16
  target_modules: all

quant_config:
  name: bnb
  quantization_bit: 4
  quantization_type: nf4
  double_quantization: true
```

`quantization_bit` 表示加载位宽，`quantization_type` 表示 4-bit 量化格式，`double_quantization` 控制 4-bit double quant。位宽分支与字段默认行为见[量化参数](../configuration/model.md#quant_config)，后端限制见 [NPU 说明](../multi-backend/npu/index.md)。

## 激活值重算

`enable_activation_checkpointing` 默认为 `true`。启用后，训练在反向传播时重新计算部分前向结果，以减少激活值占用的显存或设备内存，并增加计算量。设置为 `false` 关闭重计算：

```yaml
enable_activation_checkpointing: false
```

分布式后端、批处理与算子配置分别见[分布式训练](distributed_training.md)、[批处理](batching.md)和[融合算子加速](kernel_acceleration.md)。
