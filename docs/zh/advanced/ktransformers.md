# KTransformers LoRA SFT

KTransformers（KT）将 MoE routed experts 放在 CPU 执行，LLaMA-Factory 继续负责数据、LoRA 参数和训练入口。
当前生产范围是 routed-BF16 LoRA 与 routed-INT8 LoRA；Accelerate 配置只负责 FSDP2，不再保存 KT 参数。

## 安装检查

必须同时安装带 KT 公共接口的 `ktransformers`、`transformers-kt` 和 `accelerate-kt`。启动前可检查：

```bash
python - <<'PY'
from accelerate import Accelerator
from kt_kernel.sft import resolve_kt_pretrained_artifacts
from transformers import TrainingArguments

assert hasattr(TrainingArguments, "update_kt_config")
assert "adapter_only" in __import__("inspect").signature(Accelerator.get_state_dict).parameters
print(resolve_kt_pretrained_artifacts)
PY
```

## 配置

KT 只有一个用户配置源：训练 YAML。LoRA rank、alpha、dropout 和 runtime capacity 由 LLaMA-Factory
标准字段派生；不要在 `kt_config` 中重复填写。

BF16 示例：

```yaml
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_target: all

use_kt: true
disable_gradient_checkpointing: false
kt_cpu_activation: retain
kt_config:
  kt_expert_weight_format: bf16
  kt_backend: AMXBF16
  kt_num_threads: 96
  kt_tp_enabled: true
  kt_threadpool_count: 2
  kt_max_cache_depth: 2
```

INT8 还需要相互匹配的 routed expert 与 BF16 non-expert cache：

```yaml
kt_weight_path: /abs/path/to/routed-int8-experts
kt_non_expert_weight_path: /abs/path/to/bf16-non-expert-cache
kt_config:
  kt_expert_weight_format: int8
  kt_backend: auto
  kt_weight_lifecycle: persistent
```

完整配置见：

- `examples/ktransformers/train_lora/qwen3_5moe_lora_sft_kt.yaml`
- `examples/ktransformers/train_lora/deepseek_v3_int8_lora_sft_kt.yaml`

Activation 策略：

| `disable_gradient_checkpointing` | `kt_cpu_activation` | CPU / GPU |
| --- | --- | --- |
| `false` | `recompute` 或省略 | recompute / recompute |
| `false` | `retain` | retain / recompute |
| `true` | `retain` 或省略 | retain / retain |
| `true` | `recompute` | 不支持，启动前报错 |

## 启动与复用

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file examples/ktransformers/accelerate/fsdp2_kt_bf16.yaml \
  src/train.py examples/ktransformers/train_lora/qwen3_5moe_lora_sft_kt.yaml
```

输出 adapter 同时包含 standard PEFT 与 fused expert LoRA。

## 新进程加载

对话或评测必须使用本地的完整 KT adapter 目录，并重复训练时的 LoRA 形状配置：`finetuning_type`、
`lora_rank`、`lora_alpha`、`lora_dropout`，以及相同的 KT base weight 配置。routed INT8 尤其要沿用训练时
的 `kt_weight_path` 和 `kt_non_expert_weight_path`。

```yaml
model_name_or_path: /abs/path/to/base-model
adapter_name_or_path: /abs/path/to/output/checkpoint-300
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.0

use_kt: true
kt_cpu_activation: retain
kt_config:
  kt_expert_weight_format: bf16
  kt_backend: AMXBF16
  kt_num_threads: 96
```

```bash
llamafactory-cli chat path/to/kt_adapter_infer.yaml
llamafactory-cli eval path/to/kt_adapter_eval.yaml
```

目录必须包含 standard PEFT adapter 文件；使用 fused routed-expert LoRA 时，还必须包含
`fused_expert_lora.safetensors` 和 `kt_adapter_manifest.json`。LLaMA-Factory 先加载 standard PEFT，随后由
KT 校验并恢复 fused artifact。`adapter_folder` 可以选择本地子目录；越出 adapter 根目录的路径和 Hub
adapter ID 会在加载模型前报错，Hub bundle 需要先完整下载到本地。

续训应保留原训练 YAML，并使用 `resume_from_checkpoint`。分布式 optimizer checkpoint 暂要求相同 world
size。artifact 缺失、hash 不匹配或来源模型不一致时会直接失败，不会退回源 checkpoint。

不要同时启用 Transformers/FSDP activation checkpointing、Unsloth GC，也不要把 `kt_config` 放入
Accelerate YAML。每次训练都应确认 loss/grad finite、base model 未修改，并验证 standard/router/fused LoRA
均包含非零更新。
