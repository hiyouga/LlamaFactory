# KTransformers LoRA SFT

KTransformers (KT) executes routed MoE experts on CPU while LLaMA-Factory remains responsible for data,
LoRA arguments, and the training entry point. The production scope is routed-BF16 and routed-INT8 LoRA.

KT has one user configuration source: the training YAML. Accelerate YAML contains FSDP2 settings only.
LLaMA-Factory derives LoRA rank, alpha, dropout, activation policy, and local runtime capacity.

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

Routed INT8 additionally requires matching expert and BF16 non-expert artifacts:

```yaml
kt_weight_path: /abs/path/to/routed-int8-experts
kt_non_expert_weight_path: /abs/path/to/bf16-non-expert-cache
kt_config:
  kt_expert_weight_format: int8
  kt_backend: auto
  kt_weight_lifecycle: persistent
```

Launch the standard training entry point through Accelerate:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file examples/ktransformers/accelerate/fsdp2_kt_bf16.yaml \
  src/train.py examples/ktransformers/train_lora/qwen3_5moe_lora_sft_kt.yaml
```

## Load a saved adapter

Use a local, complete KT adapter directory for chat or evaluation. Repeat the training LoRA shape (`finetuning_type`,
`lora_rank`, `lora_alpha`, and `lora_dropout`) and the KT base-weight settings. In particular, routed INT8 loading
must use the same `kt_weight_path` and `kt_non_expert_weight_path` as training.

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

The directory must contain the standard PEFT adapter files and, when fused routed-expert LoRA is used,
`fused_expert_lora.safetensors` plus `kt_adapter_manifest.json`. LLaMA-Factory first loads the standard PEFT
adapter, then KT validates and restores the fused artifact. `adapter_folder` may select a local subdirectory;
paths outside the adapter root and Hub adapter IDs fail before model loading. Download a Hub bundle locally first.

For training resume, keep the original training YAML and use `resume_from_checkpoint`. The optimizer checkpoint
currently requires the same distributed world size. Missing, tampered, or mismatched artifacts fail closed instead
of falling back to the source checkpoint.

Do not combine KT with a second Transformers/FSDP checkpoint wrapper or Unsloth GC, and do not put `kt_config`
in the Accelerate YAML. See the BF16 and INT8 examples under `examples/ktransformers/train_lora/`.
