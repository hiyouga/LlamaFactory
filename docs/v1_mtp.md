# Multi-Token Prediction (MTP) — v1 architecture

[English](v1_mtp.md) | **简体中文**

This feature adds Multi-Token Prediction (MTP) support to the v1 architecture, adapted
from the FSDP2 MTP implementation in
[MindSpeed-LLM](https://gitcode.com/Ascend/MindSpeed-LLM) (`mindspeed_llm/fsdp2/models/common/mtp.py`).

MTP appends `K` extra prediction heads to a decoder-only causal LM. Each head `k`
predicts the token at offset `k + 2` from the current position (the main head predicts
offset `1`). The total training loss is:

```
total_loss = lm_loss + loss_scale * mtp_loss
```

where `mtp_loss` is the mean of the per-head cross-entropy losses (weighted by
`loss_weights`, like the main SFT loss), and `loss_scale` defaults to `0.3`.

## How it works

- `MultiTokenPredictionBlock` (`src/llamafactory/v1/plugins/model_plugins/mtp.py`) holds
  `K` heads. Each head reuses the base model's decoder layer class. The block owns shared
  `enorm`/`hnorm` norms, `e_proj`/`h_proj` projections and a `final_layernorm`, exactly as
  in MindSpeed-LLM.
- `MTPModelPlugin` attaches the block to the model as `model.mtp` and patches
  `model.forward` so the model output carries `mtp_logits` (a list of per-head logits)
  during training. The MTP loss is computed by the trainer through `compute_mtp_loss`.
- Under FSDP2, the MTP heads' inner decoder layers are sharded automatically: the generic
  `FSDP2Engine.prepare_model` wraps every module of the base model's decoder-layer class,
  which includes `mtp.layers.*.layer`.

## Usage

Add an `mtp_config` block to your v1 YAML:

```yaml
model: Qwen/Qwen3-0.6B
model_class: llm
template: qwen3_nothink

mtp_config:
  name: mtp
  num_layers: 1   # number of MTP heads (K)
  loss_scale: 0.3 # optional, default 0.3

dist_config:
  name: fsdp2
  dcp_path: null

train_dataset: data/v1_sft_demo.yaml
output_dir: outputs/test_mtp
micro_batch_size: 1
cutoff_len: 2048
learning_rate: 1.0e-4
max_steps: 10
```

See `examples/v1/train_full/train_full_mtp.yaml` for a complete example (MTP + CP +
checkpoint save/resume in one config).

## Compatibility

MTP currently targets Llama/Qwen3/Qwen3.5/Mistral-style models that expose `model.model.layers`,
`model.model.rotary_emb`, `model.model.norm` and `model.lm_head`. The MTP heads are
randomly initialized; loading a base checkpoint that does not contain `mtp.*` keys will
leave them at their initialization (missing-key warnings are expected).

### Layer selection (hybrid-attention models)

Each MTP head reuses the base model's decoder layer *class* and is cloned from a layer
index whose attention type is **full self-attention**. For hybrid-attention models this
matters: Qwen3 mixes `full_attention` with `sliding_attention`; Qwen3.5 mixes
`full_attention` with `linear_attention` (GDN). An MTP head predicts the token at offset
`k + 2` over the *full* sequence and needs global context, so a sliding-window or GDN
head (local / recurrent view) would be wrong. `_select_layer_idx_for_mtp` picks the last
`full_attention` layer index from `config.layer_types` (falling back to the last layer for
all-full models like Llama/Mistral). The selected index is logged on attach, e.g.
`decoder layer cloned from layer_idx=7 [full_attention]`.

## Saving and loading MTP weights

`mtp.embed_tokens` and `mtp.output_layer` are **shared** with the base model's embedding
and `lm_head`, so they are stripped from the saved state dict (`strip_shared_mtp_keys`)
before `save_pretrained` to avoid transformers' "shared tensors not properly defined"
error. Only the MTP-specific tensors (`layers.*`, `enorm`/`hnorm`/`e_proj`/`h_proj`/
`final_layernorm`) are written alongside the base model weights.

On load, `from_pretrained` drops `mtp.*` keys as unexpected (the MTP block is grafted at
runtime). `ModelEngine` therefore calls `apply_mtp` (re-creating the block and re-sharing
the embedding/`lm_head`) and then `load_mtp_weights` to restore the saved MTP tensors from
the checkpoint. This is automatic; no extra config is needed. The FSDP2 meta path loads
`mtp.*` through the regular HF weight loop, and DCP resume restores them by FQN.

## Context Parallelism (MTP + CP)

MTP also works under Ulysses context parallelism (CP). CP requires
`dist_config.name: fsdp2` and `flash_attn: flash_attention_2` (the same constraints as
non-MTP CP). When MTP and CP are both enabled:

- The MTP decoder layers go through the same globally-patched `_flash_attention_forward`
  as the main model, so they participate in Ulysses attention automatically. (Each MTP
  head is a `full_attention` layer, so it always uses `_flash_attention_forward`.)
- `BaseTrainer.fit` routes to the `sequence_parallel_mtp_loss` plugin, which computes the
  main-head CP loss (unchanged) plus the scaled MTP loss. The per-head MTP loss is
  computed on the full sequence by all-gathering `labels` / `loss_weights` / `log_probs`
  across the CP group (see `compute_mtp_loss` with `cp_group` in `mtp.py`), mirroring the
  single-head `sequence_parallel_loss` plugin.
- The MTP input shift (`shift_input_ids_for_mtp`) is CP-aware: each rank's chunk-tail is
  filled with the *next rank's first token* (all-gathered across the CP group) instead of
  a pad value, so the next-token embedding at every CP boundary is correct. Only the
  global last rank's tail (the true sequence end) is padded.

```yaml
mtp_config:
  name: mtp
  num_layers: 1
  loss_scale: 0.3

flash_attn: flash_attention_2

dist_config:
  name: fsdp2
  dcp_path: null
  cp_mode: ulysses
  cp_size: 2
```

See the CP section of `examples/v1/train_full/train_full_mtp.yaml`. CP is not supported
with DeepSpeed (use FSDP2).
