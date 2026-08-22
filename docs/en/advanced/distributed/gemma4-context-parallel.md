# Gemma 4 context parallelism

LLaMA Factory supports Ulysses context parallelism for text-only Gemma 4 SFT with DeepSpeed. The attention
all-to-all and the Gemma 4 Triton kernels are supplied by `gemma-triton-flash-attn>=0.2.0`.

Install the maintained fork's immutable release:

```bash
pip install "gemma-triton-flash-attn[hf] @ git+https://github.com/StevenShi-23/gemma-triton-flash-attn.git@v0.2.0"
```

Select `flash_attn: triton_gqa`, and set `context_parallel_size` to a divisor of the distributed world size. For an
eight-GPU run, `context_parallel_size: 4` creates two data-parallel groups with four context-parallel ranks each. See
`examples/train_lora/gemma4_lora_sft_cp.yaml` for a complete configuration.

The first supported surface deliberately requires:

- SFT with DeepSpeed and BF16;
- `per_device_train_batch_size: 1` and `dataloader_drop_last: true`;
- `average_tokens_across_devices: true` for global shifted-token normalization;
- unpacked, text-only data with the vision tower and multimodal projector frozen;
- no evaluation, generation, label smoothing, custom SFT loss, or Transformers-native parallelism in the same run.

Each sample is repeated across the contiguous ranks in its CP group before Accelerate shards the batch stream. Labels
are shifted on the full sequence before the sequence is split, so the causal target at every CP boundary is exact.
The trainer explicitly sums each shard's shifted-token loss and normalizes it by the unique global token count across
DP groups and gradient-accumulation microbatches. It compensates for DeepSpeed's full-world gradient average so CP
ranks do not dilute the gradient.

The effective unique global batch size is
`per_device_train_batch_size * gradient_accumulation_steps * (world_size / context_parallel_size)`. Trainer and
DeepSpeed may still display the physical world-size batch, which includes the repeated CP copies; use the unique batch
size when choosing the learning rate or interpreting sample throughput.

The query-head count must be divisible by `context_parallel_size`. KV heads are sharded when divisible; when a Gemma 4
global-attention layer has fewer KV heads than CP ranks, the attention package expands KV heads before its all-to-all.
