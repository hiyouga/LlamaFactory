# State-routing context parallelism for GDN

The v1 `ulysses_gdn_state` mode keeps full-attention layers on the existing Ulysses path and uses Attention Gym's state-routing implementation for Qwen3.5 Gated DeltaNet layers. Each GDN rank processes its local tokens and exchanges recurrent-state summaries and short-convolution history instead of full-sequence projected activations.

For a fixed batch size and CP topology, the GDN collective buffers have fixed size as sequence length increases. Full-attention Ulysses and FSDP communication are separate and unchanged.

## Installation and configuration

Install the optional backend in the same environment as LlamaFactory:

```bash
python -m pip install -r requirements/attention-gym.txt
```

The backend is pinned to Attention Gym revision `755691f3fe5da772a6e4023ea08bbb4db8173b49`, matching the state-routing integration in torchtitan. This mode requires Linux/CUDA, Transformers 5.9 or newer, BF16 training, and GDN key/value head dimensions of 128. The included configuration enables BF16:

```bash
USE_V1=1 llamafactory-cli train examples/v1/train_full/train_full_gdn_state_cp.yaml
```

The selection is independent of parameter sharding:

```yaml
flash_attn: flash_attention_2
bf16: true
cp_size: 2
cp_mode: ulysses_gdn_state

dist_config:
  name: fsdp2
```

The existing `cp_mode: ulysses` remains the default. Select it to use head/sequence exchange for GDN as well. State routing uses the pinned Attention Gym kernels directly; it does not require the FSDPTurbo FLA kernel plugin.

## Text batch layouts

The mode supports independent right-padded text rows and flattened padding-free packed text. An explicit all-ones mask on a packed row is normalized to `None` so the existing full-attention backend honors document boundaries. Before the shared CP split, the language-boundary hook already sees global positions and masks on every rank. It builds routing locally once per forward and passes that immutable object through model kwargs, including activation-checkpoint recomputation. There is no per-layer gather of token positions for GDN routing.

Document resets and batch boundaries terminate recurrent and convolution history. A right-padding tail becomes a separate zero-input sequence, with its targets already ignored by the standard batch preparation. Each rank still owns one contiguous fragment per batch row; internal documents increase local subsequences without adding communicated state-summary slots.

Packed rows with an actual padding mask, packed batches with more than one row, left/interior padding, non-binary masks, non-contiguous positions between resets, and true multidimensional mRoPE positions are rejected before model collectives. The first two restrictions preserve the semantics of the existing full-attention path; the GDN component itself is also tested on packed+padded spans. The state path is for training with generation caches disabled. It currently supports the split-projection Qwen3.5 GDN structure with a bias-free SiLU depthwise convolution and a common convolution width across layers.

## Tests

`tests_v1/plugins/model_plugins/test_gdn_state_cp.py` covers routing construction, hook ordering, immutable per-forward metadata, and native decoder keyword propagation. Its two-GPU cases compare against independent per-document native GDN forwards, using identical local Attention Gym kernels. They check valid outputs, input gradients, parameter gradients, cross-rank history gradients, exact document isolation, and activation checkpointing for dense, batched, packed, padded, and packed+padded layouts.

`tests_v1/plugins/model_plugins/test_gdn_attention.py` separately covers the existing Ulysses path's native-kernel lookup across Transformers API versions, retaining existing instance overrides and fast-kernel precedence.
