# Adding New Tokens and Initializing Their Embeddings

This example shows how to add new tokens to a model and control how their embeddings are
initialized before training. This is useful when you extend a model with domain-specific
markers (e.g. structural tags, tool-call delimiters, vector-graphics primitives).

## How to add tokens

There are three mutually exclusive ways to register new tokens, all gated by `resize_vocab: true`
(which resizes the embedding layer to fit the new vocabulary):

| Argument | Token kind | Descriptions | Use case |
|---|---|---|---|
| `add_tokens` | normal tokens | no | plain vocabulary expansion |
| `add_special_tokens` | special tokens | no | special markers, no semantic init |
| `new_special_tokens_config` | special tokens | yes (YAML) | special markers with semantic init |

`new_special_tokens_config` points to a YAML file mapping each token to a short description
(see [`tokens_cfg.yaml`](tokens_cfg.yaml)). It takes precedence over `add_special_tokens`.

## Initialization methods (`init_special_tokens`)

New rows are initialized with one of the methods below. The names follow a `<base>` / `<base>_noise`
convention, where the `_noise` suffix adds a small Gaussian perturbation (scaled to the embedding std)
so that otherwise-identical rows stay distinguishable.

| `init_special_tokens` | Base | `+ noise` | Meaning |
|---|---|---|---|
| `vocab_mean` | vocab mean | no | mean of all existing token embeddings (all new rows identical) |
| `vocab_mean_noise` (default) | vocab mean | yes | vocab mean + Gaussian noise |
| `description` | description | no | mean embedding of each token's description tokens |
| `description_noise` | description | yes | description mean + Gaussian noise |
| `noise` | — | — | pure Gaussian noise (no base) |

Notes:

- `description` / `description_noise` require `new_special_tokens_config` (they need descriptions).
  If no config is provided, they fall back to `vocab_mean_noise`.
- Legacy names `noise_init` / `desc_init` / `desc_init_w_noise` are still accepted as deprecated
  aliases of `vocab_mean_noise` / `description` / `description_noise`.

## Training only the new embeddings (`freeze_original_embeddings`)

Set `freeze_original_embeddings: true` to register a gradient mask that updates only the newly
added token rows while keeping the original embeddings frozen. This is intended for setups where
the embedding matrix is trained directly (e.g. `finetuning_type: full`).

Under `finetuning_type: lora` with `resize_vocab`, the input/output embeddings are added to
`additional_target` automatically, so the new token embeddings are already trainable and
`freeze_original_embeddings` is not needed.

## Demo dataset

The example trains on `add_tokens_demo` (registered in [`data/dataset_info.json`](../../../data/dataset_info.json)),
a tiny text-to-SVG set whose responses actually contain the new tokens declared in
[`tokens_cfg.yaml`](tokens_cfg.yaml), e.g.:

```text
<|START_OF_SVG|><|start_of_circle|>cx="50" cy="50" r="40" fill="red"<|end_of_circle|><|END_OF_SVG|>
```

This matters: the added tokens must appear in the training data, otherwise their freshly
initialized embeddings never receive gradients (and `freeze_original_embeddings: true` would
train nothing). After training, the model should be able to emit these tokens. Swap in your
own dataset the same way — just make sure its responses use the tokens you declared.

## Run the examples

```bash
# Full SFT, vocab-mean initialization: add new tokens, initialize their embeddings, then train
llamafactory-cli train examples/extras/add_tokens/qwen3_add_tokens_vocab_mean_noise.yaml

# Full SFT, description (semantic) initialization: init each new row from its description tokens
llamafactory-cli train examples/extras/add_tokens/qwen3_add_tokens_description.yaml
```

## Files

- [`tokens_cfg.yaml`](tokens_cfg.yaml): new special tokens with descriptions.
- [`qwen3_add_tokens_vocab_mean_noise.yaml`](qwen3_add_tokens_vocab_mean_noise.yaml): full SFT with `vocab_mean_noise` initialization.
- [`qwen3_add_tokens_description.yaml`](qwen3_add_tokens_description.yaml): full SFT with `description` (semantic) initialization.
- `add_tokens_demo` (in [`data/`](../../../data/add_tokens_demo.json)): text-to-SVG demo whose responses use the new tokens.

## References

The description-based and noisy initialization recipes here were distilled from work on
vector-graphics modeling by Ximing Xing (ximinng):

- Empowering LLMs to Understand and Generate Complex Vector Graphics. 2025. [[arxiv]](https://arxiv.org/abs/2412.11102)
- Hierarchical SVG Tokenization: Learning Compact Visual Programs for Scalable Vector Graphics Modeling. 2026. [[arxiv]](https://arxiv.org/abs/2604.05072)
