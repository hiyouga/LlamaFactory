# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections.abc import Iterable
from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional

import torch
from transformers.integrations import is_deepspeed_zero3_enabled

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


logger = logging.get_logger(__name__)

# Canonical init-method names are: 'vocab_mean', 'noise', 'vocab_mean_noise', 'description', 'description_noise'.
# The following legacy names remain accepted as deprecated aliases for backward compatibility.
# NOTE: keep this in sync with the same mapping referenced in hparams/model_args.py.
INIT_METHOD_ALIASES = {
    "noise_init": "vocab_mean_noise",
    "desc_init": "description",
    "desc_init_w_noise": "description_noise",
}


def get_embedding_vocab_size(model: "PreTrainedModel") -> int:
    r"""Get the vocab size from the input embedding layer.

    Handles DeepSpeed ZeRO-3 parameter sharding by gathering the embedding weight
    before reading its size.
    """
    embedding = model.get_input_embeddings()
    if is_deepspeed_zero3_enabled():
        import deepspeed  # type: ignore

        with deepspeed.zero.GatheredParameters([embedding.weight]):
            return embedding.weight.size(0)

    return embedding.weight.size(0)


def _resolve_new_token_ids(
    new_tokens: Optional[Iterable[str]],
    tokenizer: "PreTrainedTokenizer",
    embed_size: int,
) -> Optional[list[int]]:
    r"""Resolve the explicit embedding-row IDs of the newly added tokens.

    Relying on ``embed_weight[-num_new_tokens:]`` to locate new tokens is unsafe when
    the model embedding was already padded beyond the tokenizer vocab (e.g. Qwen2.5-VL
    has vocab 151665 but embedding 151936). In that case the appended tokens land
    inside the original padding zone and the tail slice points at the wrong rows.

    Args:
        new_tokens: Iterable of the newly added token strings.
        tokenizer: The tokenizer instance.
        embed_size: Current embedding size (upper bound for valid token IDs).

    Returns:
        A sorted list of unique, in-range token IDs, or ``None`` when no tokens are
        given so that callers can fall back to the tail-slice behaviour.
    """
    if not new_tokens:
        return None

    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    token_ids: set[int] = set()
    for token_str in new_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        if token_id is None or token_id == unk_token_id or not (0 <= token_id < embed_size):
            logger.warning_rank0(f"Token '{token_str}' not found or out of range, skipping during init.")
            continue

        token_ids.add(token_id)

    return sorted(token_ids) or None


def _existing_embeddings(
    embed_weight: "torch.Tensor", num_new_tokens: int, new_token_ids: Optional[list[int]]
) -> "torch.Tensor":
    """Return the rows treated as 'existing' embeddings used as the init baseline.

    Prefers excluding the explicit new-token rows (robust to padding). Falls back to
    dropping the last ``num_new_tokens`` rows when no explicit IDs are available.
    """
    if new_token_ids:
        mask = torch.ones(embed_weight.size(0), dtype=torch.bool, device=embed_weight.device)
        mask[torch.as_tensor(new_token_ids, device=embed_weight.device, dtype=torch.long)] = False
        return embed_weight[mask]

    if num_new_tokens > 0:
        return embed_weight[:-num_new_tokens]

    return embed_weight


def _vocab_mean_noise_initialization(
    embed_weight: "torch.Tensor", num_new_tokens: int, token_ids: Optional[list[int]] = None
) -> None:
    """Initialize new token embeddings with mean + Gaussian noise.

    This is the default initialization method used by LlamaFactory.

    Args:
        embed_weight: The embedding weight matrix to initialize (shape: [vocab_size, embedding_dim])
        num_new_tokens: Number of new tokens added at the end of the embedding matrix
        token_ids: Explicit token IDs to initialize. When provided, these exact rows are
            written (robust to padding). When ``None``, falls back to the last
            ``num_new_tokens`` rows.
    """
    embedding_dim = embed_weight.size(1)
    avg_weight = _existing_embeddings(embed_weight, num_new_tokens, token_ids).mean(dim=0, keepdim=True)

    if token_ids:
        noise_weight = torch.empty(len(token_ids), embedding_dim, device=embed_weight.device, dtype=embed_weight.dtype)
        noise_weight.normal_(mean=0, std=(1.0 / math.sqrt(embedding_dim)))
        embed_weight[token_ids] = avg_weight + noise_weight
    else:
        noise_weight = torch.empty_like(embed_weight[-num_new_tokens:])
        noise_weight.normal_(mean=0, std=(1.0 / math.sqrt(embedding_dim)))
        embed_weight[-num_new_tokens:] = avg_weight + noise_weight


def _noise_initialization(
    embed_weight: "torch.Tensor", num_new_tokens: int, token_ids: Optional[list[int]] = None
) -> None:
    """Initialize new token embeddings with pure Gaussian noise (no mean component).

    The noise std matches the std of the existing embeddings so that the new rows
    live on a comparable scale.

    Args:
        embed_weight: The embedding weight matrix to initialize (shape: [vocab_size, embedding_dim])
        num_new_tokens: Number of new tokens added (used to compute the std baseline)
        token_ids: Explicit token IDs to initialize. When ``None``, falls back to the last
            ``num_new_tokens`` rows.
    """
    existing_std = _existing_embeddings(embed_weight, num_new_tokens, token_ids).std().item()

    if token_ids:
        noise_weight = torch.empty(
            len(token_ids), embed_weight.size(1), device=embed_weight.device, dtype=embed_weight.dtype
        )
        noise_weight.normal_(mean=0, std=existing_std)
        embed_weight[token_ids] = noise_weight
    else:
        noise_weight = torch.empty_like(embed_weight[-num_new_tokens:])
        noise_weight.normal_(mean=0, std=existing_std)
        embed_weight[-num_new_tokens:] = noise_weight


def _vocab_mean_initialization(
    embed_weight: "torch.Tensor", num_new_tokens: int, token_ids: Optional[list[int]] = None
) -> None:
    """Initialize all new token embeddings with the mean of the existing embeddings.

    All new tokens share an identical embedding (no discriminability). Useful as a
    deterministic baseline or when noise is undesirable.

    Args:
        embed_weight: The embedding weight matrix to initialize (shape: [vocab_size, embedding_dim])
        num_new_tokens: Number of new tokens added (used to compute the mean baseline)
        token_ids: Explicit token IDs to initialize. When ``None``, falls back to the last
            ``num_new_tokens`` rows.
    """
    avg_weight = _existing_embeddings(embed_weight, num_new_tokens, token_ids).mean(dim=0, keepdim=True)

    if token_ids:
        embed_weight[token_ids] = avg_weight.expand(len(token_ids), -1).clone()
    else:
        embed_weight[-num_new_tokens:] = avg_weight.expand(num_new_tokens, -1).clone()


def _description_initialization(
    embed_weight: "torch.Tensor",
    num_new_tokens: int,
    descriptions: dict[str, str],
    tokenizer: "PreTrainedTokenizer",
    model: "PreTrainedModel",
    new_token_ids: Optional[list[int]] = None,
    add_noise: bool = False,
) -> None:
    """Initialize new token embeddings based on textual descriptions.

    For each new token, this function:
    1. Tokenizes its description text
    2. Gets embeddings of the description tokens
    3. Averages them to initialize the new token's embedding
    4. Optionally adds Gaussian noise

    New tokens are placed by their resolved token ID rather than by tail slicing,
    so the initialization is correct even when the embedding matrix was padded.

    Args:
        embed_weight: The embedding weight matrix to initialize (shape: [vocab_size, embedding_dim])
        num_new_tokens: Number of new tokens added
        descriptions: Dict mapping token string to its description text
                      e.g., {"<think>": "A token representing reasoning process"}
        tokenizer: The tokenizer instance
        model: The model instance (used to get input embeddings)
        new_token_ids: IDs of all newly added tokens. Used to exclude not-yet-initialized
            rows when averaging description-token embeddings (robust to embedding padding).
        add_noise: Whether to add Gaussian noise to the initialization

    Example:
        descriptions = {
            "<|START_OF_SVG|>": "Marks the beginning of an SVG document",
            "<|END_OF_SVG|>": "Marks the end of an SVG document"
        }
    """
    embedding_dim = embed_weight.size(1)
    vocab_size = embed_weight.size(0)
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    device = embed_weight.device

    # The set of rows that are NOT yet initialized (the newly added tokens). Description
    # tokens that fall into this set must be excluded, otherwise we would average garbage.
    # `num_new_tokens` (the padded resize delta) is NOT a reliable boundary, so rely on
    # the explicit IDs, falling back to resolving them from the description keys.
    if new_token_ids is None:
        new_token_ids = _resolve_new_token_ids(descriptions.keys(), tokenizer, vocab_size)

    new_id_set = set(new_token_ids or [])
    fallback_embedding = _existing_embeddings(embed_weight, num_new_tokens, new_token_ids).mean(dim=0)

    for token_str, desc in descriptions.items():
        # Resolve token ID for correct placement (robust to embedding padding)
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        if token_id is None or token_id == unk_token_id or not (0 <= token_id < vocab_size):
            logger.warning_rank0(f"description: token '{token_str}' not found or out of range, skipping.")
            continue

        # Tokenize description text
        tokens = tokenizer(desc, return_tensors="pt", add_special_tokens=False)

        with torch.no_grad():
            token_ids = tokens["input_ids"][0].tolist()

            # Keep only description tokens that already have a meaningful embedding.
            valid_token_ids = [tid for tid in token_ids if tid not in new_id_set and 0 <= tid < vocab_size]

            if len(valid_token_ids) == 0:
                # Fallback: use mean of all existing embeddings
                logger.warning_rank0(
                    f"Description for token '{token_str}' contains no valid tokens. Using mean of existing embeddings."
                )
                base_embedding = fallback_embedding
            else:
                # Get embeddings of description tokens and average them
                valid_ids_tensor = torch.as_tensor(valid_token_ids, device=device, dtype=torch.long)
                token_embeds = model.get_input_embeddings()(valid_ids_tensor)
                base_embedding = token_embeds.mean(dim=0)

            # Add noise if requested (ensure correct device and dtype)
            if add_noise:
                noise = torch.randn_like(base_embedding) * (1.0 / math.sqrt(embedding_dim))
                embed_weight[token_id] = base_embedding + noise
            else:
                embed_weight[token_id] = base_embedding


def _initialize_embeddings(
    embed_weight: "torch.Tensor",
    num_new_tokens: int,
    init_method: str,
    new_special_tokens_config: Optional[dict],
    tokenizer: "PreTrainedTokenizer",
    model: "PreTrainedModel",
    new_token_ids: Optional[list[int]] = None,
) -> None:
    """Single source of truth for embedding initialization.

    This function selects the appropriate initialization method and applies it.

    Available methods (canonical names; legacy aliases in INIT_METHOD_ALIASES still accepted):
        - 'vocab_mean_noise' (default): μ + N(0, σ)            [legacy alias: 'noise_init']
        - 'noise': N(0, σ) (pure Gaussian noise, no base)
        - 'vocab_mean': μ (pure vocab mean, all new tokens identical)
        - 'description': Σ semantic init from token descriptions [legacy alias: 'desc_init']
        - 'description_noise': semantic init + Gaussian noise     [legacy alias: 'desc_init_w_noise']

    Args:
        embed_weight: The embedding weight matrix to initialize
        num_new_tokens: Number of new tokens added
        init_method: Initialization method (see above)
        new_special_tokens_config: Config dict with token descriptions (required for description methods)
        tokenizer: The tokenizer instance
        model: The model instance
        new_token_ids: Explicit IDs of the newly added tokens (robust to embedding padding).
            When ``None``, the init helpers fall back to the last ``num_new_tokens`` rows.
    """
    # Normalize legacy names so the dispatch below only deals with canonical names.
    init_method = INIT_METHOD_ALIASES.get(init_method, init_method)

    if init_method == "description" and new_special_tokens_config:
        logger.info_rank0("Using semantic initialization (description) for new special tokens")
        _description_initialization(
            embed_weight, num_new_tokens, new_special_tokens_config, tokenizer, model, new_token_ids, add_noise=False
        )
    elif init_method == "description_noise" and new_special_tokens_config:
        logger.info_rank0("Using semantic initialization with noise (description_noise) for new special tokens")
        _description_initialization(
            embed_weight, num_new_tokens, new_special_tokens_config, tokenizer, model, new_token_ids, add_noise=True
        )
    elif init_method == "noise":
        logger.info_rank0("Using noise initialization (N(0, σ)) for new special tokens")
        _noise_initialization(embed_weight, num_new_tokens, token_ids=new_token_ids)
    elif init_method == "vocab_mean":
        logger.info_rank0("Using vocab mean initialization (μ) for new special tokens")
        _vocab_mean_initialization(embed_weight, num_new_tokens, token_ids=new_token_ids)
    else:
        if init_method != "vocab_mean_noise":
            logger.warning_rank0(
                f"init_method='{init_method}' requires descriptions config, falling back to 'vocab_mean_noise'"
            )
        logger.info_rank0("Using noisy vocab mean initialization (vocab_mean_noise) for new special tokens")
        _vocab_mean_noise_initialization(embed_weight, num_new_tokens, token_ids=new_token_ids)


def resize_embedding_layer(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    new_tokens: Optional[Iterable[str]] = None,
    new_special_tokens_config: Optional[dict] = None,
    init_special_tokens: str = "vocab_mean_noise",
) -> None:
    r"""Resize token embeddings (when needed) and initialize the newly added tokens.

    Resizing and initialization are decoupled: even when the tokenizer vocab fits inside
    the model's existing (padded) embedding matrix and no resize is triggered, the newly
    added tokens still occupy uninitialized rows and must be initialized. We therefore
    resolve the explicit row IDs of ``new_tokens`` and always initialize those rows.

    Args:
        model: The model to resize
        tokenizer: The tokenizer (used to get target vocab size)
        new_tokens: Iterable of the newly added token strings. Used to locate the exact
            embedding rows to initialize, which is robust to pre-existing embedding padding.
        new_special_tokens_config: Optional dict with token descriptions for semantic initialization
        init_special_tokens: Initialization method ('vocab_mean', 'noise', 'vocab_mean_noise',
            'description', 'description_noise'; legacy aliases 'noise_init'/'desc_init'/'desc_init_w_noise'
            are also accepted)
    """
    if is_deepspeed_zero3_enabled():
        import deepspeed  # type: ignore

        params = [model.get_input_embeddings().weight]
        if model.get_output_embeddings() is not None and not model.config.tie_word_embeddings:
            params.append(model.get_output_embeddings().weight)

        context_maybe_zero3 = deepspeed.zero.GatheredParameters(params, modifier_rank=0)
    else:
        context_maybe_zero3 = nullcontext()

    current_embedding_size = get_embedding_vocab_size(model)
    needs_resize = len(tokenizer) > current_embedding_size

    if needs_resize:
        if getattr(model, "quantization_method", None):
            raise ValueError("Cannot resize embedding layers of a quantized model.")

        if not isinstance(model.get_output_embeddings(), torch.nn.Linear):
            raise ValueError("Current model does not support resizing embedding layers.")

        # mean_resizing=False preserves the original embedding distribution exactly.
        # HuggingFace's default mean_resizing=True re-samples new rows from the mean/covariance
        # of existing embeddings, which conflicts with our explicit initialization below.
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64, mean_resizing=False)

    with context_maybe_zero3:
        new_embedding_size = model.get_input_embeddings().weight.size(0)
        num_new_tokens = new_embedding_size - current_embedding_size

        # Resolve the exact rows of the new tokens. This works whether or not a resize was
        # triggered (e.g. tokens added into a model's pre-existing padding zone).
        new_token_ids = _resolve_new_token_ids(new_tokens, tokenizer, new_embedding_size)

        if num_new_tokens <= 0 and not new_token_ids:
            return

        if needs_resize:
            logger.info_rank0(
                f"Resizing embeddings: {current_embedding_size} -> {new_embedding_size} (+{num_new_tokens} tokens)"
            )
        else:
            logger.info_rank0(
                f"No resize needed (vocab fits in padded embedding {new_embedding_size}); "
                f"initializing {len(new_token_ids or [])} new token(s) in place."
            )

        # Initialize input embeddings
        _initialize_embeddings(
            model.get_input_embeddings().weight.data,
            num_new_tokens,
            init_special_tokens,
            new_special_tokens_config,
            tokenizer,
            model,
            new_token_ids=new_token_ids,
        )

        # Initialize output embeddings if not tied
        if model.get_output_embeddings() is not None and not model.config.tie_word_embeddings:
            _initialize_embeddings(
                model.get_output_embeddings().weight.data,
                num_new_tokens,
                init_special_tokens,
                new_special_tokens_config,
                tokenizer,
                model,
                new_token_ids=new_token_ids,
            )

    if needs_resize:
        model.config.vocab_size = new_embedding_size
        # Also update the nested text_config for VL models (e.g., Qwen2.5-VL, LLaVA),
        # otherwise config.vocab_size and config.text_config.vocab_size become inconsistent.
        if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "vocab_size"):
            model.config.text_config.vocab_size = new_embedding_size

        logger.info_rank0(f"Resized token embeddings from {current_embedding_size} to {new_embedding_size}.")


def _make_embedding_freeze_hook(orig_size: int, total_size: int):
    """Create a gradient hook that zeros the gradient of the original embedding rows.

    The hook keeps a lazily-built, per-(device, dtype) cached mask so it handles AMP
    dtype switches (fp32 <-> bf16/fp16) and device migrations without repeated allocation.

    It is robust to parameter-sharded setups (DeepSpeed ZeRO-3 / FSDP): per-row masking by
    absolute index is only valid when the hook receives the full ``[total_size, dim]``
    gradient. When the gradient arrives partitioned/flattened (a different shape), masking
    by row offset would be wrong (and could even crash on the broadcast), so the hook leaves
    the gradient untouched and warns once instead.
    """
    cached_masks: dict[tuple, "torch.Tensor"] = {}
    warned = False

    def hook(grad: "torch.Tensor") -> "torch.Tensor":
        nonlocal warned
        if grad.dim() != 2 or grad.size(0) != total_size:
            if not warned:
                logger.warning_rank0(
                    f"Skipping embedding freeze: received a gradient of shape {tuple(grad.shape)}, "
                    f"expected [{total_size}, *]. This usually means the embedding is sharded "
                    "(DeepSpeed ZeRO-3 / FSDP), where per-row freezing is not supported; the "
                    "original embeddings will also be updated."
                )
                warned = True

            return grad

        cache_key = (grad.device, grad.dtype)
        if cache_key not in cached_masks:
            # 0 for original rows, 1 for new rows; shape [vocab_size, 1] to broadcast.
            mask = torch.zeros(total_size, 1, device=grad.device, dtype=grad.dtype)
            mask[orig_size:] = 1.0
            cached_masks[cache_key] = mask

        return grad * cached_masks[cache_key]

    return hook


def apply_embedding_freeze(model: "PreTrainedModel", orig_vocab_size: int) -> None:
    r"""Freeze original embedding rows, training only the newly added tokens.

    Registers gradient hooks that zero out the gradients of the original embedding
    rows, so only the new token embeddings receive updates. This is useful for
    vocabulary expansion where the base embeddings should stay intact.

    The freeze is permanent for the lifetime of the process (hook handles are not
    returned). Output embeddings are handled automatically when tied (a single tensor
    hook on the shared weight catches both backward paths), or hooked separately when
    untied.

    Note:
        Per-row freezing requires the hook to see the full embedding gradient. Under
        DeepSpeed ZeRO-3 / FSDP the embedding parameter (and its gradient) is sharded,
        so the freeze cannot be applied reliably; in that case the hook detects the
        partitioned gradient shape, leaves it untouched (no crash) and warns once.

    Args:
        model: The model with already-resized embeddings.
        orig_vocab_size: The original vocabulary size before new tokens were added.
    """
    new_vocab_size = get_embedding_vocab_size(model)
    num_new_tokens = new_vocab_size - orig_vocab_size

    if num_new_tokens <= 0:
        logger.warning_rank0("No new tokens to train, skipping embedding freeze.")
        return

    if is_deepspeed_zero3_enabled():
        logger.warning_rank0(
            "freeze_original_embeddings may not take effect under DeepSpeed ZeRO-3: the embedding "
            "gradient is partitioned across ranks, so per-row freezing cannot be applied reliably. "
            "Disable ZeRO-3 parameter sharding (or this flag) to train only the new tokens."
        )

    input_embed = model.get_input_embeddings()
    input_embed.weight.register_hook(_make_embedding_freeze_hook(orig_vocab_size, new_vocab_size))
    logger.info_rank0(
        f"Registered gradient mask for input embeddings: frozen [0:{orig_vocab_size}], "
        f"trainable [{orig_vocab_size}:{new_vocab_size}] ({num_new_tokens} new tokens)."
    )

    output_embed = model.get_output_embeddings()
    if output_embed is not None:
        if model.config.tie_word_embeddings:
            logger.info_rank0("Output embeddings are tied; the gradient mask applies to both automatically.")
        else:
            output_embed.weight.register_hook(_make_embedding_freeze_hook(orig_vocab_size, new_vocab_size))
            logger.info_rank0(
                f"Registered gradient mask for output embeddings (untied): "
                f"frozen [0:{orig_vocab_size}], trainable [{orig_vocab_size}:{new_vocab_size}]."
            )
