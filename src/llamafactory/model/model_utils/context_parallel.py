# Copyright 2026 the LlamaFactory team.
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

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from torch.distributed import ProcessGroup
    from transformers import PreTrainedModel

    from ...hparams import FinetuningArguments


def _configure_gemma4_context_parallelism(
    model: "PreTrainedModel",
    context_parallel_group: "ProcessGroup",
    context_parallel_size: int,
    finetuning_args: "FinetuningArguments",
) -> None:
    config = model.config
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    if getattr(text_config, "_attn_implementation", None) != "triton_gqa":
        raise ValueError("Gemma 4 context parallelism requires `flash_attn: triton_gqa`.")

    if getattr(config, "model_type", None) == "gemma4" and (
        not finetuning_args.freeze_vision_tower or not finetuning_args.freeze_multi_modal_projector
    ):
        raise ValueError(
            "Gemma 4 context parallelism currently requires `freeze_vision_tower: true` and "
            "`freeze_multi_modal_projector: true` for text-only SFT."
        )

    num_query_heads = text_config.num_attention_heads
    if num_query_heads % context_parallel_size != 0:
        raise ValueError(
            f"num_attention_heads ({num_query_heads}) must be divisible by ulysses_context_parallel_size "
            f"({context_parallel_size})."
        )

    kv_head_counts = {
        text_config.num_key_value_heads,
        getattr(text_config, "num_global_key_value_heads", text_config.num_key_value_heads),
    }
    for num_kv_heads in kv_head_counts:
        if num_kv_heads >= context_parallel_size and num_kv_heads % context_parallel_size != 0:
            raise ValueError(
                f"num_key_value_heads ({num_kv_heads}) must divide or be divisible by "
                f"ulysses_context_parallel_size ({context_parallel_size})."
            )

    from gemma_triton_flash_attn import register_triton_attention_ulysses

    register_triton_attention_ulysses(context_parallel_group)
    if hasattr(model, "set_attn_implementation"):
        if hasattr(config, "text_config"):
            model.set_attn_implementation({"text_config": "triton_gqa_ulysses"})
        else:
            model.set_attn_implementation("triton_gqa_ulysses")
    elif hasattr(config, "text_config"):
        setattr(config.text_config, "_attn_implementation", "triton_gqa_ulysses")
    else:
        setattr(config, "_attn_implementation", "triton_gqa_ulysses")


_CONTEXT_PARALLEL_CONFIGURERS: dict[str, Any] = {
    "gemma4": _configure_gemma4_context_parallelism,
    "gemma4_text": _configure_gemma4_context_parallelism,
}


def configure_context_parallel_model(
    model: "PreTrainedModel",
    context_parallel_group: "ProcessGroup",
    context_parallel_size: int,
    finetuning_args: "FinetuningArguments",
) -> None:
    """Configure model-specific attention for the selected CP backend."""
    model_type = getattr(model.config, "model_type", None)
    configurer = _CONTEXT_PARALLEL_CONFIGURERS.get(model_type)
    if configurer is None:
        supported_models = ", ".join(sorted(_CONTEXT_PARALLEL_CONFIGURERS))
        raise ValueError(
            f"Context parallelism does not support model type `{model_type}`. Choose from {supported_models}."
        )

    configurer(model, context_parallel_group, context_parallel_size, finetuning_args)
