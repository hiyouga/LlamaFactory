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

from types import MethodType
from typing import TYPE_CHECKING, Any

import torch
from peft import PeftModel
from transformers import GenerationMixin, PreTrainedModel, PreTrainedTokenizerBase
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled
from transformers.utils import is_torch_cuda_available, is_torch_npu_available

from ..extras import logging
from ..extras.misc import check_version, infer_optim_dtype
from ..extras.packages import is_transformers_version_greater_than
from .model_utils.attention import configure_attn_implementation, print_attn_implementation
from .model_utils.checkpointing import prepare_model_for_training
from .model_utils.embedding import resize_embedding_layer
from .model_utils.kv_cache import configure_kv_cache
from .model_utils.longlora import configure_longlora
from .model_utils.moe import add_z3_leaf_module, configure_moe
from .model_utils.quantization import configure_quantization
from .model_utils.rope import configure_rope
from .model_utils.valuehead import prepare_valuehead_model
from .model_utils.visual import autocast_projector_dtype, configure_visual_model


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer, ProcessorMixin
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import ModelArguments

if is_transformers_version_greater_than("4.57.0"):
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe


logger = logging.get_logger(__name__)


def patch_qwen3_omni_moe_thinker_text_sparse_moe_block():
    if is_transformers_version_greater_than("4.57.0") and not is_transformers_version_greater_than("4.58.0"):
        from .model_utils.moe import Qwen3OmniMoeThinkerTextSparseMoeBlock

        logger.warning_rank0(
            "You are using transformers with 4.x version, the Qwen3OmniMoeThinkerTextSparseMoeBlock will have some issues about deepspeed zero2 and fsdp2 training, so that we patched this model to avoid it. Transformers v5.0.0rc0 has fixed the issue, you can also try to update the transformers to using qwen3_omni. See more information on https://github.com/hiyouga/LLaMA-Factory/issues/9628."
        )

        modeling_qwen3_omni_moe.Qwen3OmniMoeThinkerTextSparseMoeBlock = Qwen3OmniMoeThinkerTextSparseMoeBlock


def _check_fla_dependencies() -> None:
    """Check that the FLA dependencies required for varlen GDN forwarding are available.

    Requires ``flash-linear-attention >= 0.4.1`` (which exposes the varlen
    ``causal_conv1d`` under ``fla.modules.convolution`` and the
    ``chunk_gated_delta_rule`` / ``fused_recurrent_gated_delta_rule`` kernels
    under ``fla.ops.gated_delta_rule``). Raises ``ImportError`` with an
    actionable message otherwise.
    """
    try:
        from fla.modules.convolution import causal_conv1d  # noqa: F401
        from fla.ops.gated_delta_rule import (  # noqa: F401
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )
    except ImportError as exc:
        raise ImportError(
            "Qwen3.5 packing-seq forwarding requires `flash-linear-attention>=0.4.1` "
            "(provides `fla.modules.convolution.causal_conv1d` and "
            "`fla.ops.gated_delta_rule.{chunk,fused_recurrent}_gated_delta_rule`). "
            "Please install/upgrade it."
        ) from exc


def patch_qwen3_5_forward_npu(model: "PreTrainedModel") -> None:
    """Patch for Qwen3.5 models on NPU by importing torch_npu to enable torch.cuda compatibility.

    On NPU, torch.cuda operations will fail unless torch_npu is imported.
    torch_npu provides compatibility layer that maps torch.cuda calls to NPU operations.

    Also replaces chunk_gated_delta_rule with NPU-compatible implementation.
    """
    import importlib.metadata

    if "Ascend910" not in torch.npu.get_device_name(0):
        logger.warning_rank0("Currently only 910B series NPUs are supported for the NPU GDN patch.")
        return

    try:
        importlib.metadata.version("triton_ascend")
    except importlib.metadata.PackageNotFoundError:
        logger.warning_rank0(
            "triton_ascend not installed, skipping NPU GDN patch. "
            "To enable it on NPU, reinstall Triton with the Ascend build: "
            "`pip uninstall -y triton && pip install -r requirements/triton_ascend.txt`. "
            "Note: triton and triton_ascend cannot coexist — triton must be uninstalled first."
        )
        return

    logger.info_rank0("triton_ascend detected for NPU compatibility.")

    from ..third_party.triton.chunk_gated_delta_rule import chunk_gated_delta_rule as npu_chunk_gated_delta_rule

    if model.config.architectures[0] == "Qwen3_5MoeForConditionalGeneration":
        try:
            # Qwen3.5-MoE structure: model.model.language_model.layers
            for layer in model.model.language_model.layers:
                if hasattr(layer, "linear_attn"):
                    layer.linear_attn.chunk_gated_delta_rule = npu_chunk_gated_delta_rule

            logger.info_rank0(
                "Replaced chunk_gated_delta_rule with NPU-compatible implementation for Qwen3.5-MoE model."
            )
        except Exception as e:
            logger.warning_rank0(f"Failed to replace chunk_gated_delta_rule for NPU: {e}")
    elif model.config.architectures[0] == "Qwen3_5ForConditionalGeneration":
        try:
            # Qwen3.5 structure: model.model.layers
            for layer in model.model.layers:
                if hasattr(layer, "linear_attn"):
                    layer.linear_attn.chunk_gated_delta_rule = npu_chunk_gated_delta_rule

            logger.info_rank0("Replaced chunk_gated_delta_rule with NPU-compatible implementation for Qwen3.5 model.")
        except Exception as e:
            logger.warning_rank0(f"Failed to replace chunk_gated_delta_rule for NPU: {e}")


def patch_qwen3_5_forward_gpu(model: "PreTrainedModel") -> None:
    """Patch Qwen3.5 decoder forward methods to support cu_seqlens during training.

    Refer to: https://github.com/axolotl-ai-cloud/axolotl/blob/main/src/axolotl/monkeypatch/models/qwen3_5/modeling.py.
    """
    if is_transformers_version_greater_than("5.2.0"):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states

    from torch.nn import functional as F
    from transformers.modeling_flash_attention_utils import prepare_fa_kwargs_from_position_ids

    _check_fla_dependencies()
    from fla.modules.convolution import causal_conv1d as fla_causal_conv1d
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    def _patched_decoder_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """Decoder layer forward that passes position_ids through to linear attention."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
                position_ids=position_ids,  # passing position_ids to linear attention
            )
        elif self.layer_type == "full_attention":
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids[None, 0],  # keep [1, B, L]
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):  # MoE returns (hidden_states, router_logits)
            hidden_states, _ = hidden_states

        hidden_states = residual + hidden_states

        return hidden_states

    # gdn forward (training only, cache_params is always None)
    def _patch_gdn_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ):
        # @kuangdd fix: here attention_mask is None
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        batch_size, seq_len, _ = hidden_states.shape

        # Qwen3.5 VL passes 3-D MRoPE position_ids ([axes, B, T]); collapse to [B, T].
        if position_ids is not None and position_ids.ndim == 3:
            position_ids = position_ids[0]

        # cu_seqlens for the FLA varlen path is only needed when batch_size == 1:
        # packing / neat-packing: always folded into a single sequence (bsz == 1) -> varlen
        # non-packing, bsz == 1: single segment, equivalent to a standard single sequence
        # non-packing, bsz > 1: not packed, use cu_seqlens=None and standard batched kernels
        if position_ids is not None and batch_size == 1:
            cu_seqlens = prepare_fa_kwargs_from_position_ids(position_ids)[0][0]
        else:
            cu_seqlens = None

        # FLA varlen kernels expect [B, T, D] layout, not [B, D, T] like the
        # standard causal-conv1d path that the upstream forward uses.
        mixed_qkv = self.in_proj_qkv(hidden_states)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # FLA's causal_conv1d returns (out, final_state); we don't use the state here.
        mixed_qkv, _ = fla_causal_conv1d(
            x=mixed_qkv,
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
            cu_seqlens=cu_seqlens,
        )

        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        core_attn_out, _ = chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            **({"cu_seqlens": cu_seqlens} if cu_seqlens is not None else {}),
        )

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)

        return output

    if model.config.architectures[0] in ("Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration"):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5GatedDeltaNet

        Qwen3_5DecoderLayer.forward = _patched_decoder_forward
        Qwen3_5GatedDeltaNet.forward = _patch_gdn_forward
    elif model.config.architectures[0] in ("Qwen3_5MoeForCausalLM", "Qwen3_5MoeForConditionalGeneration"):
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeDecoderLayer,
            Qwen3_5MoeGatedDeltaNet,
        )

        Qwen3_5MoeDecoderLayer.forward = _patched_decoder_forward
        Qwen3_5MoeGatedDeltaNet.forward = _patch_gdn_forward

    logger.info_rank0("Patched Qwen3.5 decoder forward to support cu_seqlens input only patch when do training.")


def patch_youtu_vl_model(model: "PreTrainedModel") -> None:
    original_forward = model.forward

    def forward(self, *args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        if "loss" not in outputs and "labels" in kwargs:
            logits = outputs.get("logits")
            labels = kwargs.get("labels")
            if logits is not None and labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
                outputs["loss"] = loss

        return outputs

    model.forward = MethodType(forward, model)


def patch_tokenizer(tokenizer: "PreTrainedTokenizer", model_args: "ModelArguments") -> None:
    if "PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__):
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)

    if model_args.model_max_length is not None and tokenizer.model_max_length < model_args.model_max_length:
        tokenizer.model_max_length = model_args.model_max_length  # enlarge the tokenizer max length

    if model_args.add_tokens is not None:
        num_added_tokens = tokenizer.add_tokens(new_tokens=model_args.add_tokens, special_tokens=False)
        logger.info_rank0("Add tokens {} to tokenizer's vocabulary.".format(",".join(model_args.add_tokens)))
        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning_rank0("New tokens have been added, changed `resize_vocab` to True.")

    if model_args.add_special_tokens is not None:
        num_added_special_tokens = tokenizer.add_tokens(new_tokens=model_args.add_special_tokens, special_tokens=True)
        logger.info_rank0(
            "Add special tokens {} to tokenizer's vocabulary.".format(",".join(model_args.add_special_tokens))
        )
        if num_added_special_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning_rank0("New special tokens have been added, changed `resize_vocab` to True.")


def patch_processor(
    processor: "ProcessorMixin",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
) -> None:
    setattr(processor, "tokenizer", tokenizer)
    setattr(processor, "image_max_pixels", model_args.image_max_pixels)
    setattr(processor, "image_min_pixels", model_args.image_min_pixels)
    setattr(processor, "image_do_pan_and_scan", model_args.image_do_pan_and_scan)
    setattr(processor, "crop_to_patches", model_args.crop_to_patches)
    setattr(processor, "video_max_pixels", model_args.video_max_pixels)
    setattr(processor, "video_min_pixels", model_args.video_min_pixels)
    setattr(processor, "video_fps", model_args.video_fps)
    setattr(processor, "video_maxlen", model_args.video_maxlen)
    setattr(processor, "use_audio_in_video", model_args.use_audio_in_video)
    setattr(processor, "audio_sampling_rate", model_args.audio_sampling_rate)


def patch_config(
    config: "PretrainedConfig",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    init_kwargs: dict[str, Any],
    is_trainable: bool,
) -> None:
    if model_args.compute_dtype is None:  # priority: bf16 > fp16 > fp32
        if model_args.infer_dtype != "auto" and not is_trainable:
            model_args.compute_dtype = getattr(torch, model_args.infer_dtype)
        else:
            model_args.compute_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))

    configure_attn_implementation(config, model_args)
    configure_rope(config, model_args)
    configure_longlora(config, model_args, is_trainable)
    configure_quantization(config, tokenizer, model_args, is_trainable, init_kwargs)
    configure_moe(config, model_args, is_trainable)
    configure_visual_model(config)
    configure_kv_cache(config, model_args, is_trainable)

    if getattr(config, "model_type", None) == "qwen":
        setattr(config, "use_flash_attn", model_args.flash_attn == "fa2")
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            setattr(config, dtype_name, model_args.compute_dtype == dtype)

    if getattr(config, "model_type", None) == "minicpmo":
        setattr(config, "init_audio", True)
        setattr(config, "init_tts", False)

    # replace the top-k gating method
    if getattr(config, "model_type", None) == "kimi_vl" and is_trainable:
        setattr(config.text_config, "topk_method", "greedy")

    architectures = getattr(config, "architectures", None)
    if isinstance(architectures, list) and "InternVLChatModel" in architectures:
        raise ValueError(
            "Please download the internvl models in a Hugging Face–compatible format "
            "(for example, https://huggingface.co/OpenGVLab/InternVL3-8B-hf)."
        )

    if isinstance(architectures, list) and "LlavaLlamaForCausalLM" in architectures:
        raise ValueError("Please download llava models with hf-compatible format: https://huggingface.co/llava-hf")

    if getattr(config, "model_type", None) == "internlm3" and not is_transformers_version_greater_than("4.47.1"):
        raise RuntimeError("InternLM3 model requires transformers>=4.47.1, please upgrade it.")

    if getattr(config, "model_type", None) == "lfm2_vl" and not is_transformers_version_greater_than("4.58.0"):
        raise RuntimeError(
            "LFM2.5-VL model requires transformers>=4.58.0 or install from commit: "
            "pip install git+https://github.com/huggingface/transformers.git@3c2517727ce28a30f5044e01663ee204deb1cdbe"
        )

    if getattr(config, "model_type", None) == "moss_vl":
        check_version("transformers==4.57.1", mandatory=True)
        check_version("torchcodec==0.7.0", mandatory=True)

    if getattr(config, "model_type", None) == "qwen3_omni_moe":
        patch_qwen3_omni_moe_thinker_text_sparse_moe_block()

    # deepspeed zero3 is not compatible with low_cpu_mem_usage
    init_kwargs["low_cpu_mem_usage"] = model_args.low_cpu_mem_usage and (not is_deepspeed_zero3_enabled())

    # fsdp/deepspeed zero3 does not need device map
    if not (is_deepspeed_zero3_enabled() or is_fsdp_enabled()) and init_kwargs["low_cpu_mem_usage"]:
        if "device_map" not in init_kwargs and model_args.device_map:
            init_kwargs["device_map"] = model_args.device_map  # device map requires low_cpu_mem_usage=True

        if init_kwargs.get("device_map", None) == "auto":
            init_kwargs["offload_folder"] = model_args.offload_folder


def patch_model(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    is_trainable: bool,
    add_valuehead: bool,
) -> None:
    gen_config = model.generation_config  # check and fix generation config
    if not gen_config.do_sample and (
        (gen_config.temperature is not None and gen_config.temperature != 1.0)
        or (gen_config.top_p is not None and gen_config.top_p != 1.0)
        or (gen_config.typical_p is not None and gen_config.typical_p != 1.0)
    ):
        gen_config.do_sample = True

    if getattr(model.config, "model_type", None) not in ["minicpmv", "minicpmo"] and "GenerationMixin" not in str(
        model.generate.__func__
    ):
        model.generate = MethodType(GenerationMixin.generate, model)

    if add_valuehead:
        prepare_valuehead_model(model)

    if model_args.resize_vocab:
        # Pass the explicit list of newly added tokens so their exact embedding rows can be
        # located and initialized, even when they land in a model's pre-existing padding zone.
        new_tokens = (model_args.add_tokens or []) + (model_args.add_special_tokens or [])

        resize_embedding_layer(
            model,
            tokenizer,
            new_tokens=new_tokens or None,
            new_special_tokens_config=getattr(model_args, "_special_token_descriptions", None),
            init_special_tokens=model_args.init_special_tokens,
        )

    if is_trainable:
        if getattr(model.config, "model_type", None) == "gemma3n":
            setattr(model_args, "disable_gradient_checkpointing", True)

        if getattr(model.config, "model_type", None) == "youtu_vl":
            patch_youtu_vl_model(model)

        prepare_model_for_training(model, model_args)
        autocast_projector_dtype(model, model_args)
        add_z3_leaf_module(model)

        if getattr(model.config, "model_type", None) in [
            "qwen3_5",
            "qwen3_5_moe",
            "qwen3_5_moe_text",
            "qwen3_5_text",
        ]:
            if is_torch_npu_available():
                patch_qwen3_5_forward_npu(model)
            elif is_torch_cuda_available() and model_args.flash_attn == "fa2":
                # this is the patch for packing/neat_packing for GPU GDN. And when setting packing, flash_attn must be fa2.
                patch_qwen3_5_forward_gpu(model)

    if not model_args.use_unsloth:
        print_attn_implementation(model.config)

    try:
        model.add_model_tags(["llama-factory"])
    except Exception:
        logger.warning_rank0("Cannot properly tag the model.")


def patch_valuehead_model(model: "AutoModelForCausalLMWithValueHead") -> None:
    def tie_weights(self: "AutoModelForCausalLMWithValueHead") -> None:
        if isinstance(self.pretrained_model, PreTrainedModel):
            self.pretrained_model.tie_weights()

    def get_input_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_input_embeddings()

    def get_output_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_output_embeddings()

    def create_or_update_model_card(self: "AutoModelForCausalLMWithValueHead", output_dir: str) -> None:
        if isinstance(self.pretrained_model, PeftModel):
            self.pretrained_model.create_or_update_model_card(output_dir)

    def get_rope_index_func(self: "AutoModelForCausalLMWithValueHead"):
        if isinstance(self.pretrained_model, PeftModel):
            base_model = self.pretrained_model.base_model.model
        else:
            base_model = self.pretrained_model

        if base_model and hasattr(base_model, "get_rope_index"):
            return base_model.get_rope_index
        elif base_model and hasattr(base_model, "model") and hasattr(base_model.model, "get_rope_index"):
            return base_model.model.get_rope_index
        else:
            return None

    ignore_modules = [name for name, _ in model.named_parameters() if "pretrained_model" in name]
    setattr(model, "_keys_to_ignore_on_save", ignore_modules)
    setattr(model, "tie_weights", MethodType(tie_weights, model))
    setattr(model, "get_input_embeddings", MethodType(get_input_embeddings, model))
    setattr(model, "get_output_embeddings", MethodType(get_output_embeddings, model))
    setattr(model, "get_rope_index", get_rope_index_func(model))
    setattr(model, "create_or_update_model_card", MethodType(create_or_update_model_card, model))
