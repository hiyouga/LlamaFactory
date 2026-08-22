# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
from functools import partial
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, patch_accelerator_for_fp8, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        ref_model: Optional["torch.nn.Module"] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        # Configure FP8 environment if enabled
        training_args: TrainingArguments = kwargs.get("args")
        if training_args.fp8:
            configure_fp8_environment(training_args)
            if getattr(training_args, "fp8_backend", "auto") == "te":
                patch_accelerator_for_fp8()

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        self.ref_model = ref_model

        if ref_model is not None:
            from trl.models.utils import prepare_deepspeed, prepare_fsdp

            if getattr(self.accelerator.state, "deepspeed_plugin", None) is not None:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif getattr(self.accelerator.state, "fsdp_plugin", None) is not None:
                if self.accelerator.is_fsdp2:
                    from accelerate.utils.fsdp_utils import fsdp2_prepare_model

                    self.ref_model = fsdp2_prepare_model(self.accelerator, self.ref_model)
                else:
                    self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        elif finetuning_args.use_eaft_loss:
            from ..trainer_utils import eaft_loss_func

            self.compute_loss_func = lambda outputs, labels, num_items_in_batch=None: eaft_loss_func(
                outputs, labels, num_items_in_batch, finetuning_args.eaft_alpha
            )
        elif finetuning_args.use_asft_loss:
            from ..trainer_utils import asft_loss_func

            self.compute_loss_func = partial(
                asft_loss_func,
                asft_alpha=finetuning_args.asft_alpha,
            )

        if training_args.fp8 and hasattr(self, "accelerator"):  # verify FP8 status after trainer initialization
            verify_fp8_status(self.accelerator, training_args)

        self.context_parallel_group = None
        self.context_parallel_size = finetuning_args.context_parallel_size
        self.context_parallel_rank = 0
        if self.context_parallel_size > 1:
            import torch.distributed as dist
            from gemma_triton_flash_attn import register_triton_attention_ulysses

            from .context_parallel import create_context_parallel_group

            self.context_parallel_group = create_context_parallel_group(self.context_parallel_size)
            self.context_parallel_rank = dist.get_rank(self.context_parallel_group)

            text_config = (
                self.model.config.get_text_config()
                if hasattr(self.model.config, "get_text_config")
                else self.model.config
            )
            num_query_heads = text_config.num_attention_heads
            kv_head_counts = {
                text_config.num_key_value_heads,
                getattr(text_config, "num_global_key_value_heads", text_config.num_key_value_heads),
            }
            if num_query_heads % self.context_parallel_size != 0:
                raise ValueError(
                    f"num_attention_heads ({num_query_heads}) must be divisible by context_parallel_size "
                    f"({self.context_parallel_size})."
                )
            for num_kv_heads in kv_head_counts:
                if num_kv_heads >= self.context_parallel_size and num_kv_heads % self.context_parallel_size != 0:
                    raise ValueError(
                        f"num_key_value_heads ({num_kv_heads}) must divide or be divisible by "
                        f"context_parallel_size ({self.context_parallel_size})."
                    )

            register_triton_attention_ulysses(self.context_parallel_group)
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            if hasattr(unwrapped_model, "set_attn_implementation"):
                if hasattr(unwrapped_model.config, "text_config"):
                    unwrapped_model.set_attn_implementation({"text_config": "triton_gqa_ulysses"})
                else:
                    unwrapped_model.set_attn_implementation("triton_gqa_ulysses")
            else:
                if hasattr(unwrapped_model.config, "text_config"):
                    setattr(unwrapped_model.config.text_config, "_attn_implementation", "triton_gqa_ulysses")
                else:
                    setattr(unwrapped_model.config, "_attn_implementation", "triton_gqa_ulysses")

            # Enables Transformers' exact global-token loss normalization. The
            # CP override below removes the duplicate count from CP peers.
            self.model_accepts_loss_kwargs = True
            logger.info_rank0(
                f"Enabled Gemma 4 Ulysses context parallelism: cp_size={self.context_parallel_size}, "
                f"dp_size={dist.get_world_size() // self.context_parallel_size}."
            )

    @override
    def create_optimizer(self, *args, **kwargs) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer(*args, **kwargs)

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            sampler = torch.utils.data.SequentialSampler(self.train_dataset)
        else:
            sampler = super()._get_train_sampler(*args, **kwargs)

        if self.context_parallel_size > 1:
            from .context_parallel import ContextParallelSampler

            sampler = ContextParallelSampler(sampler, self.context_parallel_size)

        return sampler

    @override
    def _get_num_items_in_batch(self, batch_samples, device=None):
        if self.context_parallel_group is None:
            return super()._get_num_items_in_batch(batch_samples, device)

        import torch.distributed as dist

        from .context_parallel import make_shift_labels

        num_items = None
        for batch in batch_samples:
            try:
                labels = batch["labels"]
            except (KeyError, TypeError, IndexError):
                continue
            count = make_shift_labels(labels).ne(IGNORE_INDEX).sum()
            num_items = count if num_items is None else num_items + count

        if num_items is None:
            return None

        num_items = num_items.to(device if device is not None else self.args.device)
        cp_min_items = num_items.clone()
        cp_max_items = num_items.clone()
        dist.all_reduce(cp_min_items, op=dist.ReduceOp.MIN, group=self.context_parallel_group)
        dist.all_reduce(cp_max_items, op=dist.ReduceOp.MAX, group=self.context_parallel_group)
        if not torch.equal(cp_min_items, cp_max_items):
            raise RuntimeError("Ranks in a context-parallel group received different label counts.")

        dist.all_reduce(num_items, op=dist.ReduceOp.SUM)
        if num_items.remainder(self.context_parallel_size).item() != 0:
            raise RuntimeError("The full-world label count is not divisible by the context-parallel size.")

        return num_items // self.context_parallel_size

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        if self.context_parallel_group is not None:
            import torch.distributed as dist

            from .context_parallel import context_parallel_cross_entropy, make_shift_labels, split_sequence_inputs

            inputs = dict(inputs)
            if "position_ids" not in inputs:
                inputs["position_ids"] = (
                    torch.arange(inputs["input_ids"].shape[1], device=inputs["input_ids"].device)
                    .unsqueeze(0)
                    .expand(inputs["input_ids"].shape[0], -1)
                )
            inputs["shift_labels"] = make_shift_labels(inputs["labels"])
            inputs = split_sequence_inputs(inputs, self.context_parallel_size, self.context_parallel_rank)
            shift_labels = inputs.pop("shift_labels")
            inputs.pop("labels")
            outputs = model(**inputs)
            num_items_in_batch = kwargs.get("num_items_in_batch")
            if num_items_in_batch is None:
                raise RuntimeError("Context parallel loss requires `num_items_in_batch`.")

            loss = context_parallel_cross_entropy(
                outputs.logits,
                shift_labels,
                num_items_in_batch=num_items_in_batch,
                world_size=dist.get_world_size(),
            )
            if kwargs.get("return_outputs", False):
                return loss, outputs

            return loss

        if self.finetuning_args.use_asft_loss:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                )
                ref_logits = ref_outputs.logits
            outputs = model(**inputs)
            return self.compute_loss_func(outputs, inputs["labels"], ref_logits)
        else:
            return super().compute_loss(model, inputs, *args, **kwargs)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        input_ids_column = dataset["input_ids"]
        try:
            input_ids_list = input_ids_column.to_pylist()
        except AttributeError:
            input_ids_list = list(input_ids_column)

        decoded_inputs = self.processing_class.batch_decode(input_ids_list, skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
