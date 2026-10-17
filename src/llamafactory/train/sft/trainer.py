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

import glob
import hashlib
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
from ..trainer_utils import (
    _supervised_extent,
    cig_sft_credit_func,
    cig_sft_loss_func,
    create_custom_optimizer,
    create_custom_scheduler,
)


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
        elif finetuning_args.use_cig_sft_loss:
            if self.ref_model is None:
                raise ValueError("The CIG-SFT loss needs a frozen reference model, but `ref_model` is None.")

            self.cig_credit: dict[str, torch.Tensor] = {}
            self.cig_credit_path: str = finetuning_args.cig_credit_cache or os.path.join(
                training_args.output_dir, "cig_credit_cache.pt"
            )
            self.cig_credit_fingerprint: str = self._cig_fingerprint()

        if training_args.fp8 and hasattr(self, "accelerator"):  # verify FP8 status after trainer initialization
            verify_fp8_status(self.accelerator, training_args)

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
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def train(self, *args, **kwargs):
        if self.finetuning_args.use_cig_sft_loss:
            self._materialize_cig_credit()

        return super().train(*args, **kwargs)

    def _cig_fingerprint(self) -> str:
        """Identify the frozen reference model used to produce the credit cache."""
        digest = hashlib.sha1()
        reference_model = getattr(self.ref_model, "module", self.ref_model)
        reference_config = getattr(reference_model, "config", None)
        if reference_config is not None:
            digest.update(json.dumps(reference_config.to_dict(), sort_keys=True, default=str).encode())
        else:
            digest.update(type(reference_model).__qualname__.encode())
        digest.update(b"|")
        digest.update(f"cig_eps={self.finetuning_args.cig_eps}".encode())
        try:
            embedding_sum = reference_model.get_input_embeddings().weight.detach().float().sum().item()
            digest.update(str(embedding_sum).encode())
        except Exception:
            logger.warning_rank0("Failed to fingerprint the reference model weights; using the config only.")

        return digest.hexdigest()

    def _cig_keys(self, inputs: dict[str, Any]) -> list[str]:
        """Hash both tokenized views of each row independently of batch padding."""
        input_ids = inputs["input_ids"].detach().to("cpu", torch.int32)
        labels = inputs["labels"].detach().to("cpu", torch.int32)
        erased_input_ids = inputs["erased_input_ids"].detach().to("cpu", torch.int32)
        erased_labels = inputs["erased_labels"].detach().to("cpu", torch.int32)
        attention_mask = inputs.get("attention_mask")
        erased_attention_mask = inputs.get("erased_attention_mask")
        if (
            torch.is_tensor(attention_mask)
            and torch.is_tensor(erased_attention_mask)
            and attention_mask.dim() == 2
            and erased_attention_mask.dim() == 2
        ):
            valid_tokens = attention_mask.detach().to("cpu").bool()
            erased_valid_tokens = erased_attention_mask.detach().to("cpu").bool()
        else:
            valid_tokens = None
            erased_valid_tokens = None

        extents = _supervised_extent(labels).to("cpu")
        erased_extents = _supervised_extent(erased_labels).to("cpu")
        keys = []
        for index in range(input_ids.size(0)):
            digest = hashlib.sha1()
            if valid_tokens is None:
                input_row = input_ids[index, : int(extents[index])]
                label_row = labels[index, : int(extents[index])]
                erased_input_row = erased_input_ids[index, : int(erased_extents[index])]
                erased_label_row = erased_labels[index, : int(erased_extents[index])]
            else:
                input_row = input_ids[index][valid_tokens[index]]
                label_row = labels[index][valid_tokens[index]]
                erased_input_row = erased_input_ids[index][erased_valid_tokens[index]]
                erased_label_row = erased_labels[index][erased_valid_tokens[index]]

            digest.update(np.asarray(input_row).tobytes())
            digest.update(b"|")
            digest.update(np.asarray(label_row).tobytes())
            digest.update(b"|")
            digest.update(np.asarray(erased_input_row).tobytes())
            digest.update(b"|")
            digest.update(np.asarray(erased_label_row).tobytes())
            keys.append(digest.hexdigest())

        return keys

    def _cig_credits_of(self, inputs: dict[str, Any]) -> tuple[list["torch.Tensor"], int]:
        """Return cached credits, materializing missing rows with the frozen reference model."""
        keys = self._cig_keys(inputs)
        num_new = len({key for key in keys if key not in self.cig_credit})
        if num_new:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                )
                erased_outputs = self.ref_model(
                    input_ids=inputs["erased_input_ids"],
                    attention_mask=inputs.get("erased_attention_mask", None),
                )

            credits = cig_sft_credit_func(
                ref_outputs.logits,
                inputs["labels"],
                erased_outputs.logits,
                inputs["erased_labels"],
                cig_eps=self.finetuning_args.cig_eps,
            )
            for key, credit in zip(keys, credits):
                self.cig_credit[key] = credit

        return [self.cig_credit[key] for key in keys], num_new

    def _load_cig_credit(self) -> None:
        """Restore cache shards only when they were made with the same reference model."""
        paths = sorted(glob.glob(f"{self.cig_credit_path}.rank*.pt"))
        if not paths and os.path.exists(self.cig_credit_path):
            paths = [self.cig_credit_path]

        restored = 0
        for path in paths:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
            except Exception:
                logger.warning_rank0(f"Failed to read the CIG-SFT credit cache at {path}; ignoring it.")
                continue

            if payload.get("fingerprint") != self.cig_credit_fingerprint:
                logger.warning_rank0(f"The CIG-SFT credit cache at {path} came from another reference model.")
                continue

            self.cig_credit.update(payload.get("credits", {}))
            restored += len(payload.get("credits", {}))

        if restored:
            logger.info_rank0(f"Restored {restored} cached CIG-SFT credits from {self.cig_credit_path}.")

    def _save_cig_credit(self) -> None:
        """Persist one cache shard per process so distributed workers never race."""
        path = f"{self.cig_credit_path}.rank{self.accelerator.process_index}.pt"
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        temporary = f"{path}.tmp"
        torch.save({"fingerprint": self.cig_credit_fingerprint, "credits": self.cig_credit}, temporary)
        os.replace(temporary, path)

    def _materialize_cig_credit(self) -> None:
        """Read fixed credits once before optimization and store them for later steps/runs."""
        if self.ref_model is None:
            raise ValueError("The CIG-SFT loss needs a frozen reference model, but `ref_model` is None.")

        self._load_cig_credit()
        seen = 0
        num_new = 0
        for inputs in self.get_train_dataloader():
            if "erased_input_ids" not in inputs:
                raise ValueError("The CIG-SFT loss needs `cig_erased_prompt` or an erased prompt dataset column.")

            _, batch_new = self._cig_credits_of(inputs)
            seen += len(inputs["input_ids"])
            num_new += batch_new

        logger.info_rank0(f"Read CIG-SFT credits for {num_new} new and {seen - num_new} cached examples.")
        self._save_cig_credit()

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        if self.finetuning_args.use_cig_sft_loss:
            if not model.training:
                # CIG-SFT is a training objective. Evaluation reports the ordinary SFT loss and does not need the
                # auxiliary erased view or the reference-model credit cache.
                for key in ("erased_input_ids", "erased_attention_mask", "erased_labels"):
                    inputs.pop(key, None)

                return super().compute_loss(model, inputs, *args, **kwargs)

            if "erased_input_ids" not in inputs:
                raise ValueError(
                    "The CIG-SFT loss needs an erased view for every batch; set `cig_erased_prompt` or provide the "
                    "erased prompt dataset column."
                )

            credits, _ = self._cig_credits_of(inputs)
            for key in ("erased_input_ids", "erased_attention_mask", "erased_labels"):
                inputs.pop(key, None)

            outputs = model(**inputs)
            return cig_sft_loss_func(
                outputs,
                inputs["labels"],
                credits,
                cig_lambda=self.finetuning_args.cig_lambda,
                cig_max_beta=self.finetuning_args.cig_max_beta,
            )
        elif self.finetuning_args.use_asft_loss:
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
        # The erased view is only an auxiliary training input. Evaluation and
        # generation use the ordinary SFT cross entropy/model inputs.
        inputs.pop("erased_input_ids", None)
        inputs.pop("erased_attention_mask", None)
        inputs.pop("erased_labels", None)
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
