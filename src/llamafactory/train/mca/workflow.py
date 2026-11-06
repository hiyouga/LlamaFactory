# Copyright 2025 the ROLL team and the LlamaFactory team.
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

import functools
import json
import os
from collections.abc import Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional

import torch
from transformers import DataCollatorForSeq2Seq

from ...data import (
    SFTDataCollatorWith4DAttentionMask,
    get_dataset,
    get_template_and_fix_tokenizer,
)
from ...data.collator import (
    PairwiseDataCollatorWithPadding,
)
from ...extras.constants import IGNORE_INDEX, MCA_SUPPORTED_MODELS
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps
from ...extras.packages import is_mcore_adapter_available
from ...extras.ploting import plot_loss
from ...model import load_tokenizer
from ..callbacks import SaveProcessorCallback


if not is_mcore_adapter_available():
    raise ImportError("mcore_adapter is not installed. Please install it with `pip install mcore-adapter`.")

from mcore_adapter.models import AutoConfig, AutoModel
from mcore_adapter.trainer import DPOTrainer as McaDPOTrainer
from mcore_adapter.trainer.dpo_config import DPOConfig

from .trainer import CustomMcaTrainer


if TYPE_CHECKING:
    from mcore_adapter.training_args import Seq2SeqTrainingArguments as McaSeq2SeqTrainingArguments
    from transformers import TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


logger = get_logger(__name__)


def _data_collator_wrapper(data_collator: Any):
    @functools.wraps(data_collator)
    def wrapper(features: Sequence[dict[str, Any]]):
        labels_key = [k for k in features[0].keys() if k.endswith("labels")]
        input_ids_key = [k for k in features[0].keys() if k.endswith("input_ids")]
        for feature in features:
            if len(labels_key) == 0:  # pt
                feature["labels"] = deepcopy(feature["input_ids"])[1:]
            for k in labels_key:
                feature[k] = feature[k][1:]
            for k in input_ids_key:
                feature[k] = feature[k][:-1]
            for k in ["attention_mask", "position_ids"]:
                if k in feature:
                    feature[k] = feature[k][:-1]

        # for qwen vl series model
        tmp_features = data_collator(features)
        tmp_features.pop("rope_deltas", None)
        position_ids = tmp_features.get("position_ids", None)

        if position_ids is not None and position_ids.dim() == 3:
            if position_ids.shape[0] == 4:
                position_ids = position_ids[1:]
            tmp_features["position_ids"] = position_ids

        return tmp_features

    return wrapper


def _check_model_support(model_args: "ModelArguments"):
    from transformers import AutoConfig as HfAutoConfig

    if os.path.exists(os.path.join(model_args.model_name_or_path, "mca_config.json")):  # load from mcore ckpt
        mca_config = json.load(open(os.path.join(model_args.model_name_or_path, "mca_config.json")))
        model_type = mca_config.get("hf_model_type", None)
    else:
        config = HfAutoConfig.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
        model_type = config.model_type

    if model_type not in MCA_SUPPORTED_MODELS:
        raise ValueError(
            f"Model {model_type} is not supported by mcore_adapter."
            "You can try to upgrade mcore_adapter to the latest version for more supported models."
        )


def _freeze_model_parameters(model: Any, finetuning_args: "FinetuningArguments"):
    """Freeze model parameters for qwen_vl series models based on finetuning arguments."""
    if getattr(model.config, "hf_model_type", None) not in [
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3_vl",
        "qwen3_vl_moe",
        "qwen3_5",
        "qwen3_5_moe",
    ]:
        return

    params_to_freeze = []
    if finetuning_args.freeze_vision_tower:
        params_to_freeze.extend(["vision_model.blocks", "vision_model.patch_embed"])
        if getattr(model.config, "hf_model_type", None) in ["qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"]:
            params_to_freeze.extend(["vision_model.pos_embed"])

    if finetuning_args.freeze_multi_modal_projector:
        params_to_freeze.extend(["vision_model.merger"])

    if finetuning_args.freeze_language_model:
        params_to_freeze.extend(["embedding", "decoder", "output_layer"])

    if params_to_freeze:
        for name, p in model.named_parameters():
            if any(name.startswith(k) for k in params_to_freeze):
                p.requires_grad_(False)


def _build_meta_hf_model_for_collator(model_args: "ModelArguments") -> Any | None:
    r"""Build a lightweight HF model on meta device for compatibility with collator."""
    from transformers import AutoConfig as HfAutoConfig
    from transformers import AutoModel as HfAutoModel
    from transformers import AutoModelForImageTextToText

    try:
        config = HfAutoConfig.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
        with torch.device("meta"):
            try:
                # Prefer multimodal auto class for VLMs (e.g. qwen2-vl), so get_rope_index is available.
                return AutoModelForImageTextToText.from_config(config)
            except Exception:
                return HfAutoModel.from_config(config)
    except Exception as exc:
        logger.warning("Failed to build meta HF model for collator, fallback to no model. Error: %s", exc)
        return None


def run_pt(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "McaSeq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # dataset needs +1 then cut back due to MCA shift logic
    data_args.cutoff_len += 1
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="pt", **tokenizer_module)
    data_args.cutoff_len -= 1

    _check_model_support(model_args)
    model = AutoModel.from_pretrained(model_args.model_name_or_path, training_args)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX,
    )
    data_collator = _data_collator_wrapper(data_collator)

    trainer = CustomMcaTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
    )

    if "processor" in tokenizer_module and tokenizer_module["processor"] is not None:
        trainer.add_callback(SaveProcessorCallback(tokenizer_module["processor"]))

    if training_args.do_train:
        train_result = trainer.train(training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += [f"eval_{key}_loss" for key in dataset_module["eval_dataset"].keys()]
            else:
                keys += ["eval_loss"]
            plot_loss(training_args.output_dir, keys=keys)


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "McaSeq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    # align packing flags
    # TODO: FIX SequencePacking
    data_args.neat_packing = training_args.sequence_packing = data_args.neat_packing or training_args.sequence_packing
    data_args.packing = data_args.neat_packing or data_args.packing

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # dataset needs +1 then cut back due to MCA shift logic
    data_args.cutoff_len += 1
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    data_args.cutoff_len -= 1

    _check_model_support(model_args)
    model = AutoModel.from_pretrained(model_args.model_name_or_path, training_args)
    collator_model = _build_meta_hf_model_for_collator(model_args)

    # optional freezing for qwen_vl series
    _freeze_model_parameters(model, finetuning_args)

    pad_to_max = training_args.expert_model_parallel_size is not None and training_args.expert_model_parallel_size > 1
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=collator_model,
        padding="max_length" if pad_to_max else "longest",
        max_length=data_args.cutoff_len if pad_to_max else None,
        pad_to_multiple_of=64,
        label_pad_token_id=IGNORE_INDEX,
        **tokenizer_module,
    )
    data_collator = _data_collator_wrapper(data_collator)

    trainer = CustomMcaTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
    )

    if "processor" in tokenizer_module and tokenizer_module["processor"] is not None:
        trainer.add_callback(SaveProcessorCallback(tokenizer_module["processor"]))

    train_result = trainer.train(training_args.resume_from_checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    if trainer.is_world_process_zero() and finetuning_args.plot_loss:
        keys = ["loss"]
        if isinstance(dataset_module.get("eval_dataset"), dict):
            keys += [f"eval_{key}_loss" for key in dataset_module["eval_dataset"].keys()]
        else:
            keys += ["eval_loss"]
        plot_loss(training_args.output_dir, keys=keys)


def run_dpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "McaSeq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    _check_model_support(model_args)
    model = AutoModel.from_pretrained(model_args.model_name_or_path, training_args)
    collator_model = _build_meta_hf_model_for_collator(model_args)

    _freeze_model_parameters(model, finetuning_args)

    if finetuning_args.use_ref_model:
        ref_config = AutoConfig.from_pretrained(model_args.model_name_or_path, training_args)
        ref_model = AutoModel.from_config(ref_config)
        ref_model.load_state_dict(model.state_dict())
    else:
        ref_model = None

    # dataset needs +1 then cut back due to MCA shift logic
    data_args.cutoff_len += 1
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)
    data_args.cutoff_len -= 1

    pad_to_max = training_args.expert_model_parallel_size is not None and training_args.expert_model_parallel_size > 1
    dpo_config = DPOConfig(
        beta=finetuning_args.pref_beta,
        pref_loss=finetuning_args.pref_loss,
        label_smoothing=finetuning_args.dpo_label_smoothing,
    )
    data_collator = PairwiseDataCollatorWithPadding(
        template=template,
        model=collator_model,
        pad_to_multiple_of=64,
        padding="max_length" if pad_to_max else "longest",
        max_length=data_args.cutoff_len if pad_to_max else None,
        label_pad_token_id=IGNORE_INDEX,
        **tokenizer_module,
    )
    data_collator = _data_collator_wrapper(data_collator)

    trainer = McaDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_config=dpo_config,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
    )

    if "processor" in tokenizer_module and tokenizer_module["processor"] is not None:
        trainer.add_callback(SaveProcessorCallback(tokenizer_module["processor"]))

    train_result = trainer.train(training_args.resume_from_checkpoint)
    trainer.save_model()
    if finetuning_args.include_effective_tokens_per_second:
        train_dataset = dataset_module["train_dataset"]
        train_result.metrics.update(calculate_tps(train_dataset, train_result.metrics, stage="rm"))

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    if trainer.is_world_process_zero() and finetuning_args.plot_loss:
        keys = ["loss", "rewards/accuracies"]
        if isinstance(dataset_module.get("eval_dataset"), dict):
            keys += [f"eval_{key}_loss" for key in dataset_module["eval_dataset"].keys()]
        else:
            keys += ["eval_loss"]

        plot_loss(training_args.output_dir, keys=keys)
