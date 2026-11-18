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
from contextlib import contextmanager
from pathlib import Path
from shutil import copy2

import torch
from transformers import AutoModel
from transformers.integrations.deepspeed import (
    is_deepspeed_zero3_enabled,
    set_hf_deepspeed_config,
    unset_hf_deepspeed_config,
)

from ...extras.misc import check_version, is_env_enabled
from ...extras.ploting import plot_loss
from ...model import load_tokenizer
from ...model.loader import _get_init_kwargs, load_config
from .data import DecisionDataCollator, load_decision_dataset
from .trainer import DecisionTrainer


class DecisionProcessor:
    """Preserve native processor assets if AutoProcessor falls back to a tokenizer."""

    def __init__(self, source):
        self.source = Path(source)

    def save_pretrained(self, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
            source = self.source / name
            if source.is_file():
                copy2(source, Path(output_dir) / name)


@contextmanager
def native_load_context(training_args):
    """Defer ZeRO-3 partitioning until after GS1's native meta/assign loader."""
    zero3 = is_deepspeed_zero3_enabled()
    if zero3:
        unset_hf_deepspeed_config()
    try:
        yield
    finally:
        if zero3:
            set_hf_deepspeed_config(training_args.deepspeed_plugin.hf_ds_config)


def validate_decision_args(model_args, data_args, training_args, finetuning_args):
    """Reject settings whose token/generation semantics do not apply to decisions."""
    unsupported = {
        "finetuning_type (use full)": finetuning_args.finetuning_type != "full",
        "val_size (use eval_dataset)": data_args.val_size > 0,
        "mix_strategy (use concat)": data_args.mix_strategy != "concat",
        "template (GS1 uses its native compiler)": data_args.template,
        "compute_accuracy (token metric)": finetuning_args.compute_accuracy,
    }
    for args, names in (
        (data_args, "packing streaming tokenized_path"),
        (training_args, "do_predict predict_with_generate fp8 dataloader_drop_last"),
        (
            model_args,
            "quantization_bit adapter_name_or_path resize_vocab use_unsloth use_kt enable_liger_kernel "
            "use_unsloth_gc shift_attn mixture_of_depths rope_scaling",
        ),
        (finetuning_args, "pure_bf16 use_hyper_parallel use_mca use_dft_loss use_asft_loss use_eaft_loss"),
    ):
        unsupported.update({name: getattr(args, name) for name in names.split()})
    enabled = [name for name, value in unsupported.items() if value]
    if enabled:
        raise ValueError(f"The decision stage does not support: {', '.join(enabled)}.")
    if finetuning_args.decision_warmup and finetuning_args.decision_brier_weight != 0:
        raise ValueError("decision_warmup requires decision_brier_weight=0 (NLL only).")
    if training_args.optim not in ("adamw_torch", "adamw_torch_fused"):
        raise ValueError("The decision recipe requires the standard torch AdamW optimizer.")
    if not math.isfinite(finetuning_args.decision_brier_weight) or finetuning_args.decision_brier_weight < 0:
        raise ValueError("decision_brier_weight must be finite and nonnegative.")


def configure_decision_model(model, config, training_args, finetuning_args):
    if finetuning_args.decision_warmup:
        token_ids = list(config.structural_token_ids.values())
        if len(token_ids) != 7:
            raise ValueError("GS1 warmup requires seven structural token IDs.")

        # Mask embedding-output gradients rather than sharded parameter gradients.
        # With zero embedding decay this is the native seven-row additive update.
        def mask_structural_rows(module, inputs, output):
            if output.requires_grad:
                mask = torch.isin(inputs[0], inputs[0].new_tensor(token_ids)).unsqueeze(-1)
                output.register_hook(lambda gradient: gradient * mask)

        model.get_input_embeddings().register_forward_hook(mask_structural_rows)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            not (
                name.startswith("calibrator.")
                or (finetuning_args.freeze_vision_tower and name.startswith("visual."))
                or (finetuning_args.freeze_language_model and name.startswith("language_model."))
            )
        )
        if finetuning_args.decision_warmup:
            parameter.requires_grad_(
                name.startswith("decision_head.")
                or ".attnres_routers." in name
                or name == "language_model.embed_tokens.weight"
            )
        # Match full tuning's FP32 trainable weights; ZeRO-3 manages its own masters.
        if training_args.do_train and parameter.requires_grad and not is_deepspeed_zero3_enabled():
            parameter.data = parameter.data.float()


def run_decision(model_args, data_args, training_args, finetuning_args, callbacks=None):
    validate_decision_args(model_args, data_args, training_args, finetuning_args)
    check_version("transformers>=5.3.0", mandatory=True)
    check_version("typesafe-sdk==0.7.0", mandatory=True)
    config = load_config(model_args)
    if config.model_type != "gs1":
        raise ValueError("The decision stage currently supports native GS1 checkpoints only.")
    tokenizer_module = load_tokenizer(model_args)
    # Native GS1 loads a strict CPU state dict via meta initialization. Defer
    # ZeRO partitioning to Trainer instead of nesting zero.Init inside it.
    with native_load_context(training_args):
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, **_get_init_kwargs(model_args))
    if training_args.fsdp or is_env_enabled("ACCELERATE_USE_FSDP"):
        # FSDP1 flattens the root head/vision parameters together. Native GS1
        # checkpoints mix BF16 backbone and FP32 head weights.
        model.float()
    tokenizer_module["tokenizer"].padding_side = "left"
    compiler = model.decision_compiler(tokenizer_module["tokenizer"], max_context=data_args.cutoff_len)
    if tokenizer_module["processor"] is None:
        tokenizer_module["processor"] = DecisionProcessor(compiler.processor)
    if is_deepspeed_zero3_enabled():
        from deepspeed.utils import set_z3_leaf_modules

        # Gather at native call boundaries: length cohorts vary per rank, and
        # frozen GDN weights must stay materialized through checkpoint replay.
        set_z3_leaf_modules(model, [type(model.language_model), type(model.decision_head)])
    configure_decision_model(model, config, training_args, finetuning_args)
    dataset_module = {}
    # Keep splits separate at record level; a record may contain multiple questions.
    if training_args.do_train:
        dataset_module["train_dataset"] = load_decision_dataset(data_args.dataset, data_args, compiler)
    if training_args.do_eval:
        dataset_module["eval_dataset"] = load_decision_dataset(data_args.eval_dataset, data_args, compiler)
    if training_args.do_train:
        # Complete only the last global update with zero-loss rows, avoiding
        # Accelerate's repeated real examples and preserving one visit per question.
        dataset = dataset_module["train_dataset"]
        batch_size = (
            training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
        )
        if dataset:
            dataset.extend([{**dataset[0], "weight": 0.0}] * (-len(dataset) % batch_size))
    trainer = DecisionTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=DecisionDataCollator(tokenizer=compiler.tokenizer),
        callbacks=callbacks,
        **tokenizer_module,
        **dataset_module,
    )
    if training_args.do_train:
        result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        if trainer.is_fsdp_enabled:
            # Keep sharded training checkpoints, but export a native final model.
            trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        trainer.save_model()
        trainer.log_metrics("train", result.metrics)
        trainer.save_metrics("train", result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"] if training_args.do_eval else ["loss"])
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
