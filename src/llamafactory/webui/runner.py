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

import json
import os
from collections.abc import Generator
from copy import deepcopy
from subprocess import Popen, TimeoutExpired
from typing import TYPE_CHECKING, Any

from transformers.utils import is_torch_npu_available

from ..extras.constants import LLAMABOARD_CONFIG, MULTIMODAL_SUPPORTED_MODELS, PEFT_METHODS, TRAINING_STAGES
from ..extras.misc import is_accelerator_available, torch_gc
from ..extras.packages import is_gradio_available
from .common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CONFIG_DIR,
    abort_process,
    calculate_pixels,
    gen_cmd,
    get_save_dir,
    load_args,
    load_config,
    load_eval_results,
    save_args,
    save_cmd,
)
from .control import get_trainer_info
from .locales import ALERTS, LOCALES


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

    from .manager import Manager


def _parse_seed(value: Any, default: int = 42) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class Runner:
    r"""A class to manage the running status of the trainers."""

    def __init__(self, manager: "Manager", demo_mode: bool = False) -> None:
        r"""Init a runner."""
        self.manager = manager
        self.demo_mode = demo_mode
        """ Resume """
        self.trainer: Popen | None = None
        self.do_train = True
        self.running_data: dict[Component, Any] = None
        """ State """
        self.aborted = False
        self.running = False

    def set_abort(self) -> None:
        self.aborted = True
        if self.trainer is not None:
            abort_process(self.trainer.pid)

    def _initialize(self, data: dict["Component", Any], do_train: bool, from_preview: bool) -> str:
        r"""Validate the configuration."""
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
        lang, model_name, model_path = get("top.lang"), get("top.model_name"), get("top.model_path")
        dataset = get("train.dataset") if do_train else get("eval.dataset")

        if self.running:
            return ALERTS["err_conflict"][lang]

        if not model_name:
            return ALERTS["err_no_model"][lang]

        if not model_path:
            return ALERTS["err_no_path"][lang]

        if not dataset:
            return ALERTS["err_no_dataset"][lang]

        if not from_preview and self.demo_mode:
            return ALERTS["err_demo"][lang]

        if do_train:
            if not get("train.output_dir"):
                return ALERTS["err_no_output_dir"][lang]

            try:
                json.loads(get("train.extra_args"))
            except json.JSONDecodeError:
                return ALERTS["err_json_schema"][lang]

            stage = TRAINING_STAGES[get("train.training_stage")]
            if stage == "ppo" and not get("train.reward_model"):
                return ALERTS["err_no_reward_model"][lang]
        else:
            if not get("eval.output_dir"):
                return ALERTS["err_no_output_dir"][lang]

        if not from_preview and not is_accelerator_available():
            gr.Warning(ALERTS["warn_no_cuda"][lang])

        return ""

    def _finalize(self, lang: str, finish_info: str) -> None:
        r"""Clean the cached memory and resets the runner."""
        finish_info = ALERTS["info_aborted"][lang] if self.aborted else finish_info
        gr.Info(finish_info)
        self.trainer = None
        self.aborted = False
        self.running = False
        self.running_data = None
        torch_gc()

    def _parse_train_args(self, data: dict["Component", Any]) -> dict[str, Any]:
        r"""Build and validate the training arguments."""
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
        model_name, finetuning_type = get("top.model_name"), get("top.finetuning_type")
        user_config = load_config()

        args = dict(
            stage=TRAINING_STAGES[get("train.training_stage")],
            do_train=True,
            model_name_or_path=get("top.model_path"),
            cache_dir=user_config.get("cache_dir", None),
            preprocessing_num_workers=16,
            finetuning_type=finetuning_type,
            template=get("top.template"),
            rope_scaling=get("top.rope_scaling") if get("top.rope_scaling") != "none" else None,
            flash_attn="fa2" if get("top.booster") == "flashattn2" else "auto",
            use_unsloth=(get("top.booster") == "unsloth"),
            enable_liger_kernel=(get("top.booster") == "liger_kernel"),
            dataset_dir=get("train.dataset_dir"),
            dataset=",".join(get("train.dataset")),
            cutoff_len=get("train.cutoff_len"),
            learning_rate=float(get("train.learning_rate")),
            num_train_epochs=float(get("train.num_train_epochs")),
            max_samples=int(get("train.max_samples")),
            seed=_parse_seed(get("train.train_seed")),
            per_device_train_batch_size=get("train.batch_size"),
            gradient_accumulation_steps=get("train.gradient_accumulation_steps"),
            lr_scheduler_type=get("train.lr_scheduler_type"),
            max_grad_norm=float(get("train.max_grad_norm")),
            logging_steps=get("train.logging_steps"),
            save_steps=get("train.save_steps"),
            warmup_steps=get("train.warmup_steps"),
            neftune_noise_alpha=get("train.neftune_alpha") or None,
            packing=get("train.packing") or get("train.neat_packing"),
            neat_packing=get("train.neat_packing"),
            train_on_prompt=get("train.train_on_prompt"),
            mask_history=get("train.mask_history"),
            resize_vocab=get("train.resize_vocab"),
            use_llama_pro=get("train.use_llama_pro"),
            enable_thinking=get("train.enable_thinking"),
            report_to=get("train.report_to"),
            use_galore=get("train.use_galore"),
            use_apollo=get("train.use_apollo"),
            use_badam=get("train.use_badam"),
            use_swanlab=get("train.use_swanlab"),
            output_dir=get_save_dir(model_name, finetuning_type, get("train.output_dir")),
            fp16=(get("train.compute_type") == "fp16"),
            bf16=(get("train.compute_type") == "bf16"),
            pure_bf16=(get("train.compute_type") == "pure_bf16"),
            plot_loss=True,
            trust_remote_code=True,
            ddp_timeout=180000000,
            include_num_input_tokens_seen=True,
        )
        args.update(json.loads(get("train.extra_args")))

        # checkpoints
        if get("top.checkpoint_path"):
            if finetuning_type in PEFT_METHODS:  # list
                args["adapter_name_or_path"] = ",".join(
                    [get_save_dir(model_name, finetuning_type, adapter) for adapter in get("top.checkpoint_path")]
                )
            else:  # str
                args["model_name_or_path"] = get_save_dir(model_name, finetuning_type, get("top.checkpoint_path"))

        # quantization
        if get("top.quantization_bit") != "none":
            args["quantization_bit"] = int(get("top.quantization_bit"))
            args["quantization_method"] = get("top.quantization_method")
            args["double_quantization"] = not is_torch_npu_available()

        # freeze config
        if args["finetuning_type"] == "freeze":
            args["freeze_trainable_layers"] = get("train.freeze_trainable_layers")
            args["freeze_trainable_modules"] = get("train.freeze_trainable_modules")
            args["freeze_extra_modules"] = get("train.freeze_extra_modules") or None

        # lora config
        if args["finetuning_type"] == "lora":
            args["lora_rank"] = get("train.lora_rank")
            args["lora_alpha"] = get("train.lora_alpha")
            args["lora_dropout"] = get("train.lora_dropout")
            args["loraplus_lr_ratio"] = get("train.loraplus_lr_ratio") or None
            args["create_new_adapter"] = get("train.create_new_adapter")
            args["use_rslora"] = get("train.use_rslora")
            args["use_dora"] = get("train.use_dora")
            args["pissa_init"] = get("train.use_pissa")
            args["pissa_convert"] = get("train.use_pissa")
            args["lora_target"] = get("train.lora_target") or "all"
            args["additional_target"] = get("train.additional_target") or None

            if args["use_llama_pro"]:
                args["freeze_trainable_layers"] = get("train.freeze_trainable_layers")

        # rlhf config
        if args["stage"] == "ppo":
            if finetuning_type in PEFT_METHODS:
                args["reward_model"] = ",".join(
                    [get_save_dir(model_name, finetuning_type, adapter) for adapter in get("train.reward_model")]
                )
            else:
                args["reward_model"] = get_save_dir(model_name, finetuning_type, get("train.reward_model"))

            args["reward_model_type"] = "lora" if finetuning_type == "lora" else "full"
            args["ppo_score_norm"] = get("train.ppo_score_norm")
            args["ppo_whiten_rewards"] = get("train.ppo_whiten_rewards")
            args["top_k"] = 0
            args["top_p"] = 0.9
        elif args["stage"] in ["dpo", "kto"]:
            args["pref_beta"] = get("train.pref_beta")
            args["pref_ftx"] = get("train.pref_ftx")
            args["pref_loss"] = get("train.pref_loss")

        # multimodal config
        if model_name in MULTIMODAL_SUPPORTED_MODELS:
            args["freeze_vision_tower"] = get("train.freeze_vision_tower")
            args["freeze_multi_modal_projector"] = get("train.freeze_multi_modal_projector")
            args["freeze_language_model"] = get("train.freeze_language_model")
            args["image_max_pixels"] = calculate_pixels(get("train.image_max_pixels"))
            args["image_min_pixels"] = calculate_pixels(get("train.image_min_pixels"))
            args["video_max_pixels"] = calculate_pixels(get("train.video_max_pixels"))
            args["video_min_pixels"] = calculate_pixels(get("train.video_min_pixels"))

        # galore config
        if args["use_galore"]:
            args["galore_rank"] = get("train.galore_rank")
            args["galore_update_interval"] = get("train.galore_update_interval")
            args["galore_scale"] = get("train.galore_scale")
            args["galore_target"] = get("train.galore_target")

        # apollo config
        if args["use_apollo"]:
            args["apollo_rank"] = get("train.apollo_rank")
            args["apollo_update_interval"] = get("train.apollo_update_interval")
            args["apollo_scale"] = get("train.apollo_scale")
            args["apollo_target"] = get("train.apollo_target")

        # badam config
        if args["use_badam"]:
            args["badam_mode"] = get("train.badam_mode")
            args["badam_switch_mode"] = get("train.badam_switch_mode")
            args["badam_switch_interval"] = get("train.badam_switch_interval")
            args["badam_update_ratio"] = get("train.badam_update_ratio")

        # swanlab config
        if get("train.use_swanlab"):
            args["swanlab_project"] = get("train.swanlab_project")
            args["swanlab_run_name"] = get("train.swanlab_run_name")
            args["swanlab_workspace"] = get("train.swanlab_workspace")
            args["swanlab_api_key"] = get("train.swanlab_api_key")
            args["swanlab_mode"] = get("train.swanlab_mode")

        # eval config
        if get("train.val_size") > 1e-6 and args["stage"] != "ppo":
            args["val_size"] = get("train.val_size")
            args["eval_strategy"] = "steps"
            args["eval_steps"] = args["save_steps"]
            args["per_device_eval_batch_size"] = args["per_device_train_batch_size"]

        # ds config
        if get("train.ds_stage") != "none":
            ds_stage = get("train.ds_stage")
            ds_offload = "offload_" if get("train.ds_offload") else ""
            args["deepspeed"] = os.path.join(DEFAULT_CACHE_DIR, f"ds_z{ds_stage}_{ds_offload}config.json")

        return args

    def _parse_eval_args(self, data: dict["Component", Any]) -> dict[str, Any]:
        r"""Build and validate the evaluation arguments."""
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
        model_name, finetuning_type = get("top.model_name"), get("top.finetuning_type")
        user_config = load_config()

        args = dict(
            stage="sft",
            model_name_or_path=get("top.model_path"),
            cache_dir=user_config.get("cache_dir", None),
            preprocessing_num_workers=16,
            finetuning_type=finetuning_type,
            quantization_method=get("top.quantization_method"),
            template=get("top.template"),
            rope_scaling=get("top.rope_scaling") if get("top.rope_scaling") != "none" else None,
            flash_attn="fa2" if get("top.booster") == "flashattn2" else "auto",
            use_unsloth=(get("top.booster") == "unsloth"),
            dataset_dir=get("eval.dataset_dir"),
            eval_dataset=",".join(get("eval.dataset")),
            cutoff_len=get("eval.cutoff_len"),
            max_samples=int(get("eval.max_samples")),
            per_device_eval_batch_size=get("eval.batch_size"),
            predict_with_generate=True,
            report_to="none",
            max_new_tokens=get("eval.max_new_tokens"),
            top_p=get("eval.top_p"),
            temperature=get("eval.temperature"),
            seed=_parse_seed(get("eval.eval_seed")),
            output_dir=get_save_dir(model_name, finetuning_type, get("eval.output_dir")),
            trust_remote_code=True,
            ddp_timeout=180000000,
        )

        if get("eval.predict"):
            args["do_predict"] = True
        else:
            args["do_eval"] = True

        # checkpoints
        if get("top.checkpoint_path"):
            if finetuning_type in PEFT_METHODS:  # list
                args["adapter_name_or_path"] = ",".join(
                    [get_save_dir(model_name, finetuning_type, adapter) for adapter in get("top.checkpoint_path")]
                )
            else:  # str
                args["model_name_or_path"] = get_save_dir(model_name, finetuning_type, get("top.checkpoint_path"))

        # quantization
        if get("top.quantization_bit") != "none":
            args["quantization_bit"] = int(get("top.quantization_bit"))
            args["quantization_method"] = get("top.quantization_method")
            args["double_quantization"] = not is_torch_npu_available()

        return args

    def _preview(self, data: dict["Component", Any], do_train: bool) -> Generator[dict["Component", str], None, None]:
        r"""Preview the training commands."""
        output_box = self.manager.get_elem_by_id("{}.output_box".format("train" if do_train else "eval"))
        error = self._initialize(data, do_train, from_preview=True)
        if error:
            gr.Warning(error)
            yield {output_box: error}
        else:
            args = self._parse_train_args(data) if do_train else self._parse_eval_args(data)
            yield {output_box: gen_cmd(args)}

    def _launch(self, data: dict["Component", Any], do_train: bool) -> Generator[dict["Component", Any], None, None]:
        r"""Start the training process."""
        output_box = self.manager.get_elem_by_id("{}.output_box".format("train" if do_train else "eval"))
        error = self._initialize(data, do_train, from_preview=False)
        if error:
            gr.Warning(error)
            yield {output_box: error}
        else:
            self.do_train, self.running_data = do_train, data
            args = self._parse_train_args(data) if do_train else self._parse_eval_args(data)

            os.makedirs(args["output_dir"], exist_ok=True)
            save_args(os.path.join(args["output_dir"], LLAMABOARD_CONFIG), self._build_config_dict(data))

            env = deepcopy(os.environ)
            env["LLAMABOARD_ENABLED"] = "1"
            env["LLAMABOARD_WORKDIR"] = args["output_dir"]
            if args.get("deepspeed", None) is not None:
                env["FORCE_TORCHRUN"] = "1"

            # NOTE: DO NOT USE shell=True to avoid security risk
            webui_log_path = os.path.join(args["output_dir"], "webui_subprocess.log")
            webui_log = open(webui_log_path, "a", encoding="utf-8")
            self.trainer = Popen(
                ["llamafactory-cli", "train", save_cmd(args)],
                env=env,
                stdout=webui_log,
                stderr=webui_log,
                text=True,
            )
            webui_log.close()
            yield from self.monitor()

    def _build_config_dict(self, data: dict["Component", Any]) -> dict[str, Any]:
        r"""Build a dictionary containing the current training configuration."""
        config_dict = {}
        skip_ids = ["top.lang", "top.model_path", "train.output_dir", "train.config_path"]
        for elem, value in data.items():
            elem_id = self.manager.get_id_by_elem(elem)
            if elem_id not in skip_ids:
                config_dict[elem_id] = value

        return config_dict

    def preview_train(self, data):
        yield from self._preview(data, do_train=True)

    def preview_eval(self, data):
        yield from self._preview(data, do_train=False)

    def run_train(self, data):
        yield from self._launch(data, do_train=True)

    def run_eval(self, data):
        yield from self._launch(data, do_train=False)

    def monitor(self):
        r"""Monitorgit the training progress and logs."""
        self.aborted = False
        self.running = True

        get = lambda elem_id: self.running_data[self.manager.get_elem_by_id(elem_id)]
        lang, model_name, finetuning_type = get("top.lang"), get("top.model_name"), get("top.finetuning_type")
        output_dir = get("{}.output_dir".format("train" if self.do_train else "eval"))
        output_path = get_save_dir(model_name, finetuning_type, output_dir)

        output_box = self.manager.get_elem_by_id("{}.output_box".format("train" if self.do_train else "eval"))
        progress_bar = self.manager.get_elem_by_id("{}.progress_bar".format("train" if self.do_train else "eval"))
        loss_viewer = self.manager.get_elem_by_id("train.loss_viewer") if self.do_train else None
        swanlab_link = self.manager.get_elem_by_id("train.swanlab_link") if self.do_train else None

        running_log = ""
        return_code = -1
        while return_code == -1:
            if self.aborted:
                yield {
                    output_box: ALERTS["info_aborting"][lang],
                    progress_bar: gr.Slider(visible=False),
                }
            else:
                running_log, running_progress, running_info = get_trainer_info(lang, output_path, self.do_train)
                return_dict = {
                    output_box: running_log,
                    progress_bar: running_progress,
                }
                if "loss_viewer" in running_info:
                    return_dict[loss_viewer] = running_info["loss_viewer"]

                if "swanlab_link" in running_info:
                    return_dict[swanlab_link] = running_info["swanlab_link"]

                yield return_dict

            try:
                stderr = self.trainer.communicate(timeout=2)[1]
                return_code = self.trainer.returncode
            except TimeoutExpired:
                continue

        if return_code == 0 or self.aborted:
            finish_info = ALERTS["info_finished"][lang]
            if self.do_train:
                finish_log = ALERTS["info_finished"][lang] + "\n\n" + running_log
            else:
                finish_log = load_eval_results(os.path.join(output_path, "all_results.json")) + "\n\n" + running_log
        else:
            if stderr is None:
                webui_log_path = os.path.join(output_path, "webui_subprocess.log")
                if os.path.exists(webui_log_path):
                    with open(webui_log_path, "rb") as f:
                        f.seek(0, os.SEEK_END)
                        f.seek(max(f.tell() - 20000, 0))
                        stderr = f.read().decode("utf-8", errors="replace")
                else:
                    stderr = "No subprocess log file found."

            print(stderr)
            finish_info = ALERTS["err_failed"][lang]
            finish_log = ALERTS["err_failed"][lang] + f" Exit code: {return_code}\n\n```\n{stderr}\n```\n"

        self._finalize(lang, finish_info)
        return_dict = {output_box: finish_log, progress_bar: gr.Slider(visible=False)}
        yield return_dict

    def save_args(self, data):
        r"""Save the training configuration to config path."""
        output_box = self.manager.get_elem_by_id("train.output_box")
        error = self._initialize(data, do_train=True, from_preview=True)
        if error:
            gr.Warning(error)
            return {output_box: error}

        lang = data[self.manager.get_elem_by_id("top.lang")]
        config_path = data[self.manager.get_elem_by_id("train.config_path")]
        os.makedirs(DEFAULT_CONFIG_DIR, exist_ok=True)
        save_path = os.path.join(DEFAULT_CONFIG_DIR, config_path)

        save_args(save_path, self._build_config_dict(data))
        return {output_box: ALERTS["info_config_saved"][lang] + save_path}

    def load_args(self, lang: str, config_path: str):
        r"""Load the training configuration from config path."""
        output_box = self.manager.get_elem_by_id("train.output_box")
        config_dict = load_args(os.path.join(DEFAULT_CONFIG_DIR, config_path))
        if config_dict is None:
            gr.Warning(ALERTS["err_config_not_found"][lang])
            return {output_box: ALERTS["err_config_not_found"][lang]}

        output_dict: dict[Component, Any] = {output_box: ALERTS["info_config_loaded"][lang]}
        for elem_id, value in config_dict.items():
            output_dict[self.manager.get_elem_by_id(elem_id)] = value

        return output_dict

    def check_output_dir(self, lang: str, model_name: str, finetuning_type: str, output_dir: str):
        r"""Restore the training status if output_dir exists."""
        output_box = self.manager.get_elem_by_id("train.output_box")
        output_dict: dict[Component, Any] = {output_box: LOCALES["output_box"][lang]["value"]}
        if model_name and output_dir and os.path.isdir(get_save_dir(model_name, finetuning_type, output_dir)):
            gr.Warning(ALERTS["warn_output_dir_exists"][lang])
            output_dict[output_box] = ALERTS["warn_output_dir_exists"][lang]

            output_dir = get_save_dir(model_name, finetuning_type, output_dir)
            config_dict = load_args(os.path.join(output_dir, LLAMABOARD_CONFIG))  # load llamaboard config
            for elem_id, value in config_dict.items():
                output_dict[self.manager.get_elem_by_id(elem_id)] = value

        return output_dict
