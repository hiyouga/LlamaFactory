# Copyright 2025 the KVCache.AI team, Approaching AI, and the LlamaFactory team.
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

import os
import shutil
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed as dist
from transformers import EarlyStoppingCallback, PreTrainedModel

from ..data import get_template_and_fix_tokenizer
from ..extras import logging
from ..extras.constants import V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME
from ..extras.misc import find_available_port, get_device_name, get_torch_device, infer_optim_dtype
from ..extras.packages import (
    is_hyper_parallel_available,
    is_mcore_adapter_available,
    is_megatron_bridge_available,
    is_ray_available,
    is_transformers_version_greater_than,
)
from ..hparams import RayArguments, get_infer_args, get_ray_args, get_train_args, read_args
from ..model import load_model, load_tokenizer
from .callbacks import (
    LogCallback,
    ModuleProfilerCallback,
    PissaConvertCallback,
    ReporterCallback,
    TorchProfilerCallback,
)
from .dpo import run_dpo
from .kto import run_kto
from .ppo import run_ppo
from .pt import run_pt
from .rm import run_rm
from .sft import run_sft
from .trainer_utils import (
    get_placement_group,
    get_ray_head_node_ip,
    get_ray_remote_config_for_worker,
    get_swanlab_callback,
    sort_placement_group_by_node_ip,
)


if is_ray_available():
    import ray


if TYPE_CHECKING:
    from transformers import TrainerCallback


logger = logging.get_logger(__name__)


def _training_function(config: dict[str, Any]) -> None:
    args = config.get("args")
    callbacks: list[Any] = config.get("callbacks")
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    callbacks.append(LogCallback())
    if finetuning_args.pissa_convert:
        callbacks.append(PissaConvertCallback())

    if finetuning_args.use_swanlab:
        callbacks.append(get_swanlab_callback(finetuning_args))

    if finetuning_args.early_stopping_steps is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=finetuning_args.early_stopping_steps))

    if getattr(training_args, "enable_torch_profiler", False):
        callbacks.append(TorchProfilerCallback(training_args))

    if getattr(training_args, "profile_modules", None):
        callbacks.append(ModuleProfilerCallback(training_args.profile_modules))

    callbacks.append(ReporterCallback(model_args, data_args, finetuning_args, generating_args))  # add to last

    if finetuning_args.stage in ["pt", "sft"] and finetuning_args.use_hyper_parallel:
        if not is_hyper_parallel_available():
            raise ImportError("hyper_parallel is not installed. Please install it with `pip install hyper_parallel`.")
        if finetuning_args.stage == "pt":
            from .hyper_parallel import run_pt as run_pt_hp

            run_pt_hp(model_args, data_args, training_args, finetuning_args, callbacks)
        else:
            from .hyper_parallel import run_sft as run_sft_hp

            run_sft_hp(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)

    elif finetuning_args.stage in ["pt", "sft"] and finetuning_args.use_megatron_bridge:
        if not is_megatron_bridge_available():
            raise ImportError(
                "megatron-bridge is not installed. "
                "Please install it with `pip install --no-build-isolation megatron-bridge`."
            )
        mb_args = finetuning_args.megatron_bridge_args
        if mb_args is None:
            raise ValueError("Megatron Bridge arguments are missing. Please set USE_MEGATRON_BRIDGE=1.")
        if finetuning_args.stage == "pt":
            from .megatron_bridge import run_pt as run_pt_mb

            run_pt_mb(model_args, data_args, training_args, finetuning_args, mb_args, callbacks)
        else:
            from .megatron_bridge import run_sft as run_sft_mb

            run_sft_mb(model_args, data_args, training_args, finetuning_args, mb_args, callbacks)

    elif finetuning_args.stage in ["pt", "sft", "dpo"] and finetuning_args.use_mca:
        if not is_mcore_adapter_available():
            raise ImportError("mcore_adapter is not installed. Please install it with `pip install mcore-adapter`.")
        if finetuning_args.stage == "pt":
            from .mca import run_pt as run_pt_mca

            run_pt_mca(model_args, data_args, training_args, finetuning_args, callbacks)
        elif finetuning_args.stage == "sft":
            from .mca import run_sft as run_sft_mca

            run_sft_mca(model_args, data_args, training_args, finetuning_args, callbacks)
        elif finetuning_args.stage == "dpo":
            from .mca import run_dpo as run_dpo_mca

            run_dpo_mca(model_args, data_args, training_args, finetuning_args, callbacks)

    elif finetuning_args.stage == "pt":
        run_pt(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "sft":
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "rm":
        run_rm(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "ppo":
        run_ppo(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "dpo":
        run_dpo(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "kto":
        run_kto(model_args, data_args, training_args, finetuning_args, callbacks)
    else:
        raise ValueError(f"Unknown task: {finetuning_args.stage}.")

    if is_ray_available() and ray.is_initialized():
        return  # if ray is initialized it will destroy the process group on return

    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        logger.warning(f"Failed to destroy process group: {e}.")


def run_exp(args: Optional[dict[str, Any]] = None, callbacks: Optional[list["TrainerCallback"]] = None) -> None:
    args = read_args(args)
    if "-h" in args or "--help" in args:
        get_train_args(args)

    ray_args = get_ray_args(args)
    callbacks = callbacks or []
    if ray_args.use_ray:
        _ray_training_function(ray_args, config={"args": args, "callbacks": callbacks})
    else:
        _training_function(config={"args": args, "callbacks": callbacks})


def export_model(args: Optional[dict[str, Any]] = None) -> None:
    model_args, data_args, finetuning_args, _ = get_infer_args(args)

    if model_args.export_dir is None:
        raise ValueError("Please specify `export_dir` to save model.")

    if model_args.adapter_name_or_path is not None and model_args.export_quantization_bit is not None:
        raise ValueError("Please merge adapters before quantizing the model.")

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args)  # must after fixing tokenizer to resize vocab

    if getattr(model, "quantization_method", None) is not None and model_args.adapter_name_or_path is not None:
        raise ValueError("Cannot merge adapters to a quantized model.")

    if not isinstance(model, PreTrainedModel):
        raise ValueError("The model is not a `PreTrainedModel`, export aborted.")

    if getattr(model, "quantization_method", None) is not None:  # quantized model adopts float16 type
        setattr(model.config, "torch_dtype", torch.float16)
    else:
        if model_args.infer_dtype == "auto":
            output_dtype = getattr(model.config, "torch_dtype", torch.float32)
            if output_dtype == torch.float32:  # if infer_dtype is auto, try using half precision first
                output_dtype = infer_optim_dtype(torch.bfloat16)
        else:
            output_dtype = getattr(torch, model_args.infer_dtype)

        setattr(model.config, "torch_dtype", output_dtype)
        model = model.to(output_dtype)
        logger.info_rank0(f"Convert model dtype to: {output_dtype}.")

    # Prepare save arguments (safe_serialization removed in transformers v5.0.0)
    save_kwargs = {
        "save_directory": model_args.export_dir,
        "max_shard_size": f"{model_args.export_size}GB",
    }
    if not is_transformers_version_greater_than("5.0.0"):
        save_kwargs["safe_serialization"] = not model_args.export_legacy_format

    try:
        model.save_pretrained(**save_kwargs)
    except NotImplementedError as err:
        raise RuntimeError(
            "Failed to export model: weight conversion reversal is not supported for this model architecture "
            "(NotImplementedError in transformers.core_model_loading.reverse_op). "
            "This is a known issue with transformers>=5.0 for certain model types (e.g. Mistral/Ministral). "
            "Workarounds: (1) use transformers<5.0, or (2) report the issue to the transformers repository."
        ) from err

    if model_args.export_hub_model_id is not None:
        # Prepare push arguments (safe_serialization removed in transformers v5.0.0)
        push_kwargs = {
            "max_shard_size": f"{model_args.export_size}GB",
        }
        if not is_transformers_version_greater_than("5.0.0"):
            push_kwargs["safe_serialization"] = not model_args.export_legacy_format

        model.push_to_hub(
            model_args.export_hub_model_id,
            token=model_args.hf_hub_token,
            **push_kwargs,
        )

    if finetuning_args.stage == "rm":
        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        if os.path.exists(os.path.join(vhead_path, V_HEAD_SAFE_WEIGHTS_NAME)):
            shutil.copy(
                os.path.join(vhead_path, V_HEAD_SAFE_WEIGHTS_NAME),
                os.path.join(model_args.export_dir, V_HEAD_SAFE_WEIGHTS_NAME),
            )
            logger.info_rank0(f"Copied valuehead to {model_args.export_dir}.")
        elif os.path.exists(os.path.join(vhead_path, V_HEAD_WEIGHTS_NAME)):
            shutil.copy(
                os.path.join(vhead_path, V_HEAD_WEIGHTS_NAME),
                os.path.join(model_args.export_dir, V_HEAD_WEIGHTS_NAME),
            )
            logger.info_rank0(f"Copied valuehead to {model_args.export_dir}.")

    try:
        tokenizer.padding_side = "left"  # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"
        tokenizer.save_pretrained(model_args.export_dir)
        if model_args.export_hub_model_id is not None:
            tokenizer.push_to_hub(model_args.export_hub_model_id, token=model_args.hf_hub_token)

        if processor is not None:
            processor.save_pretrained(model_args.export_dir)
            if model_args.export_hub_model_id is not None:
                processor.push_to_hub(model_args.export_hub_model_id, token=model_args.hf_hub_token)

    except Exception as e:
        logger.warning_rank0(f"Cannot save tokenizer, please copy the files manually: {e}.")

    ollama_modelfile = os.path.join(model_args.export_dir, "Modelfile")
    with open(ollama_modelfile, "w", encoding="utf-8") as f:
        f.write(template.get_ollama_modelfile(tokenizer))
        logger.info_rank0(f"Ollama modelfile saved in {ollama_modelfile}")


class Worker:
    def __init__(self):
        self._setup_env_visible_devices()

        local_rank = os.environ.get("LOCAL_RANK", "0")
        get_torch_device().set_device(int(local_rank))

    def _setup_env_visible_devices(self) -> None:
        RAY_NOSET_VISIBLE_DEVICES_LIST = [
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
            "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
        ]
        is_ray_noset_visible_devices = any(os.environ.get(env_var, None) for env_var in RAY_NOSET_VISIBLE_DEVICES_LIST)
        if is_ray_noset_visible_devices:
            device_name = get_device_name().upper()
            local_rank = ray.get_runtime_context().get_accelerator_ids()[device_name][0]
            os.environ["LOCAL_RANK"] = local_rank
        else:
            os.environ["LOCAL_RANK"] = "0"

    def _training_function(self, config: dict[str, Any]) -> None:
        _training_function(config)


def _ray_training_function(ray_args: "RayArguments", config: dict[str, Any]) -> None:
    num_workers = ray_args.ray_num_workers
    master_addr = ray_args.master_addr
    master_port = ray_args.master_port
    logger.info(f"Using ray.remote mode with {num_workers} workers for distributed training.")

    # initialize ray
    if not ray.is_initialized():
        if ray_args.ray_init_kwargs is not None:
            ray.init(**ray_args.ray_init_kwargs)
        else:
            ray.init()

    # verify resources
    device_name = get_device_name().upper()
    total_devices = int(ray.cluster_resources().get(device_name, 0))
    if num_workers > total_devices:
        raise ValueError(
            f"The number of devices in the Ray cluster ({total_devices}) should be greater than num_workers ({num_workers})."
        )

    # verify master_addr
    if master_addr is None:
        master_addr = get_ray_head_node_ip()
        logger.info(f"`master_addr` is not specified, using head node ip: {master_addr}.")
    else:
        nodes = [node["NodeManagerAddress"] for node in ray.nodes() if node["Alive"]]
        if master_addr not in nodes:
            raise ValueError(f"The `master_addr` ({master_addr}) is not in Ray cluster or not alive ")

    # create placementgroup for resource management
    pg, bundle = get_placement_group(total_devices)
    ray.get(pg.ready())
    logger.info(f"Create placement group with {num_workers} bundles: {bundle}")

    # get sorted_bundle_indices
    sorted_bundle_indices = sort_placement_group_by_node_ip(pg, master_addr)

    # get master port
    if master_port is None:
        master_port = find_available_port()
        logger.info(f"`master_port` is not specified, using available port: {master_port}.")
    master_port = str(master_port)

    # backing up environment variables
    current_env = dict(os.environ.items())

    # launch workers
    RayWorker = ray.remote(Worker)
    workers = []
    for rank in range(num_workers):
        remote_config = get_ray_remote_config_for_worker(
            placement_group=pg,
            bundle_idx=sorted_bundle_indices[rank],
            rank=rank,
            world_size=num_workers,
            master_addr=master_addr,
            master_port=master_port,
            env=current_env,
        )
        worker = RayWorker.options(**remote_config).remote()
        workers.append(worker)

    ray.get([worker._training_function.remote(config=config) for worker in workers])
    ray.shutdown()
