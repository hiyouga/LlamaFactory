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

import copy
import gc
import os

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from peft.tuners.lora import LoraLayer
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)

from ....accelerator.helper import get_current_accelerator
from ....accelerator.interface import DistributedInterface
from ....utils.logging import get_logger
from ....utils.types import HFModel, Processor


logger = get_logger(__name__)


def _fallback_dot_natural_key(name: str):
    parts = []
    for part in name.split("."):
        if part.isdigit():
            parts.append((0, int(part)))
        else:
            parts.append((1, part))
    return parts


def _get_checkpoint_sort_key():
    try:
        from transformers.core_model_loading import dot_natural_key

        return dot_natural_key
    except ImportError:
        return _fallback_dot_natural_key


def _make_safetensor_loader(checkpoint_file: str, tensor_key: str):
    # Delay tensor materialization until converter.convert() to reduce peak CPU memory.
    # This works because HF WeightConverter accepts callables and materializes them later.
    def _load_tensor():
        from safetensors import safe_open

        with safe_open(checkpoint_file, framework="pt", device="cpu") as f:
            return f.get_tensor(tensor_key)

    return _load_tensor


def _cast_norm_input_to_weight_dtype(module: nn.Module, args: tuple):
    """forward-pre-hook: cast a norm layer's input to its weight dtype."""
    if not args:
        return None
    x = args[0]
    weight = getattr(module, "weight", None)
    if isinstance(x, torch.Tensor) and weight is not None and x.dtype != weight.dtype:
        return (x.to(weight.dtype), *args[1:])
    return None


def _make_norms_dtype_safe(model: HFModel) -> int:
    """Register the dtype-safe hook on every dtype-strict ``nn.LayerNorm`` in the model."""
    n = 0
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            module.register_forward_pre_hook(_cast_norm_input_to_weight_dtype)
            n += 1
    return n


def is_lora_model(model: HFModel) -> bool:
    """Return whether PEFT LoRA layers have already been injected into the model."""
    return any(isinstance(module, LoraLayer) for module in model.modules())


def get_transformer_layer_cls(model: HFModel) -> set[type[nn.Module]]:
    classes: set[type[nn.Module]] = set()
    for module in model.modules():
        for attr in ("layers", "blocks"):
            seq = getattr(module, attr, None)
            if isinstance(seq, nn.ModuleList) and len(seq) > 0:
                classes.add(type(seq[0]))
    if classes:
        return classes

    no_split_modules = getattr(model, "_no_split_modules", None)
    if no_split_modules:
        found: dict[str, type[nn.Module]] = {}
        for _, module in model.named_modules():
            cls_name = module.__class__.__name__
            if cls_name in no_split_modules and cls_name not in found:
                found[cls_name] = module.__class__
            if len(found) == len(no_split_modules):
                break
        if found:
            return set(found.values())

    return set()


def save_model(model: HFModel, output_dir: str, processor: Processor) -> None:
    if DistributedInterface().get_rank() == 0:
        logger.info("Gathering state dict for saving...")

    lora_model = is_lora_model(model)
    options = StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=lora_model)
    state_dict = get_model_state_dict(model, options=options)

    if DistributedInterface().get_rank() == 0:
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir, state_dict=state_dict, max_shard_size="4GB")
        processor.save_pretrained(output_dir, max_shard_size="4GB")
        logger.info(f"Model saved to {output_dir}")


def save_checkpoint(model: HFModel, optimizer: torch.optim.Optimizer, ckpt_dir: str, **kwargs) -> None:
    save_ckpt_as_hf = kwargs.get("save_ckpt_as_hf", False)
    processor = kwargs.get("processor", None)

    # Always save DCP format for resume capability
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)

    model_state = get_model_state_dict(model, options=options)
    dcp.save(state_dict=model_state, checkpoint_id=os.path.join(ckpt_dir, "model"))

    optim_state = get_optimizer_state_dict(model, optimizer, options=options)
    dcp.save(state_dict=optim_state, checkpoint_id=os.path.join(ckpt_dir, "optimizer"))

    # Additionally save HF format if requested
    if save_ckpt_as_hf:
        if DistributedInterface().get_rank() == 0:
            logger.info("Gathering state dict for saving additional HF format checkpoint...")

        lora_model = is_lora_model(model)
        hf_options = StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=lora_model)
        hf_state_dict = get_model_state_dict(model, options=hf_options)

        if DistributedInterface().get_rank() == 0:
            model_to_save = model.module if hasattr(model, "module") else model
            hf_dir = os.path.join(ckpt_dir, "hf_model")
            model_to_save.save_pretrained(hf_dir, state_dict=hf_state_dict, max_shard_size="4GB")
            if processor is not None:
                processor.save_pretrained(hf_dir, max_shard_size="4GB")

            logger.info(f"Additional HF format checkpoint saved to {hf_dir}")


def load_checkpoint(model: HFModel, optimizer: torch.optim.Optimizer, ckpt_dir: str, **kwargs) -> None:
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)

    ckpt_model_dir = os.path.join(ckpt_dir, "model")
    model_state = get_model_state_dict(model, options=options)
    dcp.load(state_dict=model_state, checkpoint_id=ckpt_model_dir)
    set_model_state_dict(model, model_state, options=options)

    ckpt_optim_dir = os.path.join(ckpt_dir, "optimizer")
    optim_state = get_optimizer_state_dict(model, optimizer, options=options)
    dcp.load(state_dict=optim_state, checkpoint_id=ckpt_optim_dir)
    set_optimizer_state_dict(model, optimizer, optim_state, options=options)


class FSDP2Engine:
    def __init__(self, dist_config: dict, bf16: bool = False):
        self.dist_interface = DistributedInterface()
        self.rank = self.dist_interface.get_rank()
        self.local_rank = self.dist_interface.get_local_rank()
        self.world_size = self.dist_interface.get_world_size()
        self.mixed_precision = "bf16" if bf16 else "fp32"
        self.reshard_after_forward = dist_config.get("reshard_after_forward", True)
        self.offload_params = dist_config.get("offload_params", False)
        self.pin_memory = dist_config.get("pin_memory", True)
        self.dcp_path = dist_config.get("dcp_path", None)
        self.device_mesh = self.dist_interface.model_device_mesh

        if self.device_mesh is None:
            logger.warning(
                "Device Mesh not found in DistributedInterface. FSDP2 might fail if not running in distributed mode."
            )

        if self.device_mesh is not None:
            self.fsdp_mesh = self.device_mesh

            logger.info(f"Using Device Mesh: {self.fsdp_mesh}")
        else:
            self.fsdp_mesh = None

    def get_mp_policy(self) -> MixedPrecisionPolicy:
        if self.mixed_precision == "bf16":
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
        elif self.mixed_precision == "fp32":
            param_dtype = torch.float32
            reduce_dtype = torch.float32

        return MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            cast_forward_inputs=True,
        )

    def is_lora_module_wrap(self, model) -> bool:
        return is_lora_model(model)

    def prepare_model(self, model: HFModel, ignored_params: set[nn.Parameter] | None = None) -> HFModel:
        if self.fsdp_mesh is None:
            logger.warning("No FSDP Mesh available, skipping FSDP wrapping.")
            return model

        mp_policy = self.get_mp_policy()
        transformer_layer_cls_to_wrap = get_transformer_layer_cls(model)

        if not transformer_layer_cls_to_wrap:
            logger.warning(
                "Could not identify Transformer Layer class, applying FSDP to the whole model structure only."
            )
        else:
            names = ", ".join(cls.__name__ for cls in transformer_layer_cls_to_wrap)
            logger.info(f"Applying per-layer FSDP to: {names}")

        def _ignored_params_for(module: nn.Module) -> set[nn.Parameter] | None:
            if not ignored_params:
                return None
            return ignored_params.intersection(module.parameters()) or None

        if self.is_lora_module_wrap(model):
            lora_modules = []
            for module in model.modules():
                if len(list(module.children())) != 0:
                    continue
                if any(param.requires_grad for param in module.parameters(recurse=False)):
                    lora_modules.append(module)

            for module in lora_modules:
                fully_shard(
                    module,
                    mesh=self.fsdp_mesh,
                    reshard_after_forward=self.reshard_after_forward,
                    mp_policy=mp_policy,
                    offload_policy=CPUOffloadPolicy(pin_memory=self.pin_memory) if self.offload_params else None,
                    ignored_params=_ignored_params_for(module),
                )

            logger.info("Applying FSDP wrap for LoRA layer separately.")

        for name, module in model.named_modules():
            should_wrap = False

            if type(module) in transformer_layer_cls_to_wrap:
                should_wrap = True
            elif isinstance(module, nn.Embedding):
                if not getattr(model.config, "tie_word_embeddings", True):
                    should_wrap = True

            if should_wrap:
                fully_shard(
                    module,
                    mesh=self.fsdp_mesh,
                    reshard_after_forward=self.reshard_after_forward,
                    mp_policy=mp_policy,
                    offload_policy=CPUOffloadPolicy(pin_memory=self.pin_memory) if self.offload_params else None,
                    ignored_params=_ignored_params_for(module),
                )

        # BaseTrainer is the single source of truth for gradient checkpointing.
        # FSDP2 only applies the input-grad compatibility hook when checkpointing is already enabled.
        if getattr(model, "is_gradient_checkpointing", False):
            if self.rank == 0:
                logger.info("Gradient checkpointing is enabled. Applying FSDP2 input grad preparation.")

            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            if self.mixed_precision == "bf16":
                n_patched = _make_norms_dtype_safe(model)
                if self.rank == 0 and n_patched:
                    logger.info(f"Made {n_patched} nn.LayerNorm(s) dtype-safe for bf16 checkpointing.")

        fully_shard(
            model,
            mesh=self.fsdp_mesh,
            reshard_after_forward=self.reshard_after_forward,
            mp_policy=mp_policy,
            offload_policy=CPUOffloadPolicy(pin_memory=self.pin_memory) if self.offload_params else None,
            ignored_params=_ignored_params_for(model),
        )

        return model

    @torch.no_grad()
    def materialize_and_load(self, model: HFModel, hf_model_path: str, dcp_path: str = None):
        if self.rank == 0:
            logger.info("Materializing sharded model params...")

        device = get_current_accelerator()
        model.to_empty(device=device)

        if dcp_path and os.path.exists(dcp_path):
            if self.rank == 0:
                logger.info(f"DCP path found at {dcp_path}. Using efficient Sharded Loading (DCP Load).")
            self._load_from_dcp(model, dcp_path)
        else:
            if self.rank == 0:
                if dcp_path:
                    logger.warning(f"DCP path {dcp_path} not found.")
                logger.info("Using HF Meta Loading (Chunk Load).")
            self._load_weights_from_hf_checkpoint(model, hf_model_path)

        return model

    def _save_non_persistent_buffers(self, model: HFModel) -> dict:
        """Save non-persistent buffers, such as inv_freq."""
        saved = {}
        for mod_name, module in model.named_modules():
            for buf_name in module._non_persistent_buffers_set:
                fqn = f"{mod_name}.{buf_name}" if mod_name else buf_name
                buf = getattr(module, buf_name, None)
                if buf is not None:
                    saved[fqn] = copy.deepcopy(buf)
        if self.rank == 0 and saved:
            logger.info(f"Saved {len(saved)} non-persistent buffers")
        return saved

    def _restore_non_persistent_buffers(self, model: HFModel, saved_buffers: dict):
        """Register saved non-persistent buffers to model."""
        if not saved_buffers:
            return
        device = get_current_accelerator()
        for fqn, buf in saved_buffers.items():
            buf = buf.to(device)
            if "." in fqn:
                parent_fqn, buf_name = fqn.rsplit(".", 1)
                parent_module = model.get_submodule(parent_fqn)
            else:
                buf_name = fqn
                parent_module = model
            parent_module.register_buffer(buf_name, buf, persistent=False)
        if self.rank == 0:
            logger.info(f"Restored {len(saved_buffers)} non-persistent buffers")

    def shard_model(self, model: HFModel) -> HFModel:
        init_mode = getattr(model, "_init_mode", "init_on_default")

        if init_mode == "init_on_rank0":
            if getattr(model.config, "tie_word_embeddings", False):
                model.tie_weights()

            if self.rank == 0:
                logger.info("init_on_rank0 detected: sharding then scattering Rank 0 CPU weights.")
                full_sd = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                full_sd = {}

            model = self.prepare_model(model)

            device = get_current_accelerator()
            model.to_empty(device=device)

            # Scatter params from Rank 0 into all DTensor shards
            # Broadcast the full state dict from the global rank-0 process to all ranks in this group.
            options = StateDictOptions(full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True)
            set_model_state_dict(model, full_sd, options=options)

            if self.rank == 0:
                logger.info("init_on_rank0 sync complete.")

        elif init_mode == "init_on_meta":
            non_persistent_buffers = self._save_non_persistent_buffers(model)

            if getattr(model.config, "tie_word_embeddings", False):
                model.tie_weights()

            model = self.prepare_model(model)
            model = self.materialize_and_load(model, hf_model_path=model.config.name_or_path, dcp_path=self.dcp_path)

            # fix tied broken for no-fsdp-wrap case
            if getattr(model.config, "tie_word_embeddings", False):
                model.tie_weights()

            self._restore_non_persistent_buffers(model, non_persistent_buffers)

        else:
            model = self.prepare_model(model)

        self._warmup_grad_norm(model)

        return model

    def _warmup_grad_norm(self, model: HFModel) -> None:
        """Warmup grad norm computation to initialize NCCL communication groups."""
        if self.fsdp_mesh is None:
            return

        logger.info_rank0("Warming up grad norm computation...")

        for param in model.parameters():
            if param.requires_grad:
                param.grad = torch.zeros_like(param)

        with torch.no_grad():
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if isinstance(grad_norm, torch.distributed.tensor.DTensor):
                grad_norm = grad_norm.full_tensor()

        for param in model.parameters():
            if param.requires_grad:
                param.grad = None

        logger.info_rank0("Grad norm warmup completed.")

    def _load_from_dcp(self, model: HFModel, dcp_path: str):
        import torch.distributed.checkpoint as dcp

        try:
            if self.rank == 0:
                logger.info(f"Loading distributed checkpoint from {dcp_path} ...")

            options = StateDictOptions(full_state_dict=False, cpu_offload=True)
            local_state_dict = get_model_state_dict(model, options=options)
            dcp.load(state_dict=local_state_dict, checkpoint_id=dcp_path)
            set_model_state_dict(model, local_state_dict, options=options)

            if self.rank == 0:
                logger.info("DCP weights loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load from DCP: {e}")
            raise e

    def _try_build_hf_weight_conversion_context(self, model: HFModel) -> dict | None:
        try:
            from transformers.conversion_mapping import get_model_conversion_mapping
            from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key
        except ImportError:
            return None

        weight_mapping = get_model_conversion_mapping(model)
        if not weight_mapping:
            return None

        renamings = [entry for entry in weight_mapping if isinstance(entry, WeightRenaming)]
        converters = [entry for entry in weight_mapping if isinstance(entry, WeightConverter)]
        return {
            "prefix": getattr(model, "base_model_prefix", ""),
            "meta_state_dict": model.state_dict(),
            "rename_source_key": rename_source_key,
            "renamings": renamings,
            "converters": converters,
            "converter_templates": {
                pattern: converter for converter in converters for pattern in converter.source_patterns
            },
            "pending_converters": {},
        }

    def _load_weights_from_hf_checkpoint(self, model: HFModel, hf_model_path: str):
        import glob
        import json

        hf_model_path = self._resolve_hf_checkpoint_dir(hf_model_path)
        sort_key = _get_checkpoint_sort_key()

        if self.rank == 0:
            logger.info(f"Loading weights from {hf_model_path} ...")

        index_file = os.path.join(hf_model_path, "model.safetensors.index.json")
        is_safetensors = True
        checkpoint_files = []

        if os.path.exists(index_file):
            with open(index_file) as f:
                index = json.load(f)
            checkpoint_files = sorted(set(index["weight_map"].values()))
            checkpoint_files = [os.path.join(hf_model_path, f) for f in checkpoint_files]
        elif os.path.exists(os.path.join(hf_model_path, "model.safetensors")):
            checkpoint_files = [os.path.join(hf_model_path, "model.safetensors")]
        else:
            is_safetensors = False
            index_file = os.path.join(hf_model_path, "pytorch_model.bin.index.json")
            if os.path.exists(index_file):
                with open(index_file) as f:
                    index = json.load(f)
                checkpoint_files = sorted(set(index["weight_map"].values()))
                checkpoint_files = [os.path.join(hf_model_path, f) for f in checkpoint_files]
            elif os.path.exists(os.path.join(hf_model_path, "pytorch_model.bin")):
                checkpoint_files = [os.path.join(hf_model_path, "pytorch_model.bin")]
            else:
                checkpoint_files = sorted(glob.glob(os.path.join(hf_model_path, "*.safetensors")))
                if checkpoint_files:
                    is_safetensors = True
                else:
                    checkpoint_files = sorted(glob.glob(os.path.join(hf_model_path, "*.bin")))

        if not checkpoint_files:
            raise ValueError(f"No checkpoint files found in {hf_model_path}")

        param_map = dict(model.named_parameters())
        conversion_ctx = self._try_build_hf_weight_conversion_context(model)
        total_files = len(checkpoint_files)

        for i, ckpt_file in enumerate(checkpoint_files):
            if self.rank == 0:
                logger.info(f"[{i + 1}/{total_files}] Loading {os.path.basename(ckpt_file)} ...")

            if is_safetensors:
                from safetensors import safe_open

                with safe_open(ckpt_file, framework="pt", device="cpu") as f:
                    for key in sorted(f.keys(), key=sort_key):
                        renamed_key = key
                        source_pattern = None
                        if conversion_ctx is not None:
                            renamed_key, source_pattern = conversion_ctx["rename_source_key"](
                                key,
                                conversion_ctx["renamings"],
                                conversion_ctx["converters"],
                                prefix=conversion_ctx["prefix"],
                                meta_state_dict=conversion_ctx["meta_state_dict"],
                            )

                        if source_pattern is not None:
                            template = conversion_ctx["converter_templates"][source_pattern]
                            converter = conversion_ctx["pending_converters"].setdefault(
                                renamed_key, copy.deepcopy(template)
                            )
                            converter.add_tensor(
                                renamed_key,
                                key,
                                source_pattern,
                                _make_safetensor_loader(ckpt_file, key),
                            )
                        elif renamed_key in param_map:
                            tensor = f.get_tensor(key)
                            self._copy_weights(param_map[renamed_key], tensor)
            else:
                state_dict = torch.load(ckpt_file, map_location="cpu")
                for key, tensor in sorted(state_dict.items(), key=lambda item: sort_key(item[0])):
                    renamed_key = key
                    source_pattern = None
                    if conversion_ctx is not None:
                        renamed_key, source_pattern = conversion_ctx["rename_source_key"](
                            key,
                            conversion_ctx["renamings"],
                            conversion_ctx["converters"],
                            prefix=conversion_ctx["prefix"],
                            meta_state_dict=conversion_ctx["meta_state_dict"],
                        )

                    if source_pattern is not None:
                        template = conversion_ctx["converter_templates"][source_pattern]
                        converter = conversion_ctx["pending_converters"].setdefault(
                            renamed_key, copy.deepcopy(template)
                        )
                        converter.add_tensor(renamed_key, key, source_pattern, tensor)
                    elif renamed_key in param_map:
                        self._copy_weights(param_map[renamed_key], tensor)
                del state_dict
                gc.collect()

        if conversion_ctx is not None:
            pending_count = len(conversion_ctx["pending_converters"])
            log_fn = getattr(logger, "info_rank0", logger.info)
            log_fn(f"Applying {pending_count} deferred HF weight conversions.")
            for layer_name, converter in sorted(conversion_ctx["pending_converters"].items()):
                realized_tensors = converter.convert(layer_name, model=model, config=model.config)
                for target_name, tensor in realized_tensors.items():
                    if isinstance(tensor, list):
                        tensor = tensor[0]
                    if target_name in param_map:
                        self._copy_weights(param_map[target_name], tensor)
                del realized_tensors
                gc.collect()

    def _resolve_hf_checkpoint_dir(self, hf_model_path: str) -> str:
        """Resolve a HF model identifier or local path to a local directory containing checkpoint files.

        - If `hf_model_path` is an existing directory, return it.
        - If it's a file path, return its parent directory.
        - Otherwise treat it as a Hugging Face Hub repo id and download/resolve to the local cache dir.
        """
        if not hf_model_path:
            return hf_model_path

        # Local directory or file path.
        if os.path.isdir(hf_model_path):
            return hf_model_path
        if os.path.isfile(hf_model_path):
            return os.path.dirname(hf_model_path)

        # HuggingFace Hub repo id: snapshot to local cache so we can glob/index files.
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ValueError(
                f"hf_model_path='{hf_model_path}' does not exist locally and huggingface_hub is not available "
                f"to download it. Please provide a local model directory or install huggingface_hub. Error: {e}"
            ) from e

        revision = os.getenv("HF_REVISION")
        offline = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"

        # In distributed runs, let rank0 download first to avoid N-way concurrent downloads.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if self.rank == 0:
                local_dir = snapshot_download(
                    repo_id=hf_model_path,
                    revision=revision,
                    local_files_only=offline,
                    allow_patterns=[
                        "*.safetensors",
                        "*.bin",
                        "*.index.json",
                        "model.safetensors",
                        "model.safetensors.index.json",
                        "pytorch_model.bin",
                        "pytorch_model.bin.index.json",
                        "config.json",
                    ],
                )
                logger.info(f"Resolved HF repo id '{hf_model_path}' to local dir: {local_dir}")
            torch.distributed.barrier()
            if self.rank != 0:
                local_dir = snapshot_download(
                    repo_id=hf_model_path,
                    revision=revision,
                    local_files_only=True,
                    allow_patterns=[
                        "*.safetensors",
                        "*.bin",
                        "*.index.json",
                        "model.safetensors",
                        "model.safetensors.index.json",
                        "pytorch_model.bin",
                        "pytorch_model.bin.index.json",
                        "config.json",
                    ],
                )
            return local_dir

        local_dir = snapshot_download(
            repo_id=hf_model_path,
            revision=revision,
            local_files_only=offline,
            allow_patterns=[
                "*.safetensors",
                "*.bin",
                "*.index.json",
                "model.safetensors",
                "model.safetensors.index.json",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
                "config.json",
            ],
        )
        if self.rank == 0:
            logger.info(f"Resolved HF repo id '{hf_model_path}' to local dir: {local_dir}")
        return local_dir

    def _copy_weights(self, param, loaded_tensor):
        from torch.distributed._tensor import DTensor, Shard

        if loaded_tensor.dtype != param.dtype:
            loaded_tensor = loaded_tensor.to(param.dtype)

        if isinstance(param, DTensor):
            shard_placement = None
            mesh_dim = -1

            for i, placement in enumerate(param.placements):
                if isinstance(placement, Shard):
                    shard_placement = placement
                    mesh_dim = i
                    break

            local_tensor = param.to_local()

            if shard_placement is None:
                local_tensor.copy_(loaded_tensor)
            else:
                dim = shard_placement.dim
                mesh = param.device_mesh
                my_coordinate = mesh.get_coordinate()
                if my_coordinate is None:
                    return

                rank_in_dim = my_coordinate[mesh_dim]
                world_size_in_dim = mesh.size(mesh_dim)

                full_size = param.shape[dim]
                chunk_size = (full_size + world_size_in_dim - 1) // world_size_in_dim

                start = rank_in_dim * chunk_size
                end = min(start + chunk_size, full_size)

                if start >= full_size:
                    return

                sliced_tensor = loaded_tensor.narrow(dim, start, end - start)

                slices = [slice(None)] * local_tensor.ndim
                slices[dim] = slice(0, sliced_tensor.shape[dim])
                local_tensor[tuple(slices)].copy_(sliced_tensor)
        else:
            param.data.copy_(loaded_tensor)
