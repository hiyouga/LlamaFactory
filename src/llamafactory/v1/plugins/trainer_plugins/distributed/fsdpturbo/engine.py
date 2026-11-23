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

import torch

from .....accelerator.interface import Dim
from .....utils.logging import get_logger
from .....utils.types import HFModel
from ..fsdp2 import FSDP2Engine
from .gradients import clip_grad_norm_, synchronize_gradients
from .model_specs import FSDPTurboEPModelSpec, _get_model_type
from .parallel_state import get_fsdpturbo_parallel_state


logger = get_logger(__name__)


class FSDPTurboFSDP2Engine(FSDP2Engine):
    """FSDPTurbo EP adapter that reuses LlamaFactory's init/load flow.

    Design:
    - FSDPTurbo owns EP / EFSDP only.
    - LlamaFactory owns PEFT / FSDP / CP / init-load / checkpoint lifecycle.
    """

    def __init__(self, dist_config: dict, bf16: bool = False):
        self.dist_config = dist_config
        super().__init__(dist_config, bf16=bf16)
        self.parallel_state = get_fsdpturbo_parallel_state()
        self.parallel_state.initialize(self.dist_interface, self.dist_config)
        self.ep_size = self.parallel_state.ep_size
        self.ep_fsdp_size = self.parallel_state.efsdp_size
        dp_mesh = self.dist_interface.get_device_mesh(Dim.DP)
        if dp_mesh is not None:
            self.fsdp_mesh = dp_mesh
            logger.info(f"Using DP-orthogonal FSDP mesh: {self.fsdp_mesh}")

    @staticmethod
    def _get_ep_fsdp_modules(spec: FSDPTurboEPModelSpec) -> list[str]:
        if spec.ep_fsdp_modules is not None:
            return spec.ep_fsdp_modules

        # EFSDP shards the routed experts; their parent may also own a router
        # and shared experts whose gradients must be reduced by outer FSDP.
        return list(spec.ep_modules)

    def _copy_weights(self, param, loaded_tensor):
        """Copy full checkpoint tensors into mixed-mesh DTensors from the inherited loader."""
        from torch.distributed._tensor import DTensor, Shard

        if loaded_tensor.dtype != param.dtype:
            loaded_tensor = loaded_tensor.to(param.dtype)

        if isinstance(param, DTensor):
            local_tensor = param.to_local()
            shard_placements = [
                (i, placement) for i, placement in enumerate(param.placements) if isinstance(placement, Shard)
            ]

            if not shard_placements:
                local_tensor.copy_(loaded_tensor)
                return

            mesh = param.device_mesh
            my_coordinate = mesh.get_coordinate()
            if my_coordinate is None:
                return

            sliced_tensor = loaded_tensor
            for mesh_dim, shard_placement in shard_placements:
                dim = shard_placement.dim
                rank_in_dim = my_coordinate[mesh_dim]
                world_size_in_dim = mesh.size(mesh_dim)
                full_size = sliced_tensor.shape[dim]
                chunk_size = (full_size + world_size_in_dim - 1) // world_size_in_dim
                start = rank_in_dim * chunk_size
                end = min(start + chunk_size, full_size)

                if start >= full_size:
                    return

                sliced_tensor = sliced_tensor.narrow(dim, start, end - start)

            slices = [slice(None)] * local_tensor.ndim
            for _, shard_placement in shard_placements:
                dim = shard_placement.dim
                slices[dim] = slice(0, sliced_tensor.shape[dim])
            local_tensor[tuple(slices)].copy_(sliced_tensor)
            return

        param.data.copy_(loaded_tensor)

    def prepare_model_ep(self, model: HFModel) -> tuple[HFModel, set]:
        """Apply FSDPTurbo EP/EFSDP and return parameters excluded from outer FSDP."""
        from fsdp_turbo.distributed.expert_parallel.expert_fully_shard_parallel import (
            expert_fully_shard_modules,
        )
        from fsdp_turbo.distributed.expert_parallel.expert_parallel import expert_parallelize_modules
        from fsdp_turbo.fsdp_turbo_config import EPPlanConfig, FSDPPlanConfig
        from fsdp_turbo.utils.str_match import module_name_match

        # Resolve FSDPTurbo plans on the PEFT base model while preserving
        # the outer PeftModel for LoRA training and checkpointing.
        ep_target_model = model
        if self.is_lora_module_wrap(model):
            get_base_model = getattr(model, "get_base_model", None)
            if get_base_model is None:
                raise RuntimeError("FSDPTurbo could not access the base model from the LoRA-wrapped model.")

            ep_target_model = get_base_model()
            logger.info_rank0("Resolving FSDPTurbo EP/FSDP plans against the PEFT base model.")

        ep_modules = []
        if self.ep_size > 1:
            spec = FSDPTurboEPModelSpec.get(ep_target_model)
            if spec is None:
                raise ValueError(
                    f"No FSDPTurbo EP spec is registered for model_type={_get_model_type(ep_target_model)}."
                )

            ep_modules = spec.ep_modules
            ep_target_model = spec.prepare(ep_target_model)
            dispatcher = self.dist_config.get("ep_dispatcher", "eager")
            dispatcher_kwargs = {}
            if hasattr(EPPlanConfig, "use_fused_ops") and dispatcher in ("eager", "fused"):
                dispatcher_kwargs["use_fused_ops"] = dispatcher == "fused"
                dispatcher = "all2all"
            ep_plan = EPPlanConfig(
                apply_modules=ep_modules,
                dispatcher=dispatcher,
                apply_efsdp_modules=self._get_ep_fsdp_modules(spec),
                **dispatcher_kwargs,
            )
            ep_plan.gradient_divide_factor = float(self.ep_size * self.parallel_state.efsdp_size)
            mp_policy = self.get_mp_policy()
            fsdp_plan = FSDPPlanConfig(
                # EFSDP targets come from ep_plan.apply_efsdp_modules.
                apply_modules={},
                hook_modules=self.dist_config.get("hook_modules", []),
                fsdp_implementation=self.dist_config.get("fsdp_implementation", "native"),
                param_dtype=str(mp_policy.param_dtype).removeprefix("torch."),
                reduce_dtype=str(mp_policy.reduce_dtype).removeprefix("torch."),
                cast_forward_inputs=mp_policy.cast_forward_inputs,
            )
            ep_mesh = self.parallel_state.ep_mesh
            efsdp_mesh = self.parallel_state.efsdp_mesh
            if ep_mesh is None:
                raise RuntimeError("FSDPTurbo EP mesh is not initialized.")
            if self.ep_fsdp_size > 1 and efsdp_mesh is None:
                raise RuntimeError("FSDPTurbo EFSDP mesh is not initialized.")
            if self.rank == 0:
                logger.info("Applying FSDPTurbo EP backend.")
                logger.info(f"FSDPTurbo EP apply patterns: {ep_modules}")
                logger.info(f"FSDPTurbo EP device mesh: {ep_mesh}")
                logger.info(f"FSDPTurbo EP gradient divide factor: {ep_plan.gradient_divide_factor}")

            # Restore requires_grad=False after FSDPTurbo EP recreates frozen parameters.
            frozen_names = {name for name, param in ep_target_model.named_parameters() if not param.requires_grad}
            ep_target_model = expert_parallelize_modules(ep_target_model, ep_mesh, ep_plan)
            for name, param in ep_target_model.named_parameters():
                if name in frozen_names:
                    param.requires_grad_(False)

            if self.ep_fsdp_size > 1:
                if self.rank == 0:
                    logger.info(f"FSDPTurbo EFSDP apply patterns: {ep_plan.apply_efsdp_modules}")
                    logger.info(f"FSDPTurbo EFSDP device mesh: {efsdp_mesh}")
                ep_target_model = expert_fully_shard_modules(ep_target_model, efsdp_mesh, ep_plan, fsdp_plan)

        # Collect ignored params for the outer FSDP wrap
        fsdp_ignored_modules = list(self.dist_config.get("fsdp_ignored_modules", []))
        if self.ep_size > 1:
            fsdp_ignored_modules.extend(ep_modules)

        ignored_params = set()
        if fsdp_ignored_modules:
            # Resolve patterns against the same unwrapped model used by the EP
            # plan. The collected Parameter objects are shared with the outer
            # PeftModel, so they can be passed directly to its FSDP2 wrapper.
            for name, module in ep_target_model.named_modules():
                for pattern in fsdp_ignored_modules:
                    if module_name_match(pattern, name):
                        ignored_params.update(list(module.parameters(recurse=True)))

            if ignored_params and self.rank == 0:
                logger.info(f"FSDPTurbo FSDP2: Ignoring {len(ignored_params)} EP parameters in outer FSDP.")

        return model, ignored_params

    def prepare_model(self, model: HFModel) -> HFModel:
        # Apply FSDPTurbo EP first, then shard the remaining parameters with LlamaFactory FSDP2.
        model, ignored_params = self.prepare_model_ep(model)
        return super().prepare_model(model, ignored_params=ignored_params)

    def _warmup_grad_norm(self, model: HFModel) -> None:
        """Warm up collectives without stacking gradients from different DTensor meshes."""
        if self.fsdp_mesh is None:
            return

        logger.info_rank0("Warming up FSDPTurbo mixed-mesh grad norm computation...")
        for param in model.parameters():
            if param.requires_grad:
                param.grad = torch.zeros_like(param)

        with torch.no_grad():
            synchronize_gradients(model)
            clip_grad_norm_(model, 1.0)

        for param in model.parameters():
            if param.requires_grad:
                param.grad = None

        logger.info_rank0("FSDPTurbo mixed-mesh grad norm warmup completed.")
