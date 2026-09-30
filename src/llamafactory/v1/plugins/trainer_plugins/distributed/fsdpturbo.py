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

from collections.abc import Callable

import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from ....accelerator.interface import Dim, DistributedInterface
from ....utils.logging import get_logger
from ....utils.types import HFModel
from .fsdp2 import FSDP2Engine


logger = get_logger(__name__)


class FSDPTurboParallelState:
    """Own FSDPTurbo's expert topology without extending LlamaFactory's global interface."""

    EDP = "edp"
    EFSDP = "efsdp"
    EP = "ep"
    EXPERT_CP = "expert_cp"

    def __init__(self) -> None:
        self._initialized = False
        self.dp_size = 1
        self.cp_size = 1
        self.ep_size = 1
        self.efsdp_size = 1
        self.edp_size = 1
        self.expert_mesh: DeviceMesh | None = None
        self.edp_mesh: DeviceMesh | None = None
        self.efsdp_mesh: DeviceMesh | None = None
        self.ep_mesh: DeviceMesh | None = None
        self.expert_cp_mesh: DeviceMesh | None = None

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self, dist_interface: DistributedInterface, dist_config: dict) -> None:
        dp_size = dist_interface.get_world_size(Dim.DP)
        cp_size = dist_interface.strategy.cp_size
        ep_size = int(dist_config.get("ep_size", 1))

        if ep_size < 1:
            raise ValueError(f"ep_size must be positive, got {ep_size}.")
        if dp_size % ep_size != 0:
            raise ValueError(f"dp_size must be divisible by ep_size, got {dp_size} % {ep_size} != 0.")

        topology = (dp_size, cp_size, ep_size)
        if self._initialized:
            current_topology = (self.dp_size, self.cp_size, self.ep_size)
            if topology != current_topology:
                raise RuntimeError(
                    f"FSDPTurbo parallel state is already initialized with {current_topology}, got {topology}."
                )
            return

        self.dp_size = dp_size
        self.cp_size = cp_size
        self.ep_size = ep_size

        if ep_size > 1:
            self.efsdp_size = dp_size // ep_size
            self.edp_size = dp_size // (ep_size * self.efsdp_size)
            if dist_interface.get_device_mesh(Dim.DP) is None:
                raise RuntimeError("FSDPTurbo expert parallelism requires an initialized distributed device mesh.")

            self.expert_mesh = init_device_mesh(
                device_type=dist_interface.current_device.type,
                mesh_shape=(self.edp_size, self.efsdp_size, self.ep_size, self.cp_size),
                mesh_dim_names=(self.EDP, self.EFSDP, self.EP, self.EXPERT_CP),
            )
            self.edp_mesh = self.expert_mesh[self.EDP]
            self.efsdp_mesh = self.expert_mesh[self.EFSDP]
            self.ep_mesh = self.expert_mesh[self.EP]
            self.expert_cp_mesh = self.expert_mesh[self.EXPERT_CP]

        self._initialized = True


_FSDPTURBO_PARALLEL_STATE = FSDPTurboParallelState()


def get_fsdpturbo_parallel_state() -> FSDPTurboParallelState:
    return _FSDPTURBO_PARALLEL_STATE


def _grad_to_local_fp32(grad: torch.Tensor) -> torch.Tensor:
    from torch.distributed._tensor import DTensor

    local_grad = grad.to_local() if isinstance(grad, DTensor) else grad
    return local_grad.detach().to(torch.float32)


def _local_pth_sum(parameters: list[torch.nn.Parameter], norm_type: float, device: torch.device) -> torch.Tensor:
    total = torch.zeros((), device=device, dtype=torch.float32)
    for param in parameters:
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        total = total + torch.norm(_grad_to_local_fp32(grad), p=norm_type).pow(norm_type)
    return total


def _allreduce_sum_(value: torch.Tensor, groups: list[object]) -> torch.Tensor:
    import torch.distributed as dist

    for group in groups:
        if group is not None:
            dist.all_reduce(value, op=dist.ReduceOp.SUM, group=group)
    return value


def clip_grad_norm_(model: HFModel, max_norm: float, **kwargs) -> float:
    """CP-aware grad norm clipping for FSDPTurbo EP + EFSDP + outer FSDP2.

    Avoids torch.nn.utils.get_total_norm() since mixed DTensor meshes
    (`dp` vs `efsdp`/`ep`) may hit DTensor stack propagation failures.
    """
    from torch.distributed._tensor import DTensor

    norm_type = float(kwargs.get("norm_type", 2.0))
    dist_interface = DistributedInterface()
    parallel_state = get_fsdpturbo_parallel_state()
    if not parallel_state.initialized:
        raise RuntimeError("FSDPTurbo parallel state must be initialized before clipping gradients.")

    device = dist_interface.current_device
    dp_group = dist_interface.get_group(Dim.DP)
    cp_group = dist_interface.get_group(Dim.CP) if dist_interface.strategy.cp_size > 1 else None
    ep_group = parallel_state.ep_mesh.get_group() if parallel_state.ep_mesh is not None else None
    efsdp_group = parallel_state.efsdp_mesh.get_group() if parallel_state.efsdp_mesh is not None else None
    expert_cp_group = (
        parallel_state.expert_cp_mesh.get_group()
        if parallel_state.expert_cp_mesh is not None and parallel_state.cp_size > 1
        else None
    )

    ep_params: list[torch.nn.Parameter] = []
    non_ep_params: list[torch.nn.Parameter] = []
    for param in model.parameters():
        grad = getattr(param, "grad", None)
        if grad is None:
            continue

        mesh_names = set(getattr(getattr(grad, "device_mesh", None), "mesh_dim_names", ()) or ())
        is_ep_side = isinstance(grad, DTensor) and bool(mesh_names & {parallel_state.EP, parallel_state.EFSDP})
        if is_ep_side:
            ep_params.append(param)
        else:
            non_ep_params.append(param)

    if not ep_params and not non_ep_params:
        return 0.0

    total_pth = torch.zeros((), device=device, dtype=torch.float32)
    if non_ep_params:
        non_ep_pth = _local_pth_sum(non_ep_params, norm_type, device)
        total_pth = total_pth + _allreduce_sum_(non_ep_pth, [dp_group, cp_group])
    if ep_params:
        ep_pth = _local_pth_sum(ep_params, norm_type, device)
        total_pth = total_pth + _allreduce_sum_(ep_pth, [efsdp_group, ep_group, expert_cp_group])

    total_norm = total_pth.pow(1.0 / norm_type)
    clip_coef = min(max_norm / (float(total_norm.item()) + 1e-6), 1.0)
    if clip_coef < 1.0:
        for param in ep_params + non_ep_params:
            grad = getattr(param, "grad", None)
            if grad is not None:
                grad.detach().mul_(clip_coef)

    return float(total_norm.item())


def _get_model_type(model: HFModel) -> str | None:
    return getattr(getattr(model, "config", None), "model_type", None)


class FSDPTurboEPModelSpec:
    _registry: dict[str, "FSDPTurboEPModelSpec"] = {}

    def __init__(
        self,
        ep_modules: list[str],
        ep_fsdp_modules: list[str] | None = None,
        prepare_fn: Callable[[HFModel], HFModel] | None = None,
    ) -> None:
        self.ep_modules = ep_modules
        self.ep_fsdp_modules = ep_fsdp_modules
        self.prepare_fn = prepare_fn

    @classmethod
    def register(
        cls,
        model_type: str,
        ep_modules: list[str],
        ep_fsdp_modules: list[str] | None = None,
    ):
        def decorator(fn):
            cls._registry[model_type] = cls(
                ep_modules=ep_modules,
                ep_fsdp_modules=ep_fsdp_modules,
                prepare_fn=fn,
            )
            return fn

        return decorator

    @classmethod
    def get(cls, model: HFModel) -> "FSDPTurboEPModelSpec | None":
        model_type = _get_model_type(model)
        if model_type is None:
            return None
        return cls._registry.get(model_type)

    def prepare(self, model: HFModel) -> HFModel:
        if self.prepare_fn is None:
            return model
        return self.prepare_fn(model)


@FSDPTurboEPModelSpec.register(
    "qwen3_moe",
    ep_modules=["model.layers.{*}.mlp.experts"],
    ep_fsdp_modules=["model.layers.{*}.mlp"],
)
def _prepare_qwen3_moe_for_ep(model: HFModel) -> HFModel:
    prepared = 0
    for module in model.modules():
        if not all(hasattr(module, attr) for attr in ("gate_up_proj", "down_proj", "hidden_dim", "num_experts")):
            continue

        # FSDPTurbo's eager EP dispatcher expects sparse expert blocks to expose `hidden_size`.
        if not hasattr(module, "hidden_size"):
            module.hidden_size = module.hidden_dim
        prepared += 1

    if prepared:
        logger.info_rank0(f"FSDPTurbo EP adapter: prepared {prepared} sparse expert modules for Transformers 5.x.")
    else:
        logger.info_rank0("FSDPTurbo EP adapter did not find a sparse expert module requiring preparation.")
    return model


@FSDPTurboEPModelSpec.register(
    "qwen3_5_moe",
    ep_modules=["model.language_model.layers.{*}.mlp.experts"],
    ep_fsdp_modules=["model.language_model.layers.{*}.mlp"],
)
def _prepare_qwen3_5_moe_for_ep(model: HFModel) -> HFModel:
    return model


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

        ep_fsdp_modules = []
        for module in spec.ep_modules:
            if module.endswith(".experts"):
                ep_fsdp_modules.append(module.removesuffix(".experts"))
            else:
                ep_fsdp_modules.append(module)
        return ep_fsdp_modules

    def shard_model(self, model: HFModel) -> HFModel:
        """Set storage dtype before FSDP materialization without leaking backend config into ModelEngine."""
        param_dtype = torch.bfloat16 if self.mixed_precision == "bf16" else torch.float32
        model = model.to(param_dtype)
        logger.info_rank0(f"Using {param_dtype} for FSDPTurbo full tuning.")
        return super().shard_model(model)

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
            ep_plan = EPPlanConfig(
                apply_modules=ep_modules,
                dispatcher=self.dist_config.get("ep_dispatcher", "eager"),
                apply_efsdp_modules=self._get_ep_fsdp_modules(spec),
            )
            ep_plan.gradient_divide_factor = float(self.ep_size * self.parallel_state.efsdp_size)
            fsdp_plan = FSDPPlanConfig(
                # FSDPTurbo uses this plan only to place EFSDP hooks and select its
                # implementation. EFSDP targets come from ep_plan.apply_efsdp_modules.
                apply_modules={},
                hook_modules=self.dist_config.get("hook_modules", []),
                fsdp_implementation=self.dist_config.get("fsdp_implementation", "native"),
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

            ep_target_model = expert_parallelize_modules(ep_target_model, ep_mesh, ep_plan)

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
            clip_grad_norm_(model, 1.0)

        for param in model.parameters():
            if param.requires_grad:
                param.grad = None

        logger.info_rank0("FSDPTurbo mixed-mesh grad norm warmup completed.")
