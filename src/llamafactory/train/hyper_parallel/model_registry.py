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

"""Model-specific HyperParallel plans owned by the LlamaFactory backend."""

from collections.abc import Callable
from typing import Any

from hyper_parallel import Replicate
from hyper_parallel.distributed.context_parallel.wrappers import sdpa_hf_cp_wrapper
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec


ParallelPlan = list[dict[str, Any]]
ParallelPlanProvider = Callable[[int], ParallelPlan]


def _get_qwen3_moe_parallel_plan(cp_size: int) -> ParallelPlan:
    attention = (
        [
            {
                "match": "*.self_attn",
                "replace": {
                    "kind": "module",
                    "implementation": {
                        "_target_": "hyper_parallel.components.modules.GroupedQueryAttention",
                        "fuse_qkv": False,
                        "overlap": True,
                    },
                },
            },
        ]
        if cp_size > 1
        else []
    )
    return [
        *attention,
        {
            "match": "*.mlp.forward",
            "when": "ep",
            "region_dispatch": False,
            "replace": {
                "kind": "method",
                "implementation": {
                    "_target_": (
                        "hyper_parallel.models.qwen3_moe.adapter.distributed.expert_parallel.qwen3moe_ep_compute_fn"
                    ),
                    "use_grouped_gemm": True,
                },
            },
        },
    ]


def _get_qwen3_vl_moe_parallel_plan(_cp_size: int) -> ParallelPlan:
    replicated_vision_wrapper = (
        "hyper_parallel.models.qwen3_vl_moe.adapter.distributed.context_parallel."
        "qwen3_vl_moe_replicated_vision_cp_wrapper"
    )
    return [
        *[
            {
                "match": f"{match}.forward",
                "when": "cp",
                "params": {},
                "region_dispatch": False,
                "replace": {
                    "kind": "method",
                    "implementation": {"_target_": replicated_vision_wrapper},
                },
            }
            for match in (
                "model.visual.blocks.*.attn",
                "model.visual.merger",
                "model.visual.deepstack_merger_list",
            )
        ],
        {
            "match": "model.language_model.forward",
            "when": "cp",
            "params": {},
            "region_dispatch": False,
            "replace": {
                "kind": "method",
                "implementation": {
                    "_target_": (
                        "hyper_parallel.models.qwen3_vl_moe.adapter.distributed.context_parallel."
                        "qwen3_vl_moe_text_input_cp_wrapper"
                    )
                },
            },
        },
        {
            "match": "model.language_model.layers.*.self_attn.forward",
            "when": "cp",
            "region_dispatch": False,
            "replace": {
                "kind": "method",
                "implementation": {
                    "_target_": (
                        "hyper_parallel.models.qwen3_vl_moe.adapter.distributed.context_parallel."
                        "qwen3_vl_moe_async_ulysses_cp_wrapper"
                    )
                },
            },
        },
        {
            "match": "model.language_model.layers.*.mlp.forward",
            "when": "ep",
            "region_dispatch": False,
            "replace": {
                "kind": "method",
                "implementation": {
                    "_target_": (
                        "hyper_parallel.models.qwen3_vl_moe.adapter.distributed.expert_parallel."
                        "qwen3_vl_moe_ep_compute_fn"
                    ),
                    "use_grouped_gemm": True,
                },
            },
        },
    ]


MODEL_PARALLEL_PLAN_REGISTRY: dict[str, ParallelPlanProvider] = {
    "qwen3_moe": _get_qwen3_moe_parallel_plan,
    "qwen3_vl_moe": _get_qwen3_vl_moe_parallel_plan,
}


def apply_model_parallel_plan(hp_args: Any, model_type: str | None) -> None:
    """Prepend LlamaFactory defaults while preserving explicit user overrides."""
    provider = MODEL_PARALLEL_PLAN_REGISTRY.get(model_type or "")
    if provider is None:
        return

    hp_args.plan_overrides = [*provider(hp_args.cp_size), *(hp_args.plan_overrides or [])]


def build_model_distributed_setup(hp_args: Any, model_type: str | None) -> Any:
    """Build the model plan while leaving family-owned attention free of fallback CP hooks."""
    user_overrides = hp_args.plan_overrides or []
    apply_model_parallel_plan(hp_args, model_type)
    setup = hp_args.build_distributed_setup()
    if model_type != "qwen3_moe" or hp_args.cp_size == 1:
        return setup

    explicit_attention_hook = any(
        entry.get("match") in ("*.self_attn", "*.self_attn.forward")
        and ((entry.get("replace") or {}).get("kind") == "method" or entry.get("region_dispatch") is not None)
        for entry in user_overrides
    )
    spec = setup.plan_overrides.get("*.self_attn")
    if not explicit_attention_hook and spec is not None and spec.inner_wrapper is not None:
        if spec.inner_wrapper.callable is sdpa_hf_cp_wrapper:
            # Hyper's native-attention fallback cannot wrap a family that already owns CP.
            # Keep any user layout fields and explicit hooks for the planner to validate.
            spec.inner_wrapper = None
            spec.inner_target = None
            spec.region_dispatch = None
    return setup


def apply_replicated_vision_plan(model: Any, setup: Any) -> None:
    """Declare Qwen3-VL-MoE vision parameters without changing explicit user overrides."""
    if getattr(model.config, "model_type", None) != "qwen3_vl_moe":
        return
    mesh = setup.mesh_context
    if mesh.cp_size == mesh.ep_size == 1:
        return

    defaults = {}
    for name, module in model.named_modules():
        if name != "model.visual" and not name.startswith("model.visual."):
            continue
        params = {
            parameter_name: {"tp": Replicate(), "cp": Replicate(), "ep": Replicate()}
            for parameter_name, _ in module.named_parameters(recurse=False)
        }
        if params:
            # Parameter-only leaves avoid overlapping ownership with vision attention/merger hooks.
            defaults[name] = ModuleShardingSpec(params=params, is_boundary=False)
    setup.plan_overrides = {**defaults, **setup.plan_overrides}
