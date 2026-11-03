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


ParallelPlan = list[dict[str, Any]]
ParallelPlanProvider = Callable[[], ParallelPlan]


def _get_qwen3_moe_parallel_plan() -> ParallelPlan:
    return [
        {
            "match": "*.self_attn",
            "when": "cp",
            "inner_target": "self",
            "region_dispatch": False,
            "inner_wrapper": {
                "_target_": (
                    "hyper_parallel.models.qwen3_moe.adapter.distributed.context_parallel_async."
                    "qwen3_moe_async_ulysses_cp_wrapper"
                )
            },
        },
        {
            "match": "*.mlp",
            "when": "ep",
            "region_dispatch": False,
            "local_compute_fn": {
                "_target_": (
                    "hyper_parallel.models.qwen3_moe.adapter.distributed.expert_parallel.qwen3moe_ep_compute_fn"
                ),
                "use_grouped_gemm": True,
            },
        },
    ]


def _get_qwen3_vl_moe_parallel_plan() -> ParallelPlan:
    replicated_vision_wrapper = (
        "hyper_parallel.models.qwen3_vl_moe.adapter.distributed.context_parallel."
        "qwen3_vl_moe_replicated_vision_cp_wrapper"
    )
    return [
        *[
            {
                "match": match,
                "when": "cp",
                "params": {},
                "inner_target": "self",
                "region_dispatch": False,
                "inner_wrapper": {"_target_": replicated_vision_wrapper},
            }
            for match in (
                "model.visual.blocks.*.attn",
                "model.visual.merger",
                "model.visual.deepstack_merger_list",
            )
        ],
        {
            "match": "model.language_model",
            "when": "cp",
            "params": {},
            "inner_target": "self",
            "region_dispatch": False,
            "inner_wrapper": {
                "_target_": (
                    "hyper_parallel.models.qwen3_vl_moe.adapter.distributed.context_parallel."
                    "qwen3_vl_moe_text_input_cp_wrapper"
                )
            },
        },
        {
            "match": "model.language_model.layers.*.self_attn",
            "when": "cp",
            "inner_target": "self",
            "region_dispatch": False,
            "inner_wrapper": {
                "_target_": (
                    "hyper_parallel.models.qwen3_vl_moe.adapter.distributed.context_parallel."
                    "qwen3_vl_moe_async_ulysses_cp_wrapper"
                )
            },
        },
        {
            "match": "model.language_model.layers.*.mlp",
            "when": "ep",
            "region_dispatch": False,
            "local_compute_fn": {
                "_target_": (
                    "hyper_parallel.models.qwen3_vl_moe.adapter.distributed.expert_parallel.qwen3_vl_moe_ep_compute_fn"
                ),
                "use_grouped_gemm": True,
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

    hp_args.plan_overrides = [*provider(), *(hp_args.plan_overrides or [])]
