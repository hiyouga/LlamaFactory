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

import pytest
import torch

from llamafactory.v1.plugins.model_plugins.kernels.ops.mlp.cuda_fused_moe import _compute_expert_scatter_index


def _old_compute_expert_scatter_index(expert_index: torch.Tensor) -> torch.Tensor:
    return expert_index.flatten().argsort(stable=True).argsort().int().view(expert_index.shape)


@pytest.mark.parametrize(
    ("num_tokens", "top_k", "num_experts"),
    [
        (0, 2, 4),
        (1, 1, 1),
        (8, 2, 4),
        (17, 4, 8),
        (128, 8, 16),
    ],
)
def test_compute_expert_scatter_index_matches_old_expression(num_tokens: int, top_k: int, num_experts: int):
    expert_index = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int64)

    scatter_index = _compute_expert_scatter_index(expert_index)

    assert torch.equal(scatter_index, _old_compute_expert_scatter_index(expert_index))
    assert scatter_index.dtype == torch.int32
    assert scatter_index.device == expert_index.device


def test_compute_expert_scatter_index_preserves_stable_expert_order():
    expert_index = torch.tensor(
        [
            [2, 1],
            [2, 0],
            [1, 2],
            [0, 1],
        ],
        dtype=torch.int64,
    )

    scatter_index = _compute_expert_scatter_index(expert_index)
    sorted_flat_positions = torch.argsort(scatter_index.flatten()).tolist()
    sorted_experts = expert_index.flatten()[sorted_flat_positions].tolist()

    assert sorted_experts == sorted(expert_index.flatten().tolist())
    for expert in expert_index.unique().tolist():
        original_positions = (expert_index.flatten() == expert).nonzero(as_tuple=False).flatten().tolist()
        sorted_positions = [pos for pos in sorted_flat_positions if expert_index.flatten()[pos].item() == expert]
        assert sorted_positions == original_positions
