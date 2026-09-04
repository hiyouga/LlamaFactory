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


import numpy as np
import pytest

from llamafactory.data.processor.processor_utils import cluster_by_similarity, greedy_knapsack_indices, infer_seqlen


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.parametrize(
    "test_input,test_output",
    [
        ((3000, 2000, 1000), (600, 400)),
        ((2000, 3000, 1000), (400, 600)),
        ((1000, 100, 1000), (900, 100)),
        ((100, 1000, 1000), (100, 900)),
        ((100, 500, 1000), (100, 500)),
        ((500, 100, 1000), (500, 100)),
        ((10, 10, 1000), (10, 10)),
    ],
)
def test_infer_seqlen(test_input: tuple[int, int, int], test_output: tuple[int, int]):
    assert test_output == infer_seqlen(*test_input)


@pytest.mark.runs_on(["cpu", "mps"])
def test_greedy_knapsack_indices_preserves_index_identity():
    # two different sequences (indices 11 and 13) share the same length on purpose: a naive
    # length-keyed lookup shared across independent groups could swap them.
    indices = [10, 11, 12, 13]
    length_by_index = {10: 50, 11: 50, 12: 30, 13: 80}
    lengths = [length_by_index[index] for index in indices]

    knapsacks = greedy_knapsack_indices(indices, lengths, capacity=100)

    packed_indices = sorted(index for knapsack in knapsacks for index in knapsack)
    assert packed_indices == sorted(indices)
    for knapsack in knapsacks:
        assert sum(length_by_index[index] for index in knapsack) <= 100


@pytest.mark.runs_on(["cpu", "mps"])
def test_cluster_by_similarity_groups_similar_vectors():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ]
    )
    clusters = cluster_by_similarity(embeddings, threshold=0.8)
    cluster_sets = sorted(sorted(cluster) for cluster in clusters)
    assert cluster_sets == [[0, 1], [2]]


@pytest.mark.runs_on(["cpu", "mps"])
def test_cluster_by_similarity_single_sample():
    embeddings = np.array([[1.0, 0.0]])
    assert cluster_by_similarity(embeddings, threshold=0.5) == [[0]]
