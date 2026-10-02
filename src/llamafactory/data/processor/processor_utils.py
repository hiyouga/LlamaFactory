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

import bisect
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any, Optional

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from ...extras.packages import is_sentence_transformers_available


if TYPE_CHECKING:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..template import Template


@dataclass
class DatasetProcessor(ABC):
    r"""A class for data processors."""

    template: "Template"
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]
    data_args: "DataArguments"

    @abstractmethod
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        r"""Build model inputs from the examples."""
        ...

    @abstractmethod
    def print_data_example(self, example: dict[str, list[int]]) -> None:
        r"""Print a data example to stdout."""
        ...


def search_for_fit(numbers: list[int], capacity: int) -> int:
    r"""Find the index of largest number that fits into the knapsack with the given capacity."""
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)


def greedy_knapsack(numbers: list[int], capacity: int) -> list[list[int]]:
    r"""Implement efficient greedy algorithm with binary search for the knapsack problem."""
    numbers.sort()  # sort numbers in ascending order for binary search
    knapsacks = []

    while numbers:
        current_knapsack = []
        remaining_capacity = capacity

        while True:
            index = search_for_fit(numbers, remaining_capacity)
            if index == -1:
                break  # no more numbers fit in this knapsack

            remaining_capacity -= numbers[index]  # update the remaining capacity
            current_knapsack.append(numbers.pop(index))  # add the number to knapsack

        knapsacks.append(current_knapsack)

    return knapsacks


def greedy_knapsack_indices(indices: list[int], lengths: list[int], capacity: int) -> list[list[int]]:
    r"""Run `greedy_knapsack` over `lengths` and translate the packed length groups back to `indices`.

    `greedy_knapsack` returns groups of length *values*, which is ambiguous when two different
    indices share the same length (the caller cannot tell which one was actually packed where).
    This wraps it with an index lookup so the result unambiguously identifies which sequences ended
    up in which knapsack.
    """
    length2indexes = defaultdict(list)
    for index, length in zip(indices, lengths):
        length2indexes[length].append(index)

    knapsacks = []
    for knapsack_lengths in greedy_knapsack(list(lengths), capacity):
        knapsacks.append([length2indexes[length].pop() for length in knapsack_lengths])

    return knapsacks


@cache
def _load_sentence_transformer(embedding_model: str) -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(embedding_model)


def compute_embeddings(texts: list[str], embedding_model: str) -> "np.ndarray":
    r"""Encode `texts` into L2-normalized sentence embeddings using a sentence-transformers model."""
    if not is_sentence_transformers_available():
        raise ImportError("Please install `sentence-transformers` to use semantic-aware packing.")

    model = _load_sentence_transformer(embedding_model)
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def cluster_by_similarity(embeddings: "np.ndarray", threshold: float) -> list[list[int]]:
    r"""Group embedding indices into clusters using average-linkage clustering on cosine distance.

    Two sequences end up in the same cluster when their average pairwise cosine similarity stays at
    or above `threshold` (equivalently, cosine distance at or below `1 - threshold`).

    e.g.
    ```python
    # input
    embeddings = [[1, 0], [1, 0], [0, 1]], threshold = 0.5
    # output
    [[0, 1], [2]]
    ```
    """
    num_samples = len(embeddings)
    if num_samples <= 1:
        return [list(range(num_samples))]

    distances = pdist(embeddings, metric="cosine")
    cluster_ids = fcluster(linkage(distances, method="average"), t=1 - threshold, criterion="distance")

    clusters = defaultdict(list)
    for index, cluster_id in enumerate(cluster_ids):
        clusters[cluster_id].append(index)

    return list(clusters.values())


def semantic_aware_knapsack(
    sequences: list[dict[str, Any]], capacity: int, embedding_model: str, similarity_threshold: float = 0.7
) -> list[list[int]]:
    r"""Group `sequences` into knapsacks of at most `capacity` tokens, keeping related sequences together.

    Each element of `sequences` must be a dict with a `text` key (used to compute the embedding) and
    an `input_ids` key (used as the packed length). Sequences are first clustered by cosine
    similarity of their text embedding (see `cluster_by_similarity`), then the length-based greedy
    knapsack is applied within each cluster so that every knapsack still respects `capacity` while
    grouping semantically related sequences first. Returns a list of knapsacks, each a list of
    indices into `sequences`.
    """
    texts = [sequence["text"] for sequence in sequences]
    embeddings = compute_embeddings(texts, embedding_model)
    clusters = cluster_by_similarity(embeddings, similarity_threshold)

    knapsacks = []
    for cluster in clusters:
        cluster_lengths = [len(sequences[index]["input_ids"]) for index in cluster]
        knapsacks.extend(greedy_knapsack_indices(cluster, cluster_lengths, capacity))

    return knapsacks


def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> tuple[int, int]:
    r"""Compute the real sequence length after truncation by the cutoff_len."""
    if target_len * 2 < cutoff_len:  # truncate source
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:  # truncate target
        max_target_len = cutoff_len - source_len
    else:  # truncate both
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len
