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

import itertools
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

from ..extras import logging


if TYPE_CHECKING:
    from datasets import IterableDataset


logger = logging.get_logger(__name__)


def _iter_parquet_rows(paths: list[str], ids_key: str, mask_key: str | None) -> Iterable[dict[str, Any]]:
    r"""Iterate over rows from multiple Parquet files, yielding pre-tokenized samples."""
    for path in paths:
        try:
            pf = pq.ParquetFile(path)
        except FileNotFoundError:
            logger.warning(f"Parquet file not found, skipping: {path}")
            continue

        with pf:
            for i in range(pf.num_row_groups):
                table: pa.Table = pf.read_row_group(i)
                ids_col = table[ids_key]
                mask_col = table[mask_key] if mask_key and mask_key in table.column_names else None
                ids_py = ids_col.to_pylist()
                mask_py = mask_col.to_pylist() if mask_col is not None else itertools.repeat(None)
                for ids, mask in zip(ids_py, mask_py):
                    yield {
                        "input_ids": list(ids) if isinstance(ids, (list, tuple)) else ids,
                        **(
                            {"attention_mask": (list(mask) if isinstance(mask, (list, tuple)) else mask)}
                            if mask is not None
                            else {}
                        ),
                    }


def load_tokenized_parquet_dataset(
    data_files: list[str],
    ids_key: str = "input_ids",
    mask_key: str | None = "attention_mask",
) -> "IterableDataset":
    r"""Create a streaming HF IterableDataset over pre-tokenized Parquet samples.

    Args:
        data_files: List of local Parquet file paths.
        ids_key: Column name for input token IDs.
        mask_key: Column name for attention mask (optional).

    Returns:
        IterableDataset yielding dictionaries with `input_ids` and optionally `attention_mask`.

    Note:
        Always streams row groups to avoid materializing large corpora in memory.
    """
    from datasets import IterableDataset

    if not data_files:
        raise ValueError("data_files must be a non-empty list of Parquet paths")

    logger.info_rank0(f"Building streaming dataset from {len(data_files)} parquet file(s)")
    return IterableDataset.from_generator(
        _iter_parquet_rows, gen_kwargs={"paths": data_files, "ids_key": ids_key, "mask_key": mask_key}
    )  # type: ignore
