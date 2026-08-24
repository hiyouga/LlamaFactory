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
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from llamafactory.data.collator_tokenized import TokenizedIdsCollator, _resolve_pad_token_id
from llamafactory.data.tokenized_parquet import load_tokenized_parquet_dataset
from llamafactory.extras.constants import IGNORE_INDEX


class _Tok:
    pad_token_id = 99
    eos_token_id = 2


def test_load_and_collate(tmp_path):
    path = str(tmp_path / "t.parquet")
    pq.write_table(pa.table({"input_ids": [[5, 6, 7], [8, 9]], "attention_mask": [[1, 1, 1], [1, 1]]}), path)
    rows = list(load_tokenized_parquet_dataset([str(tmp_path / "missing.parquet"), path]))
    assert len(rows) == 2

    batch = TokenizedIdsCollator(tokenizer=_Tok(), model=None)(rows)
    assert batch["input_ids"].tolist() == [[5, 6, 7], [8, 9, 99]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["labels"].tolist() == [[5, 6, 7], [8, 9, IGNORE_INDEX]]


def test_collator_accepts_numpy():
    batch = TokenizedIdsCollator(tokenizer=_Tok(), model=None)(
        [{"input_ids": np.array([3, 4])}, {"input_ids": np.array([5])}]
    )
    assert batch["input_ids"].tolist() == [[3, 4], [5, 99]]


def test_pad_id_required():
    class _NoPad:
        pad_token_id = None
        eos_token_id = None

    with pytest.raises(ValueError):
        _resolve_pad_token_id(_NoPad(), None)
