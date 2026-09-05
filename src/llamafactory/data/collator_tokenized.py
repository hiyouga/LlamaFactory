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

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from transformers import DataCollatorForSeq2Seq

from ..extras.constants import IGNORE_INDEX


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


def _resolve_pad_token_id(tokenizer: "PreTrainedTokenizer", model: "PreTrainedModel") -> int:
    r"""Resolve the padding token ID from the tokenizer, falling back to the model config."""
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        pad_id = getattr(getattr(model, "config", None), "pad_token_id", None)
    if pad_id is None:
        raise ValueError("Cannot resolve a pad token id: set `tokenizer.pad_token` before using TokenizedIdsCollator.")
    return int(pad_id)


@dataclass
class TokenizedIdsCollator(DataCollatorForSeq2Seq):
    r"""Collator for pre-tokenized LM data.

    Expects features containing `input_ids` and optionally `attention_mask`.
    Pads to batch max length with `pad_token_id`, generates labels and masks missing fields when needed.
    """

    strict: bool = True

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        pad_id = _resolve_pad_token_id(self.tokenizer, self.model)

        # Validate and compute max length
        max_len = 0
        for f in features:
            ids = f.get("input_ids")
            if ids is None or isinstance(ids, (str, bytes)) or not isinstance(ids, Iterable):
                if self.strict:
                    raise ValueError("Each feature must contain a sequence of ints in `input_ids`.")
                ids = []
            f["input_ids"] = [int(x) for x in ids]  # accept list, tuple, numpy array, tensor
            if f.get("attention_mask") is not None:
                f["attention_mask"] = [int(x) for x in f["attention_mask"]]
            max_len = max(max_len, len(f["input_ids"]))

        input_ids = []
        attention_mask = []
        labels = []
        for f in features:
            ids = f["input_ids"]
            pad_amt = max_len - len(ids)
            row_ids = ids + [pad_id] * pad_amt
            input_ids.append(row_ids)

            if f.get("attention_mask") is not None:
                if self.strict and len(f["attention_mask"]) != len(ids):
                    raise ValueError("attention_mask length must match input_ids length.")
                mask = f["attention_mask"] + [0] * pad_amt
            else:
                mask = [1] * len(ids) + [0] * pad_amt
            attention_mask.append(mask)

            labels.append(ids + [IGNORE_INDEX] * pad_amt)

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        return batch
