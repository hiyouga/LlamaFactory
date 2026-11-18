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

"""Compile native GS1 records without chat conversion or context truncation."""

import json
import math
from importlib import import_module
from itertools import islice
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorWithPadding

from ...data.parser import get_dataset_list
from ...extras.logging import get_logger


logger = get_logger(__name__)


def load_decision_dataset(dataset_names, data_args, compiler) -> list[dict[str, Any]]:
    # Use the validation schema shipped with this compiler, including Hub remote code.
    request_type = import_module(type(compiler).__module__).DecisionRequest
    examples = []
    if data_args.max_samples is not None and data_args.max_samples < 1:
        raise ValueError("max_samples must be positive.")
    for attr in get_dataset_list(dataset_names, data_args.dataset_dir):
        if attr.load_from != "file" or attr.formatting != "gs1" or attr.ranking or attr.num_samples is not None:
            raise ValueError(
                "The decision stage requires local JSONL datasets with formatting: gs1, without num_samples."
            )
        path = Path(attr.dataset_name)
        if not path.is_absolute():
            if isinstance(data_args.dataset_dir, dict):
                raise ValueError("Use absolute file_name paths with an inline dataset registry.")
            path = Path(data_args.dataset_dir) / path
        if path.suffix != ".jsonl":
            raise ValueError("GS1 records must be JSONL to preserve variable question and candidate IDs.")
        records = 0
        with path.open(encoding="utf-8") as stream:
            for line in islice((line for line in stream if line.strip()), data_args.max_samples):
                record = json.loads(line)
                try:
                    request = request_type.model_validate(record["request"])
                    if set(request.questions) != set(record["targets"]):
                        raise ValueError("Question and target IDs differ.")
                    compiled = compiler.compile(request)
                    for branch in compiled.branches:
                        if branch.visual_inputs is not None:
                            raise ValueError("The decision stage currently supports text requests only.")
                        target = record["targets"][branch.question_id]
                        ids, distribution = target["candidate_ids"], target["distribution"]
                        if (
                            len(ids) != len(distribution)
                            or len(set(ids)) != len(ids)
                            or set(ids) != set(branch.candidates)
                            or any(not math.isfinite(p) or p < 0 for p in distribution)
                            or not math.isclose(sum(distribution), 1.0, abs_tol=1e-6, rel_tol=0)
                        ):
                            raise ValueError("Invalid distribution or mismatched candidate IDs.")
                        by_id = dict(zip(ids, distribution))
                        examples.append(
                            {
                                "input_ids": list(compiled.state + branch.suffix),
                                "branch": branch,
                                "state_length": len(compiled.state),
                                "labels": [by_id[c] for c in branch.candidates],
                                "weight": 1.0,  # The native 4B recipe averages logical questions equally.
                            }
                        )
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"Invalid GS1 record {record.get('id', records)} in {path}: {error}") from error
                records += 1
        logger.info_rank0(f"Compiled {records} GS1 records from {path}.")
    if not examples:
        raise ValueError("The GS1 dataset contains no decision questions.")
    logger.info_rank0(f"Loaded {len(examples)} decision questions without truncation.")
    return examples


class DecisionDataCollator(DataCollatorWithPadding):
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = super().__call__([{"input_ids": feature["input_ids"]} for feature in features])
        return {
            **batch,
            "position_ids": (batch["attention_mask"].cumsum(-1) - 1).clamp_min_(0),
            "labels": pad_sequence(
                [torch.tensor(feature["labels"], dtype=torch.float32) for feature in features], batch_first=True
            ),
            "branches": [feature["branch"] for feature in features],
            "state_lengths": [feature["state_length"] for feature in features],
            "question_weights": torch.tensor([feature["weight"] for feature in features], dtype=torch.float32),
        }
