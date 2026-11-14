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

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Optional

from datasets import Sequence, Value

from ...extras import logging
from ..data_utils import Role
from .processor_utils import DatasetProcessor, build_dataset_features, infer_seqlen


if TYPE_CHECKING:
    from datasets import Dataset, Features

    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


class UnsupervisedDatasetProcessor(DatasetProcessor):
    @staticmethod
    def get_dataset_features(dataset: "Dataset") -> "Features":
        r"""Return the schema for the fields emitted by ``preprocess_dataset``."""
        return build_dataset_features(
            dataset,
            {
                "input_ids": Sequence(Value("int32")),
                "attention_mask": Sequence(Value("int8")),
                "labels": Sequence(Value("int64")),
            },
        )

    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> tuple[list[int], list[int]]:
        if len(response) == 1:
            messages = prompt + response
        else:
            messages = prompt + [{"role": Role.ASSISTANT.value, "content": ""}]

        messages = self.template.mm_plugin.process_messages(messages, images, videos, audios, self.processor)
        input_ids, labels = self.template.encode_oneturn(self.tokenizer, messages, system, tools)
        if self.template.efficient_eos:
            labels += [self.tokenizer.eos_token_id]

        input_ids, _ = self.template.mm_plugin.process_token_ids(
            input_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        source_len, target_len = infer_seqlen(len(input_ids), len(labels), self.data_args.cutoff_len)
        input_ids = input_ids[:source_len]
        labels = labels[:target_len]
        return input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # Keep these output keys in sync with ``get_dataset_features`` above.
        # build inputs with format `<bos> X` and labels with format `Y <eos>`
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(self.tokenizer.decode(example["labels"], skip_special_tokens=False)))
