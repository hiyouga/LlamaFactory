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
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, build_dataset_features, infer_seqlen


if TYPE_CHECKING:
    from datasets import Dataset, Features

    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


class FeedbackDatasetProcessor(DatasetProcessor):
    @staticmethod
    def get_dataset_features(dataset: "Dataset") -> "Features":
        r"""Return the schema for the fields emitted by ``preprocess_dataset``."""
        return build_dataset_features(
            dataset,
            {
                "input_ids": Sequence(Value("int32")),
                "attention_mask": Sequence(Value("int8")),
                "labels": Sequence(Value("int64")),
                "kl_input_ids": Sequence(Value("int32")),
                "kl_attention_mask": Sequence(Value("int8")),
                "kl_labels": Sequence(Value("int64")),
                "kto_tags": Value("bool"),
            },
        )

    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        kl_response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> tuple[list[int], list[int], list[int], list[int], bool]:
        if response[0]["content"]:  # desired example
            kto_tag = True
            messages = prompt + [response[0]]
        else:  # undesired example
            kto_tag = False
            messages = prompt + [response[1]]

        if kl_response[0]["content"]:
            kl_messages = prompt + [kl_response[0]]
        else:
            kl_messages = prompt + [kl_response[1]]

        messages = self.template.mm_plugin.process_messages(messages, images, videos, audios, self.processor)
        kl_messages = self.template.mm_plugin.process_messages(kl_messages, images, videos, audios, self.processor)
        prompt_ids, response_ids = self.template.encode_oneturn(self.tokenizer, messages, system, tools)
        kl_prompt_ids, kl_response_ids = self.template.encode_oneturn(self.tokenizer, kl_messages, system, tools)

        if self.template.efficient_eos:
            response_ids += [self.tokenizer.eos_token_id]
            kl_response_ids += [self.tokenizer.eos_token_id]

        prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        kl_prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            kl_prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )

        source_len, target_len = infer_seqlen(len(prompt_ids), len(response_ids), self.data_args.cutoff_len)
        prompt_ids = prompt_ids[:source_len]
        response_ids = response_ids[:target_len]
        kl_source_len, kl_target_len = infer_seqlen(
            len(kl_prompt_ids), len(kl_response_ids), self.data_args.cutoff_len
        )
        kl_prompt_ids = kl_prompt_ids[:kl_source_len]
        kl_response_ids = kl_response_ids[:kl_target_len]

        input_ids = prompt_ids + response_ids
        labels = [IGNORE_INDEX] * source_len + response_ids
        kl_input_ids = kl_prompt_ids + kl_response_ids
        kl_labels = [IGNORE_INDEX] * kl_source_len + kl_response_ids
        return input_ids, labels, kl_input_ids, kl_labels, kto_tag

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # Keep these output keys in sync with ``get_dataset_features`` above.
        # Creates mismatched pairs of prompts and completions for the KL dataset by adding a +1 offset to the order of completions.
        kl_response = [examples["_response"][-1]] + examples["_response"][:-1]
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels, kl_input_ids, kl_labels, kto_tag = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                kl_response=kl_response[i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["kl_input_ids"].append(kl_input_ids)
            model_inputs["kl_attention_mask"].append([1] * len(kl_input_ids))
            model_inputs["kl_labels"].append(kl_labels)
            model_inputs["kto_tags"].append(kto_tag)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        desirable_num = sum([1 for tag in model_inputs["kto_tags"] if tag])
        undesirable_num = len(model_inputs["kto_tags"]) - desirable_num
        if desirable_num == 0 or undesirable_num == 0:
            logger.warning_rank0("Your dataset only has one preference type.")

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")
