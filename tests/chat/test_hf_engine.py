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

from inspect import signature

import torch

from llamafactory.chat.hf_engine import HuggingfaceEngine


class DummyBatch(dict):
    def to(self, device: str) -> "DummyBatch":
        return self


class DummyTokenizer:
    max_length: int

    def __call__(
        self,
        batch_input: list[str],
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
        add_special_tokens: bool,
    ) -> DummyBatch:
        self.max_length = max_length
        return DummyBatch({"attention_mask": torch.ones(len(batch_input), max_length, dtype=torch.long)})


class DummyPretrainedModel:
    device = "cpu"


class DummyConfig:
    max_position_embeddings = 3


class DummyModel:
    pretrained_model = DummyPretrainedModel()
    config = DummyConfig()

    def __call__(self, **inputs):
        batch_size, seq_len = inputs["attention_mask"].shape
        values = torch.arange(batch_size * seq_len, dtype=torch.float).reshape(batch_size, seq_len)
        return (values,)


def test_hf_engine_input_kwargs_default_to_none():
    for method_name in ["_process_args", "_chat", "_stream_chat", "_get_scores"]:
        method = HuggingfaceEngine.__dict__[method_name].__func__
        assert signature(method).parameters["input_kwargs"].default is None


def test_get_scores_does_not_mutate_input_kwargs():
    input_kwargs = {"max_length": 2}
    tokenizer = DummyTokenizer()

    scores = HuggingfaceEngine._get_scores(DummyModel(), tokenizer, ["first", "second"], input_kwargs)

    assert input_kwargs == {"max_length": 2}
    assert tokenizer.max_length == 2
    assert scores == [1.0, 3.0]
