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
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from llamafactory.data.processor.pretrain import PretrainDatasetProcessor
from llamafactory.hparams import DataArguments


def _build_tokenizer(adds_bos: bool, **kwargs: bool) -> PreTrainedTokenizerFast:
    vocab = {"<s>": 0, "</s>": 1, "<unk>": 2, **{f"w{i}": i + 3 for i in range(20)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    if adds_bos:
        tokenizer.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 0)])

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, bos_token="<s>", eos_token="</s>", unk_token="<unk>", **kwargs
    )


DOCUMENT = " ".join(f"w{i}" for i in range(13))

TOKENIZER_CASES = [
    (True, {}),  # bos added by the post-processor only, `add_bos_token` is not set
    (True, {"add_bos_token": True}),
    (False, {}),
]


def _preprocess(tokenizer: PreTrainedTokenizerFast, packing: bool) -> tuple[dict[str, list], int]:
    data_args = DataArguments(template="default", cutoff_len=6, packing=packing)
    processor = PretrainDatasetProcessor(template=None, tokenizer=tokenizer, processor=None, data_args=data_args)
    return processor.preprocess_dataset({"_prompt": [[{"role": "user", "content": DOCUMENT}]]}), data_args.cutoff_len


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
@pytest.mark.parametrize("adds_bos,tokenizer_kwargs", TOKENIZER_CASES)
def test_pretrain_bos_token(adds_bos: bool, tokenizer_kwargs: dict[str, bool]):
    tokenizer = _build_tokenizer(adds_bos, **tokenizer_kwargs)
    result, cutoff_len = _preprocess(tokenizer, packing=False)
    stream = tokenizer(DOCUMENT + "</s>", add_special_tokens=False)["input_ids"]
    expected = ([tokenizer.bos_token_id] if adds_bos else []) + stream
    assert result["input_ids"] == [expected[:cutoff_len]]


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
@pytest.mark.parametrize("adds_bos,tokenizer_kwargs", TOKENIZER_CASES)
def test_pretrain_packing_bos_token(adds_bos: bool, tokenizer_kwargs: dict[str, bool]):
    tokenizer = _build_tokenizer(adds_bos, **tokenizer_kwargs)
    result, cutoff_len = _preprocess(tokenizer, packing=True)
    stream = tokenizer(DOCUMENT + "</s>", add_special_tokens=False)["input_ids"]
    content = []
    for input_ids, attention_mask in zip(result["input_ids"], result["attention_mask"]):
        assert len(input_ids) == len(attention_mask) == cutoff_len
        assert all(mask == 1 for mask in attention_mask)
        if adds_bos:
            assert input_ids[0] == tokenizer.bos_token_id
            input_ids = input_ids[1:]

        content += input_ids

    assert content == stream[: len(content)]  # no document token is lost or overwritten
