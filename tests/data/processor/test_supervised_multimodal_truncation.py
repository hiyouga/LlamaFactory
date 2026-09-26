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

import os

import pytest
from PIL import Image

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer


TINY_QWEN2_VL = os.getenv("TINY_QWEN2_VL", "Qwen/Qwen2-VL-2B-Instruct")


def _build_processor(cutoff_len: int) -> SupervisedDatasetProcessor:
    model_args, data_args, *_ = get_infer_args(
        {"model_name_or_path": TINY_QWEN2_VL, "template": "qwen2_vl", "cutoff_len": cutoff_len}
    )
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    return SupervisedDatasetProcessor(
        template=template,
        tokenizer=tokenizer_module["tokenizer"],
        processor=tokenizer_module["processor"],
        data_args=data_args,
    )


def _single_image_example(image: Image.Image) -> dict[str, list]:
    return {
        "_prompt": [[{"role": "user", "content": "<image>Describe this image in detail please."}]],
        "_response": [[{"role": "assistant", "content": "A green square."}]],
        "_system": [None],
        "_tools": [None],
        "_images": [[image]],
        "_videos": [None],
        "_audios": [None],
    }


@pytest.mark.runs_on(["cpu", "mps"])
def test_multimodal_example_truncated_inside_image_span_is_dropped():
    """Truncation must not cut an image token span in half.

    `cutoff_len` smaller than the expanded image would leave `input_ids` with
    fewer image tokens than the intact image's grid implies, and `get_rope_index`
    then fails with a shape mismatch during training. Such examples are dropped
    instead of emitting inconsistent features.
    """
    processor = _build_processor(cutoff_len=16)
    image = Image.new("RGB", (448, 448), (10, 200, 10))
    model_inputs = processor.preprocess_dataset(_single_image_example(image))

    assert model_inputs["input_ids"] == []
    assert model_inputs["images"] == []


@pytest.mark.runs_on(["cpu", "mps"])
def test_multimodal_example_within_cutoff_is_kept():
    """A sample that fits must be encoded normally, image tokens intact."""
    processor = _build_processor(cutoff_len=8192)
    image = Image.new("RGB", (448, 448), (10, 200, 10))
    model_inputs = processor.preprocess_dataset(_single_image_example(image))

    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    num_image_tokens = sum(1 for token in model_inputs["input_ids"][0] if token == image_token_id)

    assert len(model_inputs["input_ids"]) == 1
    assert num_image_tokens > 0
