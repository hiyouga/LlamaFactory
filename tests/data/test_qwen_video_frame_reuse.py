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
from PIL import Image

from llamafactory.data.mm_plugin import get_mm_plugin


@pytest.mark.parametrize("plugin_name", ["qwen2_vl", "qwen3_vl"])
@pytest.mark.parametrize("frame_count", [4, 5])
@pytest.mark.parametrize("frame_input", ["images", "paths"])
def test_regularizing_qwen_video_frames_does_not_modify_input(
    plugin_name: str, frame_count: int, frame_input, tmp_path
):
    plugin = get_mm_plugin(name=plugin_name, image_token="<|image_pad|>", video_token="<|video_pad|>")
    frames = [Image.new("RGB", (32, 32), (i, 0, 0)) for i in range(frame_count)]
    if frame_input == "paths":
        paths = []
        for index, frame in enumerate(frames):
            path = tmp_path / f"{index}.png"
            frame.save(path)
            paths.append(str(path))

        frames = paths

    original_frames = frames.copy()

    result = plugin._regularize_videos([frames], video_fps=2.0, image_min_pixels=16 * 16, image_max_pixels=256 * 256)

    assert len(frames) == frame_count
    assert all(frame is original for frame, original in zip(frames, original_frames))
    assert len(result["videos"][0]) == frame_count + frame_count % 2
    if frame_count % 2:
        assert result["videos"][0][-1].tobytes() == result["videos"][0][-2].tobytes()

    assert result["frames_indices"] == [list(range(frame_count))]
    assert result["durations"] == [frame_count / 2.0]


@pytest.fixture
def qwen3_processor():
    transformers = pytest.importorskip("transformers")
    qwen3_processor = getattr(transformers, "Qwen3VLProcessor", None)
    qwen3_video_processor = getattr(transformers, "Qwen3VLVideoProcessor", None)
    if qwen3_processor is None or qwen3_video_processor is None:
        pytest.skip("Qwen3-VL processor is not available in this Transformers version")

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import Qwen2TokenizerFast, Qwen2VLImageProcessor

    vocabulary = {"[UNK]": 0, "0": 1, "2": 2, ".": 3, "<": 4, ">": 5, "seconds": 6}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = Qwen2TokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    processor = qwen3_processor(
        image_processor=Qwen2VLImageProcessor(), video_processor=qwen3_video_processor(), tokenizer=tokenizer
    )
    processor.video_fps = 2.0
    return processor


def test_qwen3_vl_reusing_video_frames_keeps_timestamps(qwen3_processor):
    processor = qwen3_processor
    plugin = get_mm_plugin(name="qwen3_vl", image_token="<|image_pad|>", video_token="<|video_pad|>")
    frames = [Image.new("RGB", (112, 112), (i * 30, 0, 0)) for i in range(5)]
    messages = [{"role": "user", "content": "<video>Describe this clip."}]

    first = plugin.process_messages(messages, [], [frames], [], processor)
    second = plugin.process_messages(messages, [], [frames], [], processor)

    assert "<2.0 seconds>" in first[0]["content"]
    assert first == second
    assert len(frames) == 5


def test_qwen3_vl_supervised_preprocessing_is_repeatable(qwen3_processor):
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.hparams import DataArguments

    tokenizer = qwen3_processor.tokenizer
    data_args = DataArguments(template="qwen3_vl_nothink", cutoff_len=8192)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_processor = SupervisedDatasetProcessor(
        template=template, tokenizer=tokenizer, processor=qwen3_processor, data_args=data_args
    )
    frames = [Image.new("RGB", (112, 112), (i * 30, 0, 0)) for i in range(5)]
    examples = {
        "_prompt": [[{"role": "user", "content": "<video>Describe this clip."}]],
        "_response": [[{"role": "assistant", "content": "A short clip."}]],
        "_system": [""],
        "_tools": [""],
        "_images": [None],
        "_videos": [[frames]],
        "_audios": [None],
    }

    first = dataset_processor.preprocess_dataset(examples)
    second = dataset_processor.preprocess_dataset(examples)

    assert len(first["input_ids"]) == 1
    assert first["input_ids"] == second["input_ids"]
    assert first["labels"] == second["labels"]
    assert len(frames) == 5
