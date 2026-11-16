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

from llamafactory.data import Role
from llamafactory.data.converter import get_dataset_converter
from llamafactory.data.parser import DatasetAttr
from llamafactory.hparams import DataArguments


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
def test_alpaca_converter():
    dataset_attr = DatasetAttr("hf_hub", "llamafactory/tiny-supervised-dataset")
    data_args = DataArguments()
    example = {
        "instruction": "Solve the math problem.",
        "input": "3 + 4",
        "output": "The answer is 7.",
    }
    dataset_converter = get_dataset_converter("alpaca", dataset_attr, data_args)
    assert dataset_converter(example) == {
        "_prompt": [{"role": Role.USER.value, "content": "Solve the math problem.\n3 + 4"}],
        "_response": [{"role": Role.ASSISTANT.value, "content": "The answer is 7."}],
        "_system": "",
        "_tools": "",
        "_images": None,
        "_videos": None,
        "_audios": None,
    }


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
def test_sharegpt_converter():
    dataset_attr = DatasetAttr("hf_hub", "llamafactory/tiny-supervised-dataset")
    data_args = DataArguments()
    example = {
        "conversations": [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": "Solve the math problem.\n3 + 4"},
            {"from": "gpt", "value": "The answer is 7."},
        ]
    }
    dataset_converter = get_dataset_converter("sharegpt", dataset_attr, data_args)
    assert dataset_converter(example) == {
        "_prompt": [{"role": Role.USER.value, "content": "Solve the math problem.\n3 + 4"}],
        "_response": [{"role": Role.ASSISTANT.value, "content": "The answer is 7."}],
        "_system": "You are a helpful assistant.",
        "_tools": "",
        "_images": None,
        "_videos": None,
        "_audios": None,
    }


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
def test_sharegpt_converter_preserves_video_frame_paths(tmp_path):
    dataset_attr = DatasetAttr("file", "sample.json", formatting="sharegpt", videos="videos")
    example = {
        "conversations": [
            {"from": "human", "value": "Compare <video> with <video>."},
            {"from": "gpt", "value": "They show different scenes."},
        ],
        "videos": [["first.jpg", "missing.jpg"], ["second.jpg"]],
    }
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    for media_dir in (first_dir, second_dir):
        media_dir.mkdir()
        (media_dir / "first.jpg").touch()
        (media_dir / "second.jpg").touch()

    first_converter = get_dataset_converter("sharegpt", dataset_attr, DataArguments(media_dir=str(first_dir)))
    second_converter = get_dataset_converter("sharegpt", dataset_attr, DataArguments(media_dir=str(second_dir)))

    first_result = first_converter(example)
    second_result = second_converter(example)

    assert first_result["_videos"] == [[str(first_dir / "first.jpg"), "missing.jpg"], [str(first_dir / "second.jpg")]]
    assert second_result["_videos"] == [
        [str(second_dir / "first.jpg"), "missing.jpg"],
        [str(second_dir / "second.jpg")],
    ]
    assert example["videos"] == [["first.jpg", "missing.jpg"], ["second.jpg"]]
