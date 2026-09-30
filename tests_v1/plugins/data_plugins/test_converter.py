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

import random

import pytest
from datasets import load_dataset

from llamafactory.v1.config.data_args import DataArguments
from llamafactory.v1.core.data_engine import DataEngine
from llamafactory.v1.plugins.data_plugins.converter import DataConverterPlugin


@pytest.mark.parametrize("num_samples", [16])
def test_alpaca_converter(num_samples: int):
    data_args = DataArguments(train_dataset="llamafactory/v1-dataset-info/tiny-supervised-dataset.yaml")
    data_engine = DataEngine(data_args.train_dataset)
    original_data = load_dataset("llamafactory/tiny-supervised-dataset", split="train")
    indexes = random.choices(range(len(data_engine)), k=num_samples)
    for index in indexes:
        print(data_engine[index])
        expected_data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "value": original_data[index]["instruction"] + original_data[index]["input"]}
                    ],
                    "loss_weight": 0.0,
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "value": original_data[index]["output"]}],
                    "loss_weight": 1.0,
                },
            ]
        }
        assert data_engine[index] == {"_dataset_name": "tiny_dataset", **expected_data}


def test_sharegpt_converter():
    example = {
        "conversations": [
            {"from": "system", "value": "System"},
            {"from": "human", "value": "User"},
            {"from": "function_call", "value": "1"},
            {"from": "observation", "value": "Observation"},
            {"from": "gpt", "value": "Assistant"},
        ]
    }
    expected_data = {
        "messages": [
            {"role": "system", "content": [{"type": "text", "value": "System"}], "loss_weight": 0.0},
            {"role": "user", "content": [{"type": "text", "value": "User"}], "loss_weight": 0.0},
            {"role": "assistant", "content": [{"type": "tool_call", "value": "1"}], "loss_weight": 1.0},
            {"role": "tool", "content": [{"type": "text", "value": "Observation"}], "loss_weight": 0.0},
            {"role": "assistant", "content": [{"type": "text", "value": "Assistant"}], "loss_weight": 1.0},
        ]
    }
    assert DataConverterPlugin("sharegpt")(example) == expected_data


def test_sharegpt_converter_multimodal():
    example = {
        "conversations": [
            {"from": "human", "value": "What is <image> and what happens in <video>?"},
            {"from": "gpt", "value": "An image and a video."},
        ],
        "images": ["/p/a.jpg"],
        "videos": ["/p/v.mp4"],
    }
    expected_data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "value": "What is "},
                    {"type": "image_url", "value": "/p/a.jpg"},
                    {"type": "text", "value": " and what happens in "},
                    {"type": "video_url", "value": "/p/v.mp4"},
                    {"type": "text", "value": "?"},
                ],
                "loss_weight": 0.0,
            },
            {"role": "assistant", "content": [{"type": "text", "value": "An image and a video."}], "loss_weight": 1.0},
        ]
    }
    assert DataConverterPlugin("sharegpt")(example) == expected_data


def test_sharegpt_converter_multiple_images_in_order():
    # images are a sample-level list consumed by <image> tags in document order across turns
    example = {
        "conversations": [
            {"from": "human", "value": "<image><image>Compare these."},
            {"from": "gpt", "value": "Done."},
        ],
        "images": ["/p/a.jpg", "/p/b.jpg"],
    }
    user = DataConverterPlugin("sharegpt")(example)["messages"][0]
    assert user["content"] == [
        {"type": "image_url", "value": "/p/a.jpg"},
        {"type": "image_url", "value": "/p/b.jpg"},
        {"type": "text", "value": "Compare these."},
    ]


def test_sharegpt_converter_no_media_unchanged():
    # backward compatibility: a scalar (non-list) image column and no tags is normalized; with no
    # media columns at all the output is byte-identical to the text-only path.
    example = {"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "yo"}]}
    assert DataConverterPlugin("sharegpt")(example) == {
        "messages": [
            {"role": "user", "content": [{"type": "text", "value": "hi"}], "loss_weight": 0.0},
            {"role": "assistant", "content": [{"type": "text", "value": "yo"}], "loss_weight": 1.0},
        ]
    }


def test_alpaca_converter_multimodal():
    example = {"instruction": "Describe <image>", "input": "", "output": "ok", "images": ["/p/a.jpg"]}
    user = DataConverterPlugin("alpaca")(example)["messages"][0]
    assert user["content"] == [
        {"type": "text", "value": "Describe "},
        {"type": "image_url", "value": "/p/a.jpg"},
    ]


def test_pair_converter_multimodal_shared_media():
    # chosen and rejected each reference the same sample-level image
    example = {
        "chosen": [
            {"role": "user", "content": "Look at <image>"},
            {"role": "assistant", "content": "good"},
        ],
        "rejected": [
            {"role": "user", "content": "Look at <image>"},
            {"role": "assistant", "content": "bad"},
        ],
        "images": ["/p/a.jpg"],
    }
    out = DataConverterPlugin("pair")(example)
    for side in ("chosen_messages", "rejected_messages"):
        assert out[side][0]["content"] == [
            {"type": "text", "value": "Look at "},
            {"type": "image_url", "value": "/p/a.jpg"},
        ]


def test_converter_media_count_mismatch():
    # more tags than media files
    with pytest.raises(ValueError, match="More <image> tags"):
        DataConverterPlugin("sharegpt")(
            {
                "conversations": [{"from": "human", "value": "<image><image>"}, {"from": "gpt", "value": "x"}],
                "images": ["/p/a.jpg"],
            }
        )
    # fewer tags than media files
    with pytest.raises(ValueError, match="Fewer <image> tags"):
        DataConverterPlugin("sharegpt")(
            {
                "conversations": [{"from": "human", "value": "<image>"}, {"from": "gpt", "value": "x"}],
                "images": ["/p/a.jpg", "/p/b.jpg"],
            }
        )


def test_converter_audio_column_and_tag():
    # an <audio> tag consumes the next path from the audios column, lifted into an audio_url block
    example = {
        "conversations": [
            {"from": "human", "value": "hear <audio>What is this?"},
            {"from": "gpt", "value": "A bell."},
        ],
        "audios": ["/p/a.wav"],
    }
    user = DataConverterPlugin("sharegpt")(example)["messages"][0]
    assert user["content"] == [
        {"type": "text", "value": "hear "},
        {"type": "audio_url", "value": "/p/a.wav"},
        {"type": "text", "value": "What is this?"},
    ]


def test_converter_audio_count_mismatch():
    # more audio tags than files
    with pytest.raises(ValueError, match="More <audio> tags"):
        DataConverterPlugin("sharegpt")(
            {
                "conversations": [{"from": "human", "value": "<audio><audio>"}, {"from": "gpt", "value": "x"}],
                "audios": ["/p/a.wav"],
            }
        )
    # fewer audio tags than files
    with pytest.raises(ValueError, match="Fewer <audio> tags"):
        DataConverterPlugin("sharegpt")(
            {
                "conversations": [{"from": "human", "value": "<audio>"}, {"from": "gpt", "value": "x"}],
                "audios": ["/p/a.wav", "/p/b.wav"],
            }
        )


@pytest.mark.parametrize("num_samples", [16])
def test_pair_converter(num_samples: int):
    data_args = DataArguments(train_dataset="llamafactory/v1-dataset-info/orca-dpo-pairs.yaml")
    data_engine = DataEngine(data_args.train_dataset)
    original_data = load_dataset("HuggingFaceH4/orca_dpo_pairs", split="train_prefs")
    indexes = random.choices(range(len(data_engine)), k=num_samples)
    for index in indexes:
        print(data_engine[index])
        print(original_data[index])
        expected_data = {
            "chosen_messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "value": original_data[index]["chosen"][0]["content"]}],
                    "loss_weight": 0.0,
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "value": original_data[index]["chosen"][1]["content"]}],
                    "loss_weight": 0.0,
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "value": original_data[index]["chosen"][2]["content"]}],
                    "loss_weight": 1.0,
                },
            ],
            "rejected_messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "value": original_data[index]["rejected"][0]["content"]}],
                    "loss_weight": 0.0,
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "value": original_data[index]["rejected"][1]["content"]}],
                    "loss_weight": 0.0,
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "value": original_data[index]["rejected"][2]["content"]}],
                    "loss_weight": 1.0,
                },
            ],
        }
        assert data_engine[index] == {"_dataset_name": "tiny_dataset", **expected_data}
