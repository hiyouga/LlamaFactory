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

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq
from llamafactory.data.mm_plugin import get_mm_plugin
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
from llamafactory.extras.constants import IGNORE_INDEX


IMAGE_TOKEN_ID = 101
VIDEO_TOKEN_ID = 102
VISION_START_TOKEN_ID = 103
VISION_END_TOKEN_ID = 104
TIME_START_TOKEN_ID = 105
TIME_END_TOKEN_ID = 106
IM_END_TOKEN_ID = 107


class _ImageProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, images, return_tensors, **kwargs):
        self.calls.append({"return_tensors": return_tensors, **kwargs})
        values = []
        for image in images:
            marker = image.getpixel((0, 0))[0] + 1
            values.append(torch.full((1, 3), marker, dtype=torch.float32))

        return {
            "pixel_values": torch.cat(values),
            "image_grid_thw": torch.tensor([[1, 1, 1]] * len(images)),
        }


class _VideoProcessor:
    temporal_patch_size = 1

    def __init__(self):
        self.calls = []

    def __call__(self, videos, return_tensors, return_metadata, **kwargs):
        self.calls.append(
            {
                "return_tensors": return_tensors,
                "return_metadata": return_metadata,
                **kwargs,
            }
        )
        result = {
            "pixel_values_videos": torch.cat(
                [torch.full((2, 3), 9 + index, dtype=torch.float32) for index in range(len(videos))]
            ),
            "video_grid_thw": torch.tensor([[2, 1, 1]] * len(videos)),
        }
        if return_metadata:
            result["video_metadata"] = [
                SimpleNamespace(frames_indices=[0, 2], total_num_frames=2, fps=2.0, duration=2.0) for _ in videos
            ]

        return result


class _Tokenizer:
    pad_token_id = 0
    padding_side = "right"
    _token_ids = {
        "<|time_start|>": TIME_START_TOKEN_ID,
        "<|time_end|>": TIME_END_TOKEN_ID,
        "<|im_end|>": IM_END_TOKEN_ID,
    }

    def convert_tokens_to_ids(self, token):
        return self._token_ids[token]

    def pad(self, features, padding, max_length, pad_to_multiple_of, return_tensors):
        del padding, max_length, return_tensors
        sequence_length = max(len(feature["input_ids"]) for feature in features)
        if pad_to_multiple_of is not None:
            sequence_length = ((sequence_length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

        padded = {"input_ids": [], "attention_mask": []}
        for feature in features:
            pad_length = sequence_length - len(feature["input_ids"])
            if self.padding_side == "right":
                padded["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * pad_length)
                padded["attention_mask"].append(feature["attention_mask"] + [0] * pad_length)
            else:
                padded["input_ids"].append([self.pad_token_id] * pad_length + feature["input_ids"])
                padded["attention_mask"].append([0] * pad_length + feature["attention_mask"])

        return {key: torch.tensor(value) for key, value in padded.items()}


class _Processor:
    image_token_id = IMAGE_TOKEN_ID
    video_token_id = VIDEO_TOKEN_ID
    vision_start_token_id = VISION_START_TOKEN_ID
    vision_end_token_id = VISION_END_TOKEN_ID

    def __init__(self):
        self.image_processor = _ImageProcessor()
        self.video_processor = _VideoProcessor()
        self.tokenizer = _Tokenizer()

    @staticmethod
    def _calculate_timestamps(*args, **kwargs):
        del args, kwargs
        return [0.0, 1.0]


def _get_plugin():
    return get_mm_plugin(
        name="moss_vl",
        image_token="<|image_pad|>",
        video_token="<|video_pad|>",
        vision_bos_token="<|vision_start|>",
        vision_eos_token="<|vision_end|>",
        time_bos_token="<|time_start|>",
        time_eos_token="<|time_end|>",
    )


def _video_ids(seed):
    return [
        VISION_START_TOKEN_ID,
        TIME_START_TOKEN_ID,
        seed,
        TIME_END_TOKEN_ID,
        IMAGE_TOKEN_ID,
        TIME_START_TOKEN_ID,
        seed + 1,
        TIME_END_TOKEN_ID,
        IMAGE_TOKEN_ID,
        VISION_END_TOKEN_ID,
    ]


def _left_pad(sequences, pad_value):
    max_len = max(map(len, sequences))
    return torch.tensor([[pad_value] * (max_len - len(sequence)) + sequence for sequence in sequences])


def test_moss_vl_process_messages_expands_video_frames():
    plugin = _get_plugin()
    processor = _Processor()
    messages = [
        {"role": "user", "content": "First <image>, then <video>, finally <image>."},
        {"role": "assistant", "content": "Done."},
    ]
    images = [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2), (2, 0, 0))]

    processed = plugin.process_messages(messages, images, ["video.mp4"], [], processor)

    video_tokens = (
        "<|vision_start|>"
        "<|time_start|>0.0 seconds<|time_end|><|image_pad|>"
        "<|time_start|>1.0 seconds<|time_end|><|image_pad|>"
        "<|vision_end|>"
    )
    assert processed[0]["content"] == (f"First <|image_pad|>, then {video_tokens}, finally <|image_pad|>.")
    assert messages[0]["content"] == "First <image>, then <video>, finally <image>."


@pytest.mark.parametrize(
    ("content", "images", "videos", "error"),
    [
        ("Missing media: <image>.", [], [], "number of images does not match"),
        ("Missing media: <video>.", [], [], "number of videos does not match"),
    ],
)
def test_moss_vl_rejects_placeholder_count_mismatch(content, images, videos, error):
    plugin = _get_plugin()
    with pytest.raises(ValueError, match=error):
        plugin.process_messages(
            [{"role": "user", "content": content}],
            images,
            videos,
            [],
            _Processor(),
        )


def test_moss_vl_process_messages_expands_multiple_videos_in_order():
    plugin = _get_plugin()
    messages = [{"role": "user", "content": "Compare <video> with <video>."}]

    processed = plugin.process_messages(messages, [], ["first.mp4", "second.mp4"], [], _Processor())

    frame_tokens = (
        "<|vision_start|>"
        "<|time_start|>0.0 seconds<|time_end|><|image_pad|>"
        "<|time_start|>1.0 seconds<|time_end|><|image_pad|>"
        "<|vision_end|>"
    )
    assert processed[0]["content"] == f"Compare {frame_tokens} with {frame_tokens}."


def test_moss_vl_forwards_spatial_pixel_limits_to_native_processors():
    plugin = _get_plugin()
    processor = _Processor()
    processor.image_min_pixels = 1024
    processor.image_max_pixels = 262144
    processor.video_min_pixels = 256
    processor.video_max_pixels = 16384
    processor.video_fps = 1.0
    processor.video_maxlen = 8
    image = Image.new("RGB", (1024, 1024))

    plugin.process_messages(
        [{"role": "user", "content": "Compare <image> and <video>."}],
        [image],
        ["video.mp4"],
        [],
        processor,
    )
    plugin.get_mm_inputs(
        [image],
        ["video.mp4"],
        [],
        [1],
        [1],
        [0],
        [[IMAGE_TOKEN_ID, *_video_ids(201)]],
        processor,
    )

    assert processor.image_processor.calls == [
        {
            "return_tensors": "pt",
            "min_pixels": 1024,
            "max_pixels": 262144,
        }
    ]
    assert processor.video_processor.calls == [
        {
            "return_tensors": "pt",
            "return_metadata": True,
            "video_fps": 1.0,
            "max_frames": 8,
            "size": {"shortest_edge": 256, "longest_edge": 16384},
        },
        {
            "return_tensors": "pt",
            "return_metadata": False,
            "video_fps": 1.0,
            "max_frames": 8,
            "size": {"shortest_edge": 256, "longest_edge": 16384},
        },
    ]


def test_moss_vl_rejects_invalid_batch_metadata():
    plugin = _get_plugin()
    processor = _Processor()
    image = Image.new("RGB", (2, 2))

    with pytest.raises(ValueError, match="batch metadata must have one entry per sample"):
        plugin.get_mm_inputs([image], [], [], [1], [], [0], [[IMAGE_TOKEN_ID]], processor)

    with pytest.raises(ValueError, match="media lengths do not consume all provided inputs"):
        plugin.get_mm_inputs([image], [], [], [0], [0], [0], [[201]], processor)


def test_moss_vl_rejects_truncated_media_tokens():
    plugin = _get_plugin()
    with pytest.raises(ValueError, match="increase `cutoff_len`"):
        plugin.get_mm_inputs(
            [Image.new("RGB", (2, 2))],
            [],
            [],
            [1],
            [0],
            [0],
            [[201, 202]],
            _Processor(),
        )


def test_moss_vl_rejects_incomplete_video_token_block():
    plugin = _get_plugin()
    truncated_video_ids = _video_ids(201)[:-1]

    with pytest.raises(ValueError, match="incomplete video token block"):
        plugin.get_mm_inputs(
            [],
            ["video.mp4"],
            [],
            [0],
            [1],
            [0],
            [truncated_video_ids],
            _Processor(),
        )


def test_moss_vl_rejects_video_frame_token_count_mismatch():
    plugin = _get_plugin()
    incomplete_frame_ids = [VISION_START_TOKEN_ID, IMAGE_TOKEN_ID, VISION_END_TOKEN_ID]

    with pytest.raises(ValueError, match="video frame tokens do not match"):
        plugin.get_mm_inputs(
            [],
            ["video.mp4"],
            [],
            [0],
            [1],
            [0],
            [incomplete_frame_ids],
            _Processor(),
        )


def test_moss_vl_media_order_batch_mask_and_labels():
    plugin = _get_plugin()
    processor = _Processor()
    images = [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2), (2, 0, 0))]
    first_ids = [
        IMAGE_TOKEN_ID,
        201,
        VISION_START_TOKEN_ID,
        TIME_START_TOKEN_ID,
        202,
        TIME_END_TOKEN_ID,
        IMAGE_TOKEN_ID,
        TIME_START_TOKEN_ID,
        203,
        TIME_END_TOKEN_ID,
        IMAGE_TOKEN_ID,
        VISION_END_TOKEN_ID,
        IMAGE_TOKEN_ID,
        204,
    ]
    second_ids = [301, 302]
    assert plugin._get_media_order_from_ids(first_ids, processor, 2, 1) == ["image", "video", "image"]

    mm_inputs = plugin.get_mm_inputs(
        images=images,
        videos=["video.mp4"],
        audios=[],
        imglens=[2, 0],
        vidlens=[1, 0],
        audlens=[0, 0],
        batch_ids=[first_ids, second_ids],
        processor=processor,
    )

    assert mm_inputs["grid_thw"].tolist() == [[1, 1, 1], [2, 1, 1], [1, 1, 1], [1, 1, 1]]
    assert mm_inputs["media_nums_per_sample"] == [3, 1]
    assert mm_inputs["pixel_values"][:, 0].tolist() == [1.0, 9.0, 9.0, 3.0, 256.0]

    pre_padding_mask = mm_inputs["cross_attention_mask"]
    assert pre_padding_mask.shape == (2, 1, len(first_ids), 4)
    assert pre_padding_mask[0, 0, 0].tolist() == [False, True, True, True]
    assert pre_padding_mask[0, 0, 10].tolist() == [False, False, False, True]
    assert pre_padding_mask[0, 0, 12].tolist() == [False, False, False, False]
    assert pre_padding_mask[1].all()

    seq_len = len(first_ids)
    input_ids = torch.tensor([first_ids, [0] * (seq_len - 2) + second_ids])
    attention_mask = torch.tensor([[1] * seq_len, [0] * (seq_len - 2) + [1, 1]])
    labels = input_ids.clone()
    labels[attention_mask == 0] = IGNORE_INDEX
    features = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "position_ids": torch.arange(seq_len).repeat(2, 1),
    }

    mm_inputs = plugin.post_process_mossvl_inputs(features, mm_inputs, processor)

    mask = mm_inputs["cross_attention_mask"]
    assert mask.shape == (2, 1, seq_len, 4)
    assert mask[0, 0, 0].tolist() == [False, True, True, True]
    assert mask[0, 0, 10].tolist() == [False, False, False, True]
    assert mask[0, 0, 12].tolist() == [False, False, False, False]
    assert mask[1].all()
    assert "position_ids" not in features
    assert not torch.any((features["input_ids"] == IMAGE_TOKEN_ID) & ~features["attention_mask"].bool())
    assert features["labels"][0, 1].item() == 201
    assert features["labels"][0, 13].item() == 204
    assert features["labels"][0, 0].item() == IGNORE_INDEX
    assert features["labels"][0, 12].item() == IGNORE_INDEX
    assert torch.all(features["labels"][0, 2:12] == IGNORE_INDEX)


def test_moss_vl_complex_batch_keeps_media_and_masks_sample_local():
    plugin = _get_plugin()
    processor = _Processor()
    batch_ids = [
        [IMAGE_TOKEN_ID, 211, IMAGE_TOKEN_ID, 212],
        [221, *_video_ids(222), 223, *_video_ids(224), 225],
        [IMAGE_TOKEN_ID, 231, *_video_ids(232), 233, IMAGE_TOKEN_ID, 234],
        [241, 242, 243],
    ]
    images = [Image.new("RGB", (2, 2), (marker, 0, 0)) for marker in range(4)]
    mm_inputs = plugin.get_mm_inputs(
        images=images,
        videos=["first.mp4", "second.mp4", "third.mp4"],
        audios=[],
        imglens=[2, 0, 2, 0],
        vidlens=[0, 2, 1, 0],
        audlens=[0, 0, 0, 0],
        batch_ids=batch_ids,
        processor=processor,
    )

    assert mm_inputs["grid_thw"].tolist() == [
        [1, 1, 1],
        [1, 1, 1],
        [2, 1, 1],
        [2, 1, 1],
        [1, 1, 1],
        [2, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    assert mm_inputs["media_nums_per_sample"] == [2, 2, 3, 1]
    assert mm_inputs["pixel_values"][:, 0].tolist() == [
        1.0,
        2.0,
        9.0,
        9.0,
        10.0,
        10.0,
        3.0,
        9.0,
        9.0,
        4.0,
        256.0,
    ]

    input_ids = _left_pad(batch_ids, 0)
    attention_mask = _left_pad([[1] * len(ids) for ids in batch_ids], 0)
    labels = input_ids.clone()
    labels[attention_mask == 0] = IGNORE_INDEX
    features = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "position_ids": torch.arange(input_ids.shape[1]).repeat(len(batch_ids), 1),
    }

    plugin.post_process_mossvl_inputs(features, mm_inputs, processor)

    cross_mask = mm_inputs["cross_attention_mask"]
    assert cross_mask.shape == (4, 1, input_ids.shape[1], 4)
    assert (~cross_mask[0]).sum().item() > 0
    assert (~cross_mask[1]).sum().item() > 0
    assert (~cross_mask[2]).sum().item() > 0
    assert cross_mask[3].all()
    assert cross_mask[0, ..., 2:].all()
    assert not cross_mask[1, ..., :4].all()
    assert not cross_mask[2, ..., :4].all()
    assert features["labels"][3, -3:].tolist() == [241, 242, 243]
    assert torch.all(features["labels"][features["attention_mask"] == 0] == IGNORE_INDEX)
    assert "position_ids" not in features


def test_moss_vl_supervised_processor_to_collator_mixed_batch(monkeypatch):
    plugin = _get_plugin()
    processor = _Processor()
    tokenizer = processor.tokenizer
    template = SimpleNamespace(mm_plugin=plugin)
    dataset_processor = SupervisedDatasetProcessor(
        template=template,
        tokenizer=tokenizer,
        processor=processor,
        data_args=SimpleNamespace(),
    )
    first_ids = [IMAGE_TOKEN_ID, 211, *_video_ids(212), IMAGE_TOKEN_ID, 214]
    second_ids = [221, 222]

    def encode_example(prompt, **kwargs):
        del kwargs
        input_ids = first_ids if "<image>" in prompt[0]["content"] else second_ids
        return input_ids, input_ids.copy()

    monkeypatch.setattr(dataset_processor, "_encode_data_example", encode_example)
    examples = {
        "_prompt": [
            [{"role": "user", "content": "Compare <image>, <video>, and <image>."}],
            [{"role": "user", "content": "Text-only question."}],
        ],
        "_response": [
            [{"role": "assistant", "content": "Mixed answer."}],
            [{"role": "assistant", "content": "Text answer."}],
        ],
        "_system": ["", ""],
        "_tools": ["", ""],
        "_images": [
            [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2), (2, 0, 0))],
            None,
        ],
        "_videos": [["video.mp4"], None],
        "_audios": [None, None],
    }
    model_inputs = dataset_processor.preprocess_dataset(examples)
    assert "media_order" not in model_inputs
    collator = MultiModalDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=SimpleNamespace(config=SimpleNamespace(model_type="moss_vl")),
        template=template,
        processor=processor,
        label_pad_token_id=IGNORE_INDEX,
    )
    features = [
        {key: values[index] for key, values in model_inputs.items()} for index in range(len(model_inputs["input_ids"]))
    ]
    batch = collator(features)

    assert batch["grid_thw"].tolist() == [[1, 1, 1], [2, 1, 1], [1, 1, 1], [1, 1, 1]]
    assert batch["media_nums_per_sample"] == [3, 1]
    assert batch["pixel_values"][:, 0].tolist() == [1.0, 9.0, 9.0, 3.0, 256.0]
    assert batch["cross_attention_mask"].shape == (2, 1, len(first_ids), 4)
    assert batch["cross_attention_mask"][1].all()
    assert torch.all(batch["labels"][1, len(second_ids) :] == IGNORE_INDEX)
    assert "position_ids" not in batch


def test_moss_vl_generate_collator_keeps_left_padded_cross_attention_mask():
    plugin = _get_plugin()
    processor = _Processor()
    processor.tokenizer.padding_side = "left"
    template = SimpleNamespace(mm_plugin=plugin)
    batch_ids = [
        [IMAGE_TOKEN_ID, 211],
        [301, IMAGE_TOKEN_ID, 302, 303],
    ]
    features = [
        {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": input_ids.copy(),
            "images": [Image.new("RGB", (2, 2))],
        }
        for input_ids in batch_ids
    ]
    collator = MultiModalDataCollatorForSeq2Seq(
        tokenizer=processor.tokenizer,
        model=SimpleNamespace(config=SimpleNamespace(model_type="moss_vl")),
        template=template,
        processor=processor,
        label_pad_token_id=IGNORE_INDEX,
        pad_to_multiple_of=8,
    )

    batch = collator(features)

    assert batch["cross_attention_mask"].shape == (2, 1, 8, 1)
    assert batch["cross_attention_mask"][0, 0, :, 0].tolist() == [True] * 6 + [False, False]
    assert batch["cross_attention_mask"][1, 0, :, 0].tolist() == [True] * 5 + [False, False, False]


def test_moss_vl_predict_collator_uses_precomputed_cross_attention_mask_without_model():
    plugin = _get_plugin()
    processor = _Processor()
    template = SimpleNamespace(mm_plugin=plugin)
    batch_ids = [
        [IMAGE_TOKEN_ID, 211],
        [301, IMAGE_TOKEN_ID, 302, 303],
    ]
    features = [
        {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": input_ids.copy(),
            "images": [Image.new("RGB", (2, 2))],
        }
        for input_ids in batch_ids
    ]
    collator = MultiModalDataCollatorForSeq2Seq(
        tokenizer=processor.tokenizer,
        model=None,
        template=template,
        processor=processor,
        label_pad_token_id=IGNORE_INDEX,
    )

    batch = collator(features)

    assert batch["cross_attention_mask"].shape == (2, 1, 4, 1)
    assert batch["cross_attention_mask"][0, 0, :, 0].tolist() == [False, False, True, True]
    assert batch["cross_attention_mask"][1, 0, :, 0].tolist() == [True, False, False, False]


def test_moss_vl_masks_only_the_token_after_im_end():
    plugin = _get_plugin()
    processor = _Processor()
    input_ids = torch.tensor([[301, IM_END_TOKEN_ID, 302, 303]])
    features = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }
    mm_inputs = plugin.get_mm_inputs([], [], [], [0], [0], [0], [input_ids[0].tolist()], processor)

    plugin.post_process_mossvl_inputs(features, mm_inputs, processor)

    assert features["labels"].tolist() == [[301, IM_END_TOKEN_ID, IGNORE_INDEX, 303]]


def test_moss_vl_native_text_dummy_shape_and_values():
    plugin = _get_plugin()
    processor = _Processor()
    processor.image_processor = SimpleNamespace(patch_size=16, temporal_patch_size=1, merge_size=2)

    mm_inputs = plugin.get_mm_inputs([], [], [], [0], [0], [0], [[301]], processor)

    assert mm_inputs["grid_thw"].tolist() == [[1, 8, 8]]
    assert mm_inputs["pixel_values"].shape == (64, 768)
    assert torch.count_nonzero(mm_inputs["pixel_values"]).item() == 0
    assert mm_inputs["media_nums_per_sample"] == [1]
