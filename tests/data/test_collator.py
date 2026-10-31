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

import inspect
import os
from collections import Counter

import pytest
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq, prepare_4d_attention_mask
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.packages import is_transformers_version_greater_than
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer


TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
def test_base_collator():
    model_args, data_args, *_ = get_infer_args({"model_name_or_path": TINY_LLAMA3, "template": "default"})
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    data_collator = MultiModalDataCollatorForSeq2Seq(
        template=template,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX,
        **tokenizer_module,
    )
    p = tokenizer_module["tokenizer"].pad_token_id
    q = IGNORE_INDEX
    features = [
        {
            "input_ids": [0, 1, 2, 3, 4, 5],
            "attention_mask": [1, 1, 1, 1, 1, 1],
            "labels": [q, q, 2, 3, 4, 5],
        },
        {
            "input_ids": [6, 7],
            "attention_mask": [1, 1],
            "labels": [q, 7],
        },
    ]
    batch_input = data_collator(features)
    expected_input = {
        "input_ids": [
            [0, 1, 2, 3, 4, 5, p, p],
            [6, 7, p, p, p, p, p, p],
        ],
        "attention_mask": [
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
        ],
        "labels": [
            [q, q, 2, 3, 4, 5, q, q],
            [q, 7, q, q, q, q, q, q],
        ],
    }
    for k in batch_input.keys():
        assert batch_input[k].eq(torch.tensor(expected_input[k])).all()


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
def test_multimodal_collator():
    model_args, data_args, *_ = get_infer_args(
        {"model_name_or_path": "Qwen/Qwen2-VL-2B-Instruct", "template": "qwen2_vl"}
    )
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(config)

    data_collator = MultiModalDataCollatorForSeq2Seq(
        template=template,
        model=model,
        pad_to_multiple_of=4,
        label_pad_token_id=IGNORE_INDEX,
        **tokenizer_module,
    )
    p = tokenizer_module["tokenizer"].pad_token_id
    q = IGNORE_INDEX
    s = tokenizer_module["tokenizer"].convert_tokens_to_ids("<|vision_start|>")
    e = tokenizer_module["tokenizer"].convert_tokens_to_ids("<|vision_end|>")
    m = tokenizer_module["tokenizer"].convert_tokens_to_ids("<|image_pad|>")
    fake_image = Image.new("RGB", (64, 64), (255, 255, 255))

    features = [
        {
            "input_ids": [0, 1, 2, 3],
            "attention_mask": [1, 1, 1, 1],
            "labels": [0, 1, 2, 3],
        },
    ]
    batch_input = data_collator(features)
    expected_input = {
        "input_ids": [
            [0, 1, 2, 3, s, m, m, m, m, e, p, p],
        ],
        "attention_mask": [
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        "labels": [
            [0, 1, 2, 3, q, q, q, q, q, q, q, q],
        ],
        "position_ids": [[[0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0]]] * 3,
        "rope_deltas": [[0]],
        **tokenizer_module["processor"].image_processor(fake_image),
    }
    if not is_transformers_version_greater_than("5.0.0"):
        # adapt position_ids and rope_deltas for transformers < 5.0.0
        # https://github.com/huggingface/transformers/pull/43972
        expected_input["position_ids"] = [[[0, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1]]] * 3
        expected_input["rope_deltas"] = [[-8]]

    assert batch_input.keys() == expected_input.keys()
    for k in batch_input.keys():
        if k == "position_ids" and batch_input[k].dim() == 3 and batch_input[k].shape[0] == 4:
            batch_input[k] = batch_input[k][1:]

        assert batch_input[k].eq(torch.tensor(expected_input[k])).all()


def _make_packed_feature(
    *,
    packing_params: dict,
    pad_token_id: int,
    label_ignore_id: int,
    fake_image: Image.Image,
    vision_start_id: int | None = None,
    vision_end_id: int | None = None,
    image_pad_id: int | None = None,
) -> dict:
    r"""Build one packed sample using the new PackingParams schema."""
    sequence_boundaries = packing_params["sequence_boundaries"]
    image_subseq_ids = packing_params["image_subseq_ids"]
    video_subseq_ids = packing_params["video_subseq_ids"]
    audio_subseq_ids = packing_params["audio_subseq_ids"]
    unpadded_length = packing_params["unpadded_length"]
    right_padding_length = packing_params["right_padding_length"]  # which only preserved in tests
    cutoff_plus_one = sequence_boundaries[-1]
    content_len = unpadded_length
    pad_len = right_padding_length
    assert content_len + pad_len == cutoff_plus_one
    assert sequence_boundaries[0] == 0
    assert sequence_boundaries[-1] == cutoff_plus_one

    content_ids = list(range(100, 100 + content_len))
    if vision_start_id is not None and vision_end_id is not None and image_pad_id is not None:
        image_counts_by_subseq = Counter(image_subseq_ids)
        for subseq_idx, image_count in sorted(image_counts_by_subseq.items()):
            if subseq_idx >= len(sequence_boundaries) - 1:
                continue

            subseq_start = sequence_boundaries[subseq_idx]
            subseq_end = sequence_boundaries[subseq_idx + 1]
            subseq_len = subseq_end - subseq_start
            if subseq_len < 3:
                continue

            # Build repeated image groups while preserving at least 3 tokens for each remaining image.
            injected_tokens: list[int] = []
            remaining = subseq_len
            for image_idx in range(image_count):
                remaining_images = image_count - image_idx
                min_reserved_for_rest = 3 * (remaining_images - 1)
                current_group_len = min(6, remaining - min_reserved_for_rest)
                if current_group_len < 3:
                    break

                group = [vision_start_id] + [image_pad_id] * max(1, current_group_len - 2) + [vision_end_id]
                injected_tokens.extend(group[:current_group_len])
                remaining -= current_group_len

            if injected_tokens:
                insert_end = subseq_start + len(injected_tokens)
                content_ids[subseq_start:insert_end] = injected_tokens

    input_ids = content_ids + [pad_token_id] * pad_len
    attention_mask = [1] * content_len + [0] * pad_len
    labels = [label_ignore_id] * cutoff_plus_one

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "images": [fake_image] * len(image_subseq_ids),
        "videos": [None] * len(video_subseq_ids),
        "audios": [None] * len(audio_subseq_ids),
        "packing_params": packing_params,
    }


def _make_packed_features(
    *,
    packing_params: dict,
    pad_token_id: int,
    label_ignore_id: int,
    fake_image: Image.Image,
    vision_start_id: int,
    vision_end_id: int,
    image_pad_id: int,
) -> list[dict]:
    r"""Build packed features from caller-provided packing_params."""
    return [
        _make_packed_feature(
            packing_params=packing_params,
            pad_token_id=pad_token_id,
            label_ignore_id=label_ignore_id,
            fake_image=fake_image,
            vision_start_id=vision_start_id,
            vision_end_id=vision_end_id,
            image_pad_id=image_pad_id,
        )
    ]


def _get_expected_position_ids(
    packing_params,
    get_rope_func,
    input_ids,
    attention_mask,
    image_token_id: int | None = None,
    video_token_id: int | None = None,
) -> torch.Tensor:
    bound_list = packing_params["sequence_boundaries"]
    input_ids_slices = [input_ids[bound_list[i] : bound_list[i + 1]] for i in range(len(bound_list) - 1)]
    attention_mask_slices = [attention_mask[bound_list[i] : bound_list[i + 1]] for i in range(len(bound_list) - 1)]
    img_counts_by_subseq = Counter(packing_params["image_subseq_ids"])
    needs_mm_token_type_ids = "mm_token_type_ids" in inspect.signature(get_rope_func).parameters
    all_position_ids = []
    for i, input_ids_slice in enumerate(input_ids_slices):
        img_cnt = img_counts_by_subseq[i]
        if sum(attention_mask_slices[i]) == 0:
            continue

        input_ids_tensor = torch.tensor(input_ids_slice).unsqueeze(0)
        rope_func_kwargs = {
            "input_ids": input_ids_tensor,
            "attention_mask": torch.tensor(attention_mask_slices[i]).unsqueeze(0),
            "image_grid_thw": [torch.tensor([1, 4, 4])] * img_cnt,
        }
        if needs_mm_token_type_ids:
            mm_token_type_ids = torch.zeros_like(input_ids_tensor)
            if image_token_id is not None:
                mm_token_type_ids[input_ids_tensor == image_token_id] = 1
            if video_token_id is not None:
                mm_token_type_ids[input_ids_tensor == video_token_id] = 2
            rope_func_kwargs["mm_token_type_ids"] = mm_token_type_ids

        position_ids, _ = get_rope_func(**rope_func_kwargs)
        all_position_ids.append(position_ids)

    return torch.cat(all_position_ids, dim=-1)


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
def test_multimodal_collator_with_packing():
    model_args, data_args, *_ = get_infer_args(
        {"model_name_or_path": "Qwen/Qwen2-VL-2B-Instruct", "template": "qwen2_vl"}
    )
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    tokenizer_module["tokenizer"].padding_side = "right"
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(config)

    data_collator = MultiModalDataCollatorForSeq2Seq(
        template=template,
        model=model,
        pad_to_multiple_of=4,
        label_pad_token_id=IGNORE_INDEX,
        **tokenizer_module,
    )

    tokenizer = tokenizer_module["tokenizer"]
    packing_params = {
        "sequence_boundaries": [0, 2, 10, 18, 28, 32],
        "image_subseq_ids": [1, 2, 3],
        "video_subseq_ids": [],
        "audio_subseq_ids": [],
        "unpadded_length": 28,
        "right_padding_length": 4,
    }
    fake_image = Image.new("RGB", (64, 64), (255, 255, 255))
    features = _make_packed_features(
        packing_params=packing_params,
        pad_token_id=tokenizer.pad_token_id,
        label_ignore_id=IGNORE_INDEX,
        fake_image=fake_image,
        vision_start_id=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        vision_end_id=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        image_pad_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"),
    )
    expected_position_ids = _get_expected_position_ids(
        packing_params,
        data_collator.get_rope_func,
        features[0]["input_ids"],
        features[0]["attention_mask"],
        image_token_id=getattr(model.config, "image_token_id", None),
        video_token_id=getattr(model.config, "video_token_id", None),
    )
    batch_input = data_collator(features)  # [3, bsz, seq_len]
    valid_len = expected_position_ids.shape[-1]
    assert batch_input["position_ids"][1:, :, :valid_len].eq(expected_position_ids).all()


@pytest.mark.runs_on(["cpu", "xpu"])
def test_4d_attention_mask():
    o = 0.0
    x = torch.finfo(torch.float16).min
    attention_mask_with_indices = torch.tensor(
        [
            [1, 1, 2, 2, 2, 0],
            [1, 2, 2, 3, 3, 3],
        ]
    )
    attention_mask_computed = prepare_4d_attention_mask(attention_mask_with_indices, torch.float16)
    attention_mask_expected = torch.tensor(
        [
            [
                [
                    [o, x, x, x, x, x],
                    [o, o, x, x, x, x],
                    [x, x, o, x, x, x],
                    [x, x, o, o, x, x],
                    [x, x, o, o, o, x],
                    [x, x, x, x, x, x],
                ]
            ],
            [
                [
                    [o, x, x, x, x, x],
                    [x, o, x, x, x, x],
                    [x, o, o, x, x, x],
                    [x, x, x, o, x, x],
                    [x, x, x, o, o, x],
                    [x, x, x, o, o, o],
                ]
            ],
        ],
        dtype=torch.float16,
    )
    assert list(attention_mask_computed.size()) == [2, 1, 6, 6]
    assert torch.all(attention_mask_computed == attention_mask_expected)


if __name__ == "__main__":
    test_multimodal_collator()
