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
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq
from llamafactory.data.mm_plugin import LlavaPlugin, Qwen2VLPlugin
from llamafactory.data.processor.processor_utils import infer_seqlen
from llamafactory.data.processor.supervised import PackedSupervisedDatasetProcessor, SupervisedDatasetProcessor
from llamafactory.extras.constants import IGNORE_INDEX
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


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.parametrize("packing", ["none", "packed", "neat"])
def test_qwen2_vl_keeps_example_when_only_text_is_truncated(packing):
    processor = _build_processor(cutoff_len=8192)
    image = Image.new("RGB", (448, 448), (10, 200, 10))
    example = _single_image_example(image)
    full = processor.preprocess_dataset(example)
    full_ids = full["input_ids"][0]

    processor.data_args.cutoff_len = len(full_ids) - 1
    if packing != "none":
        processor.data_args.neat_packing = packing == "neat"
        processor = PackedSupervisedDatasetProcessor(
            template=processor.template,
            tokenizer=processor.tokenizer,
            processor=processor.processor,
            data_args=processor.data_args,
        )
    truncated = processor.preprocess_dataset(example)

    assert len(truncated["input_ids"]) == 1
    assert len(truncated["input_ids"][0]) == len(full_ids) - 1 + int(packing != "none")
    for token in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>"):
        token_id = processor.tokenizer.convert_tokens_to_ids(token)
        assert truncated["input_ids"][0].count(token_id) == full_ids.count(token_id)


@pytest.mark.runs_on(["cpu", "mps"])
def test_qwen2_vl_partial_image_is_dropped_before_rope():
    processor = _build_processor(cutoff_len=8192)
    image = Image.new("RGB", (448, 448), (10, 200, 10))
    example = _single_image_example(image)
    messages = processor.template.mm_plugin.process_messages(
        example["_prompt"][0] + example["_response"][0], [image], [], [], processor.processor
    )
    source, target = processor.template.encode_multiturn(processor.tokenizer, messages, None, None, False)[0]
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    image_positions = [index for index, token_id in enumerate(source) if token_id == image_token_id]
    cutoff = image_positions[len(image_positions) // 2] + len(target)
    source_len, target_len = infer_seqlen(len(source), len(target), cutoff)
    partial_ids = source[:source_len] + target[:target_len]
    assert 0 < partial_ids.count(image_token_id) < len(image_positions)

    config = AutoConfig.from_pretrained(TINY_QWEN2_VL)
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(config)

    collator = MultiModalDataCollatorForSeq2Seq(
        template=processor.template,
        tokenizer=processor.tokenizer,
        processor=processor.processor,
        model=model,
        pad_to_multiple_of=4,
        label_pad_token_id=IGNORE_INDEX,
    )
    partial_feature = {
        "input_ids": partial_ids,
        "labels": partial_ids,
        "attention_mask": [1] * len(partial_ids),
        "images": [image],
        "videos": None,
        "audios": None,
    }
    with pytest.raises(RuntimeError, match="shape mismatch"):
        collator([partial_feature])

    processor.data_args.cutoff_len = cutoff
    assert not processor.preprocess_dataset(example)["input_ids"]
    processor.data_args.cutoff_len = len(source) + len(target) - 1
    safe = processor.preprocess_dataset(example)
    batch = collator([{key: value[0] for key, value in safe.items()}])
    assert batch["input_ids"].shape[-1] == batch["position_ids"].shape[-1]


@pytest.mark.runs_on(["cpu", "mps"])
def test_qwen2_vl_video_truncation():
    processor = _build_processor(cutoff_len=8192)
    frames = [Image.new("RGB", (112, 112), (color, 0, 0)) for color in (10, 20, 30, 40)]
    example = {
        "_prompt": [[{"role": "user", "content": "<video>Describe this clip."}]],
        "_response": [[{"role": "assistant", "content": "Red frames."}]],
        "_system": [None],
        "_tools": [None],
        "_images": [None],
        "_videos": [[frames]],
        "_audios": [None],
    }
    full = processor.preprocess_dataset(example)
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    assert full["input_ids"][0].count(video_token_id) > 1

    processor.data_args.cutoff_len = len(full["input_ids"][0]) - 1
    safe = processor.preprocess_dataset(example)
    assert safe["input_ids"][0].count(video_token_id) == full["input_ids"][0].count(video_token_id)

    messages = processor.template.mm_plugin.process_messages(
        example["_prompt"][0] + example["_response"][0], [], [frames], [], processor.processor
    )
    source, target = processor.template.encode_multiturn(processor.tokenizer, messages, None, None, False)[0]
    video_positions = [index for index, token_id in enumerate(source) if token_id == video_token_id]
    cutoff = video_positions[len(video_positions) // 2] + len(target)
    source_len, target_len = infer_seqlen(len(source), len(target), cutoff)
    partial_ids = source[:source_len] + target[:target_len]
    assert 0 < partial_ids.count(video_token_id) < len(video_positions)
    processor.data_args.cutoff_len = cutoff
    assert not processor.preprocess_dataset(example)["input_ids"]


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.parametrize("packing", ["none", "packed", "neat"])
@pytest.mark.parametrize("dropped_first", [False, True])
@pytest.mark.parametrize("workers", [1, 2])
def test_dataset_map_handles_a_batch_with_only_dropped_examples(tmp_path, packing, dropped_first, workers):
    processor = _build_processor(cutoff_len=80)
    if packing != "none":
        processor.data_args.neat_packing = packing == "neat"
        processor = PackedSupervisedDatasetProcessor(
            template=processor.template,
            tokenizer=processor.tokenizer,
            processor=processor.processor,
            data_args=processor.data_args,
        )

    rows = []
    for size in [448, 56] if dropped_first else [56, 448]:
        path = tmp_path / f"image-{size}.png"
        Image.new("RGB", (size, size), (10, 200, 10)).save(path)
        rows.append(
            {
                "_prompt": [{"role": "user", "content": "<image>Describe this image."}],
                "_response": [{"role": "assistant", "content": "Green."}],
                "_system": None,
                "_tools": None,
                "_images": [str(path)],
                "_videos": None,
                "_audios": None,
            }
        )

    dataset = Dataset.from_list(rows)
    mapped = dataset.map(
        processor.preprocess_dataset,
        batched=True,
        batch_size=1,
        num_proc=workers,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    )
    assert len(mapped) == 1
    assert mapped[0]["images"] == [str(tmp_path / "image-56.png")]

    if workers == 1 and dropped_first:
        streamed = dataset.to_iterable_dataset().map(
            processor.preprocess_dataset, batched=True, batch_size=1, remove_columns=dataset.column_names
        )
        streamed_rows = list(streamed)
        assert len(streamed_rows) == 1
        assert streamed_rows[0]["images"] == [str(tmp_path / "image-56.png")]

    only_dropped = dataset.select([0 if dropped_first else 1])
    empty = only_dropped.map(
        processor.preprocess_dataset,
        batched=True,
        batch_size=1,
        remove_columns=only_dropped.column_names,
        load_from_cache_file=False,
    )
    assert len(empty) == 0


class _FakeTokenizer:
    eos_token_id = 2
    unk_token_id = 0
    token_ids = {"<vision>": 91, "</vision>": 92, "<image>": 93, "<video>": 94}
    all_special_ids = [91, 92, 93, 94]

    def convert_tokens_to_ids(self, token):
        return self.token_ids.get(token, self.unk_token_id)

    def convert_ids_to_tokens(self, token_id):
        return next((token for token, value in self.token_ids.items() if value == token_id), None)

    def encode(self, token, add_special_tokens=False):
        return [self.convert_tokens_to_ids(token)]


def _fake_processor(
    monkeypatch, pairs, cutoff_len, *, efficient_eos=False, mask_history=False, plugin=None, prefix=None
):
    if plugin is None:
        plugin = Qwen2VLPlugin(
            image_token="<image>",
            video_token="<video>",
            audio_token=None,
            vision_bos_token="<vision>",
            vision_eos_token="</vision>",
        )
    monkeypatch.setattr(plugin, "process_messages", lambda messages, *args: messages)
    monkeypatch.setattr(
        plugin,
        "process_token_ids",
        lambda input_ids, labels, *args: ((prefix or []) + input_ids, (prefix or []) + labels),
    )
    template = SimpleNamespace(
        mm_plugin=plugin,
        preserve_thinking=True,
        efficient_eos=efficient_eos,
        encode_multiturn=lambda *args: pairs,
    )
    return SupervisedDatasetProcessor(
        template=template,
        tokenizer=_FakeTokenizer(),
        processor=None,
        data_args=SimpleNamespace(cutoff_len=cutoff_len, mask_history=mask_history, train_on_prompt=False),
    )


def _encode_fake(processor, *, images=None, videos=None):
    return processor._encode_data_example(
        prompt=[], response=[], system=None, tools=None, images=images or [], videos=videos or [], audios=[]
    )


def test_qwen2_vl_efficient_eos_does_not_hide_truncated_media(monkeypatch):
    processor = _fake_processor(monkeypatch, [([91, 93, 93, 92], [7])], 5, efficient_eos=True)
    assert _encode_fake(processor, images=[object()]) is None


def test_qwen2_vl_prefix_and_eos_do_not_hide_truncated_media(monkeypatch):
    processor = _fake_processor(monkeypatch, [([91, 93, 93, 92], [7])], 6, efficient_eos=True, prefix=[96])
    assert _encode_fake(processor, images=[object()]) is None


def test_qwen2_vl_keeps_plain_text_truncation(monkeypatch):
    processor = _fake_processor(monkeypatch, [([91, 93, 93, 92, 8, 8], [7])], 6)
    encoded = _encode_fake(processor, images=[object()])
    assert encoded is not None
    assert encoded[0] == [91, 93, 93, 92, 8, 7]


def test_qwen2_vl_keeps_response_text_truncation(monkeypatch):
    processor = _fake_processor(monkeypatch, [([91, 93, 93, 92], [7] * 20)], 12)
    encoded = _encode_fake(processor, images=[object()])
    assert encoded is not None
    assert encoded[0] == [91, 93, 93, 92] + [7] * 8


@pytest.mark.parametrize("modality,token_id", [("image", 93), ("video", 94)])
@pytest.mark.parametrize("cutoff", [2, 3, 4])
def test_qwen2_vl_drops_removed_or_partial_media(monkeypatch, modality, token_id, cutoff):
    processor = _fake_processor(monkeypatch, [([91, token_id, token_id, 92], [7])], cutoff)
    kwargs = {"images" if modality == "image" else "videos": [object()]}
    assert _encode_fake(processor, **kwargs) is None


def test_qwen2_vl_drops_media_removed_from_masked_history(monkeypatch):
    pairs = [([91, 93, 93, 92], [7]), ([10, 11], [7])]
    processor = _fake_processor(monkeypatch, pairs, 3, mask_history=True)
    assert _encode_fake(processor, images=[object()]) is None


def test_qwen2_vl_drops_partial_video_even_with_an_intact_image(monkeypatch):
    pairs = [([91, 93, 93, 92, 10, 91, 94, 94, 92, 8], [7])]
    processor = _fake_processor(monkeypatch, pairs, 8)
    assert _encode_fake(processor, images=[object()], videos=[object()]) is None


@pytest.mark.parametrize("efficient_eos", [False, True])
@pytest.mark.parametrize("prefix", [[], [96, 96]])
def test_qwen2_vl_all_cutoff_boundaries_preserve_complete_media(monkeypatch, efficient_eos, prefix):
    source = [5, 91, 93, 93, 92, 6, 7, 91, 94, 94, 92, 8, 9]
    target = [20, 21, 22]
    processor = _fake_processor(monkeypatch, [(source, target)], 1, efficient_eos=efficient_eos, prefix=prefix)
    media_token_ids = {91, 92, 93, 94}
    reserved = len(prefix) + int(efficient_eos)
    for cutoff in range(1, reserved + len(source) + len(target) + 1):
        processor.data_args.cutoff_len = cutoff
        available = cutoff - reserved
        if available <= 0:
            source_len, target_len = 0, 0
        else:
            source_len, target_len = infer_seqlen(len(source), len(target), available)

        removed_media = any(token in media_token_ids for token in source[source_len:] + target[target_len:])
        encoded = _encode_fake(processor, images=[object()], videos=[object()])
        assert (encoded is None) == removed_media, (cutoff, efficient_eos, prefix)
        if encoded is not None:
            for token in media_token_ids:
                assert encoded[0].count(token) == source.count(token) + target.count(token)


def test_qwen2_vl_unknown_media_token_drops_truncated_sample(monkeypatch):
    processor = _fake_processor(monkeypatch, [([91, 93, 93, 92, 8], [7])], 5)
    processor.tokenizer.token_ids = {"<vision>": 91, "</vision>": 92}
    assert _encode_fake(processor, images=[object()]) is None


def test_qwen2_vl_unknown_media_token_keeps_untruncated_sample(monkeypatch):
    processor = _fake_processor(monkeypatch, [([91, 93, 93, 92, 8], [7])], 6)
    processor.tokenizer.token_ids = {"<vision>": 91, "</vision>": 92}
    assert _encode_fake(processor, images=[object()]) is not None


@pytest.mark.parametrize("failure", ["missing", "not_special", "roundtrip", "non_atomic", "duplicate"])
def test_qwen2_vl_unverifiable_media_mapping_drops_truncated_sample(monkeypatch, failure):
    processor = _fake_processor(monkeypatch, [([91, 93, 93, 92, 8], [7])], 5)
    if failure == "missing":
        processor.template.mm_plugin.image_token = None
    elif failure == "not_special":
        processor.tokenizer.all_special_ids = [91, 92]
    elif failure == "roundtrip":
        monkeypatch.setattr(processor.tokenizer, "convert_ids_to_tokens", lambda token_id: "different")
    elif failure == "non_atomic":
        monkeypatch.setattr(processor.tokenizer, "encode", lambda token, add_special_tokens=False: [91, 92])
    else:
        processor.template.mm_plugin.vision_eos_token = processor.template.mm_plugin.vision_bos_token

    assert _encode_fake(processor, images=[object()]) is None


@pytest.mark.parametrize("modality", ["image", "video"])
@pytest.mark.parametrize("missing", ["boundary", "media"])
def test_qwen2_vl_missing_expected_media_markers_drops_truncated_sample(monkeypatch, modality, missing):
    token_id = 93 if modality == "image" else 94
    source = [91, token_id, 8, 8] if missing == "boundary" else [91, 92, 8, 8]
    processor = _fake_processor(monkeypatch, [(source, [7])], 4)
    kwargs = {"images" if modality == "image" else "videos": [object()]}
    assert _encode_fake(processor, **kwargs) is None


def test_other_multimodal_plugins_keep_original_truncation_behavior(monkeypatch):
    plugin = LlavaPlugin(image_token="<image>", video_token=None, audio_token=None)
    processor = _fake_processor(monkeypatch, [([93, 93, 8, 8], [7])], 4, plugin=plugin)
    encoded = _encode_fake(processor, images=[object()])
    assert encoded is not None
    assert encoded[0] == [93, 93, 8, 7]


def test_text_only_truncation_is_unchanged(monkeypatch):
    processor = _fake_processor(monkeypatch, [([10, 11, 12, 13], [7])], 4)
    encoded = _encode_fake(processor)
    assert encoded is not None
    assert encoded[0] == [10, 11, 12, 7]


@pytest.mark.runs_on(["cpu", "mps"])
def test_qwen2_vl_text_truncation_runs_tiny_model_forward_and_backward():
    processor = _build_processor(cutoff_len=8192)
    image = Image.new("RGB", (112, 112), (10, 200, 10))
    example = _single_image_example(image)
    full = processor.preprocess_dataset(example)
    processor.data_args.cutoff_len = len(full["input_ids"][0]) - 1
    safe = processor.preprocess_dataset(example)
    assert len(safe["input_ids"]) == 1

    config = AutoConfig.from_pretrained(TINY_QWEN2_VL)
    text_config = config.text_config
    text_config.hidden_size = 64
    text_config.intermediate_size = 128
    text_config.num_hidden_layers = 1
    text_config.num_attention_heads = 4
    text_config.num_key_value_heads = 2
    text_config.layer_types = ["full_attention"]
    text_config.max_window_layers = 1
    rope_parameters = getattr(text_config, "rope_parameters", None) or text_config.rope_scaling
    rope_parameters["mrope_section"] = [2, 3, 3]
    vision_config = config.vision_config
    vision_config.depth = 1
    vision_config.embed_dim = 64
    vision_config.hidden_size = 64
    vision_config.num_heads = 4
    vision_config.mlp_ratio = 2
    config.dtype = "float32"
    text_config.dtype = "float32"
    vision_config.dtype = "float32"
    model = AutoModelForImageTextToText.from_config(config)
    model.train()
    collator = MultiModalDataCollatorForSeq2Seq(
        template=processor.template,
        tokenizer=processor.tokenizer,
        processor=processor.processor,
        model=model,
        pad_to_multiple_of=4,
        label_pad_token_id=IGNORE_INDEX,
    )
    batch = collator([{key: value[0] for key, value in safe.items()}])
    result = model(**batch)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert model.get_input_embeddings().weight.grad is not None
