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

from llamafactory.v1.config import DataArguments, ModelArguments, TrainingArguments
from llamafactory.v1.core.data_engine import DataEngine
from llamafactory.v1.core.model_engine import ModelEngine
from llamafactory.v1.core.utils.batching import BatchGenerator
from llamafactory.v1.plugins.trainer_plugins.batching import (
    BatchingPlugin,
    _get_dynamic_micro_batch_sizes,
    _get_dynamic_padding_free_micro_batch_sizes,
)
from llamafactory.v1.utils.constants import IGNORE_INDEX
from llamafactory.v1.utils.objects import StatefulBuffer


def _make_model_input(length: int, start: int = 0):
    input_ids = list(range(start, start + length))
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * length,
        "labels": input_ids.copy(),
        "loss_weights": [1.0] * length,
        "position_ids": list(range(1, length + 1)),
    }


class _RestartableDataProvider:
    def __init__(self, batches):
        self.batches = batches
        self.num_iters = 0

    def __iter__(self):
        self.num_iters += 1
        return iter(self.batches)


def test_padding_free():
    buffer = StatefulBuffer()
    # Input samples:
    #   sample 0 input_ids: [0, 1]
    #   sample 1 input_ids: [10, 11, 12, 13]
    buffer.put([_make_model_input(2, 0), _make_model_input(4, 10)])
    batch_info = {"micro_batch_size": 2, "num_micro_batch": 1, "cutoff_len": 3}

    batch = BatchingPlugin("padding_free").generate_batch(buffer, batch_info)

    # Output batch:
    #   sample 1 is truncated to [10, 11, 12]
    #   both samples are packed into one sequence: [[0, 1, 10, 11, 12]]
    assert batch is not None
    assert len(batch) == 1
    assert batch[0]["input_ids"].shape == (1, 5)
    assert batch[0]["input_ids"].tolist() == [[0, 1, 10, 11, 12]]
    assert batch[0]["attention_mask"] is None
    assert batch[0]["position_ids"].tolist() == [[0, 1, 0, 1, 2]]
    assert batch[0]["labels"].tolist() == [[0, 1, IGNORE_INDEX, 11, 12]]
    assert batch[0]["loss_weights"].tolist() == [[1.0, 1.0, 0.0, 1.0, 1.0]]
    assert len(buffer) == 0


def test_batching_plugin_data_provider_batch_sizes():
    batch_info = {
        "micro_batch_size": 2,
        "num_micro_batch": 3,
        "cutoff_len": 10,
    }

    assert BatchingPlugin("padding_free").get_data_provider_batch_size(batch_info) == 6
    assert BatchingPlugin("dynamic_batching").get_data_provider_batch_size(batch_info) == 1
    assert BatchingPlugin("dynamic_padding_free").get_data_provider_batch_size(batch_info) == 1


def test_dynamic_batching():
    # Input samples:
    #   sample lengths: [3, 4, 6, 2, 8, 9]
    #   input_ids:
    #     [0, 1, 2]
    #     [10, 11, 12, 13]
    #     [20, 21, 22, 23, 24, 25]
    #     [30, 31]
    #     [40, 41, 42, 43, 44, 45, 46, 47]
    #     [50, 51, 52, 53, 54, 55, 56, 57, 58]
    samples = [
        _make_model_input(3, 0),
        _make_model_input(4, 10),
        _make_model_input(6, 20),
        _make_model_input(2, 30),
        _make_model_input(8, 40),
        _make_model_input(9, 50),
    ]
    batch_info = {"micro_batch_size": 2, "num_micro_batch": 1, "cutoff_len": 10}

    # Dynamic batching output plan:
    #   dynamic batching reads one sample at a time and uses cutoff_len * micro_batch_size
    #   as the padded-token budget for one training micro batch.
    #   [3, 4, 6] fits within budget 20 as shape [3, 6]; adding [2] would exceed it.
    assert _get_dynamic_micro_batch_sizes(samples, batch_info) == [3]

    buffer = StatefulBuffer()
    buffer.put(samples)
    batch = BatchingPlugin("dynamic_batching").generate_batch(buffer, batch_info)

    assert batch is not None
    assert len(batch) == 1
    assert batch[0]["input_ids"].shape == (3, 6)
    assert batch[0]["input_ids"].tolist()[0] == [0, 1, 2, 0, 0, 0]
    assert batch[0]["position_ids"].shape == (3, 6)
    assert batch[0]["position_ids"].tolist()[0] == [1, 2, 3, 0, 0, 0]
    assert len(buffer) == 3


def test_dynamic_batching_returns_none_when_token_budget_is_incomplete():
    buffer = StatefulBuffer()
    # Input buffer:
    #   only one sample with length [6].
    #   cutoff_len * micro_batch_size gives a padded-token budget of 20.
    #   this buffer has not filled the budget and has no next sample to prove overflow,
    #   so dynamic batching cannot produce a batch yet.
    buffer.put([_make_model_input(6, 0)])
    batch_info = {"micro_batch_size": 2, "num_micro_batch": 1, "cutoff_len": 10}

    assert _get_dynamic_micro_batch_sizes(buffer.samples, batch_info) == []
    assert BatchingPlugin("dynamic_batching").generate_batch(buffer, batch_info) is None
    # Batch generation does not read from the data iterator. It only returns None and keeps
    # existing samples in the buffer; BatchGenerator._fill_buffer handles refilling.
    assert len(buffer) == 1


def test_dynamic_batching_fill_buffer_restarts_until_micro_batch_is_complete():
    # Input data provider:
    #   each iterator pass yields one sample with length [6].
    #   each yielded item is a list[ModelInput], matching BatchGenerator._next_samples.
    #   _fill_buffer keeps restarting the iterator until the next appended sample
    #   proves that the previous dynamic micro batch has reached its budget boundary.
    samples = [_make_model_input(6, 0)]
    data_provider = _RestartableDataProvider([[sample] for sample in samples])

    batch_generator = BatchGenerator.__new__(BatchGenerator)
    batch_generator.batching_strategy = "dynamic_batching"
    batch_generator.micro_batch_size = 2
    batch_generator.num_micro_batch = 1
    batch_generator._buffer = StatefulBuffer()
    batch_generator._data_provider = data_provider
    batch_generator._data_iter = iter(data_provider)
    batch_generator._batch_info = {
        "micro_batch_size": 2,
        "num_micro_batch": 1,
        "cutoff_len": 10,
    }

    batch_generator._fill_buffer()

    # Filled buffer after restart:
    #   existing buffer [6, 6, 6] is kept; the fourth [6] remains for the next batch
    #   because adding it to the first dynamic micro batch would exceed the budget.
    assert data_provider.num_iters == 4
    assert _get_dynamic_micro_batch_sizes(batch_generator._buffer.samples, batch_generator._batch_info) == [3]

    batch = batch_generator._generate_batch()

    # Output batch:
    #   dynamic batching returns [micro_batch_0]
    #   micro_batch_0 consumes [6, 6, 6] => 3 samples, padded to shape [3, 6].
    assert batch is not None
    assert len(batch) == 1
    assert batch[0]["input_ids"].shape == (3, 6)
    assert len(batch_generator._buffer) == 1


def test_normal_batching():
    data_args = DataArguments(train_dataset="llamafactory/v1-sft-demo")
    data_engine = DataEngine(data_args.train_dataset)
    model_args = ModelArguments(model="llamafactory/tiny-random-qwen3")
    model_engine = ModelEngine(model_args=model_args)
    training_args = TrainingArguments(
        micro_batch_size=4,
        global_batch_size=8,
        cutoff_len=10,
        batching_workers=0,
        batching_strategy="normal",
    )
    batch_generator = BatchGenerator(
        data_engine,
        model_engine.renderer,
        micro_batch_size=training_args.micro_batch_size,
        global_batch_size=training_args.global_batch_size,
        cutoff_len=training_args.cutoff_len,
        batching_workers=training_args.batching_workers,
        batching_strategy=training_args.batching_strategy,
    )
    assert len(batch_generator) == len(data_engine) // training_args.global_batch_size
    batch = next(iter(batch_generator))
    assert len(batch) == 2
    assert batch[0]["input_ids"].shape == (4, 10)
    assert batch[0]["position_ids"].shape == (4, 10)


def test_dynamic_padding_free():
    """Test core logic of dynamic padding free strategy: pack samples by total token budget without padding."""
    # Construct test samples (lengths: 3, 4, 6, 2, 8, 9)
    # input_ids breakdown:
    #   sample 0: [0,1,2] (length=3)
    #   sample 1: [10,11,12,13] (length=4)
    #   sample 2: [20,21,22,23,24,25] (length=6)
    #   sample 3: [30,31] (length=2)
    #   sample 4: [40-47] (length=8)
    #   sample 5: [50-58] (length=9)
    samples = [
        _make_model_input(3, 0),
        _make_model_input(4, 10),
        _make_model_input(6, 20),
        _make_model_input(2, 30),
        _make_model_input(8, 40),
        _make_model_input(9, 50),
    ]
    # Batch config: micro_batch_size=2 → token budget = cutoff_len * micro_batch_size = 10*2=20
    batch_info = {"micro_batch_size": 2, "num_micro_batch": 1, "cutoff_len": 10}

    # Budget=20: 3+4+6+2=15 ≤20 (adding 8 would exceed) → first 4 samples are selected
    assert _get_dynamic_padding_free_micro_batch_sizes(samples, batch_info) == [4]

    buffer = StatefulBuffer()
    buffer.put(samples)
    batch = BatchingPlugin("dynamic_padding_free").generate_batch(buffer, batch_info)

    assert batch is not None
    assert len(batch) == 1  # num_micro_batch=1
    packed_batch = batch[0]

    # Total packed length: 3+4+6+2=15 → input_ids shape = (1,15) (no padding)
    assert packed_batch["input_ids"].shape == (1, 15)

    # Verify input_ids concatenation (first label of non-initial samples set to IGNORE_INDEX)
    assert packed_batch["input_ids"].tolist() == [
        [
            0,
            1,
            2,  # Sample 0
            10,
            11,
            12,
            13,  # Sample 1
            20,
            21,
            22,
            23,
            24,
            25,  # Sample 2
            30,
            31,
        ]  # Sample 3
    ]

    # Verify labels (first token of non-initial samples is IGNORE_INDEX)
    assert packed_batch["labels"].tolist() == [
        [
            0,
            1,
            2,  # Sample 0
            IGNORE_INDEX,
            11,
            12,
            13,  # Sample 1
            IGNORE_INDEX,
            21,
            22,
            23,
            24,
            25,  # Sample 2
            IGNORE_INDEX,
            31,
        ]  # Sample 3
    ]

    # Verify attention_mask: padding-free relies on reset-style position_ids instead of a dense mask.
    assert packed_batch["attention_mask"] is None

    # Verify position_ids
    assert packed_batch["position_ids"].tolist() == [
        [
            0,
            1,
            2,  # Sample 0
            0,
            1,
            2,
            3,  # Sample 1
            0,
            1,
            2,
            3,
            4,
            5,  # Sample 2
            0,
            1,
        ]  # Sample 3
    ]

    # Verify remaining samples in buffer: 6-4=2 samples (length 8,9)
    assert len(buffer) == 2


def test_dynamic_padding_free_returns_none_when_token_budget_is_incomplete():
    buffer = StatefulBuffer()
    buffer.put([_make_model_input(6, 0)])
    batch_info = {"micro_batch_size": 2, "num_micro_batch": 1, "cutoff_len": 10}

    assert _get_dynamic_micro_batch_sizes(buffer.samples, batch_info) == []
    assert BatchingPlugin("dynamic_padding_free").generate_batch(buffer, batch_info) is None
    # Batch generation does not read from the data iterator. It only returns None and keeps
    # existing samples in the buffer; BatchGenerator._fill_buffer handles refilling.
    assert len(buffer) == 1


def test_dynamic_padding_free_fill_buffer_restarts_until_micro_batch_is_complete():
    """Test fill_buffer logic for dynamic_padding_free: restart data iterator until token budget is full.

    Data provider yields one sample of length 6 per iteration.
    _fill_buffer keeps restarting iterator until next sample exceeds budget.
    Budget = 2 * 10 = 20 tokens.
    3 samples (6*3=18) fit; 4th sample (24) exceeds budget.
    So buffer will have 4 samples after fill_buffer.
    """
    samples = [_make_model_input(6, 0)]
    data_provider = _RestartableDataProvider([[sample] for sample in samples])

    batch_generator = BatchGenerator.__new__(BatchGenerator)
    batch_generator.batching_strategy = "dynamic_padding_free"
    batch_generator.micro_batch_size = 2
    batch_generator.num_micro_batch = 1
    batch_generator._buffer = StatefulBuffer()
    batch_generator._data_provider = data_provider
    batch_generator._data_iter = iter(data_provider)
    batch_generator._batch_info = {
        "micro_batch_size": 2,
        "num_micro_batch": 1,
        "cutoff_len": 10,
    }

    # Execute fill buffer (will restart iterator multiple times to collect enough samples)
    batch_generator._fill_buffer()

    # Buffer after restarts:
    #   3 samples can fit (18 tokens)
    #   4th sample is kept in buffer for next batch
    #   => num_iters = 4
    assert data_provider.num_iters == 4
    assert _get_dynamic_padding_free_micro_batch_sizes(
        batch_generator._buffer.samples, batch_generator._batch_info
    ) == [3]

    batch = batch_generator._generate_batch()

    # Output batch:
    #   dynamic_padding_free returns [micro_batch_0]
    #   3 samples packed into shape [1, 18]
    assert batch is not None
    assert len(batch) == 1
    assert batch[0]["input_ids"].shape == (1, 18)
    assert len(batch_generator._buffer) == 1


def _image_fragment(n_pad: int = 4, merge_sq: int = 4):
    """Hand-crafted image fragment: vision_start + n_pad image_pad + vision_end."""
    import torch

    pad, vstart, vend = 9, 8, 7
    return {
        "input_ids": [vstart] + [pad] * n_pad + [vend],
        "mm_token_type_ids": [0] + [1] * n_pad + [0],
        "pixel_values": torch.zeros((n_pad * merge_sq, 16), dtype=torch.float32),
        "image_grid_thw": torch.tensor([[1, 2, n_pad * 2]], dtype=torch.long),
    }


def _text_sample(n: int, base: int = 100):
    s = _make_model_input(n, start=base)
    s["position_ids"] = list(range(1, n + 1))
    return s


def test_inject_appends_zero_loss_dummy_into_collated_text_batch():
    import torch

    from llamafactory.v1.core.utils.batching import _collate_micro_batch, _inject_dummy_into_collated

    collated = _collate_micro_batch([_text_sample(20), _text_sample(8)], cutoff_len=4096)
    assert "pixel_values" not in collated
    bsz, seqlen = collated["input_ids"].shape

    frag = _image_fragment(n_pad=4)
    fl = len(frag["input_ids"])
    _inject_dummy_into_collated(collated, frag, marker=1)

    new_len = seqlen + fl
    # every sequence field grew by the fragment length, batch size unchanged
    for key in ("input_ids", "attention_mask", "labels", "loss_weights", "position_ids", "mm_token_type_ids"):
        assert collated[key].shape == (bsz, new_len)

    # dummy lives only in row 0's tail; other rows are padding (attention 0) there
    assert collated["input_ids"][0, seqlen:].tolist() == frag["input_ids"]
    assert collated["attention_mask"][0, seqlen:].tolist() == [1] * fl
    assert collated["attention_mask"][1, seqlen:].tolist() == [0] * fl
    # zero loss contribution
    assert collated["labels"][0, seqlen:].tolist() == [IGNORE_INDEX] * fl
    assert torch.all(collated["loss_weights"][:, seqlen:] == 0.0)
    assert collated["mm_token_type_ids"][0, seqlen:].tolist() == frag["mm_token_type_ids"]
    # pixel features carried verbatim
    assert torch.equal(collated["pixel_values"], frag["pixel_values"])
    assert torch.equal(collated["image_grid_thw"], frag["image_grid_thw"])


def test_inject_video_concatenates_alongside_existing_image():
    """Injecting a missing modality leaves the other modality's features intact (dim-0 cat)."""
    import torch

    from llamafactory.v1.core.utils.batching import _collate_micro_batch, _inject_dummy_into_collated

    img = _text_sample(10)
    img["pixel_values"] = torch.ones((8, 16), dtype=torch.float32)
    img["image_grid_thw"] = torch.tensor([[1, 2, 4]], dtype=torch.long)
    img["mm_token_type_ids"] = [0] * 10
    collated = _collate_micro_batch([img], cutoff_len=4096)

    video_frag = {
        "input_ids": [8, 6, 6, 7],
        "mm_token_type_ids": [0, 2, 2, 0],
        "pixel_values_videos": torch.zeros((8, 16), dtype=torch.float32),
        "video_grid_thw": torch.tensor([[1, 2, 4]], dtype=torch.long),
    }
    _inject_dummy_into_collated(collated, video_frag, marker=2)

    # image features untouched, video features added
    assert torch.equal(collated["pixel_values"], torch.ones((8, 16)))
    assert collated["pixel_values_videos"].shape[0] == 8
    assert collated["video_grid_thw"].shape[0] == 1
    assert collated["mm_token_type_ids"][0, -4:].tolist() == [0, 2, 2, 0]


def test_collate_creates_mm_token_type_ids_for_pure_text_then_inject():
    """A pure-text micro batch has no mm_token_type_ids; injection must create it."""
    from llamafactory.v1.core.utils.batching import _collate_micro_batch, _inject_dummy_into_collated

    collated = _collate_micro_batch([_text_sample(12)], cutoff_len=4096)
    assert "mm_token_type_ids" not in collated
    seqlen = collated["input_ids"].shape[1]

    frag = _image_fragment(n_pad=3)
    _inject_dummy_into_collated(collated, frag, marker=1)

    assert "mm_token_type_ids" in collated
    assert collated["mm_token_type_ids"].shape == collated["input_ids"].shape
    # original region all zero (text), dummy region carries the markers
    assert collated["mm_token_type_ids"][0, :seqlen].tolist() == [0] * seqlen
    assert collated["mm_token_type_ids"][0, seqlen:].tolist() == frag["mm_token_type_ids"]


def _audio_fragment(n_tok: int = 2, n_frames: int = 3000):
    """Hand-crafted audio fragment: audio_bos + n_tok AUDIO + audio_eos, with feature rows."""
    import torch

    aud, bos, eos = 50, 51, 52
    return {
        "input_ids": [bos] + [aud] * n_tok + [eos],
        "mm_token_type_ids": [0] + [3] * n_tok + [0],
        "input_features": torch.zeros((1, 128, n_frames), dtype=torch.float32),
        "feature_attention_mask": torch.ones((1, n_frames), dtype=torch.long),
    }


def test_inject_audio_dummy_into_text_batch():
    """A pure-text micro batch gets an audio dummy appended so the audio tower fires on every rank."""
    import torch

    from llamafactory.v1.core.utils.batching import _collate_micro_batch, _inject_dummy_into_collated

    collated = _collate_micro_batch([_text_sample(12)], cutoff_len=4096)
    assert "input_features" not in collated
    seqlen = collated["input_ids"].shape[1]

    frag = _audio_fragment(n_tok=2)
    fl = len(frag["input_ids"])
    _inject_dummy_into_collated(collated, frag, marker=3)

    # audio feature tensors carried verbatim; placeholder tokens marked 3 in the dummy tail
    assert torch.equal(collated["input_features"], frag["input_features"])
    assert torch.equal(collated["feature_attention_mask"], frag["feature_attention_mask"])
    assert collated["mm_token_type_ids"][0, seqlen:].tolist() == frag["mm_token_type_ids"]
    # zero loss contribution from the dummy
    assert collated["labels"][0, seqlen:].tolist() == [IGNORE_INDEX] * fl
    assert torch.all(collated["loss_weights"][:, seqlen:] == 0.0)


def test_audio_truncation_drops_orphaned_item_and_zeros_tokens():
    """Truncating mid-audio trims the orphaned feature row and zeros its in-window tokens."""
    import torch

    from llamafactory.v1.core.utils.collation import _align_multimodal_on_truncation

    aud = 50
    # text(2) + [audio#0: 4 tok] + text(1) + [audio#1: 4 tok] + text(1)
    input_ids = [1, 2] + [aud] * 4 + [3] + [aud] * 4 + [4]
    mm = [0, 0] + [3] * 4 + [0] + [3] * 4 + [0]
    sample = {
        "input_ids": input_ids,
        "labels": input_ids.copy(),
        "loss_weights": [1.0] * len(input_ids),
        "mm_token_type_ids": mm,
        "input_features": torch.zeros((2, 128, 10), dtype=torch.float32),
        "feature_attention_mask": torch.ones((2, 10), dtype=torch.long),
    }
    # audio#1 occupies positions 7..10; cut at 9 so its last token (10) is orphaned, audio#0 intact
    out = _align_multimodal_on_truncation(dict(sample), max_length=9)

    assert out["input_features"].shape[0] == 1  # only the complete audio#0 survives
    assert out["feature_attention_mask"].shape[0] == 1
    # audio#0 tokens (positions 2..5) untouched
    assert all(out["input_ids"][i] == aud and out["mm_token_type_ids"][i] == 3 for i in range(2, 6))
    # audio#1's in-window tokens (positions 7,8) zeroed + delabeled (positions >= 9 cut by truncation)
    for i in (7, 8):
        assert out["input_ids"][i] == 0
        assert out["mm_token_type_ids"][i] == 0
        assert out["labels"][i] == IGNORE_INDEX
        assert out["loss_weights"][i] == 0.0


def test_audio_truncation_keeps_all_when_complete():
    """No trimming when the cut falls after every audio's last token."""
    import torch

    from llamafactory.v1.core.utils.collation import _align_multimodal_on_truncation

    aud = 50
    input_ids = [1] + [aud] * 4 + [2]
    sample = {
        "input_ids": input_ids,
        "labels": input_ids.copy(),
        "loss_weights": [1.0] * len(input_ids),
        "mm_token_type_ids": [0] + [3] * 4 + [0],
        "input_features": torch.zeros((1, 128, 10), dtype=torch.float32),
        "feature_attention_mask": torch.ones((1, 10), dtype=torch.long),
    }
    out = _align_multimodal_on_truncation(dict(sample), max_length=6)
    assert out["input_features"].shape[0] == 1
    assert out["input_ids"] == input_ids


def test_drop_unsupervised_samples():
    """Samples whose supervised tokens fall entirely beyond cutoff_len are dropped (warn once)."""
    from types import SimpleNamespace

    def _s(weights):  # a sample's input_ids length matches its loss_weights length
        return {"input_ids": list(range(len(weights))), "loss_weights": weights}

    gen = SimpleNamespace(cutoff_len=4, _warned_truncation=False)
    samples = [
        _s([0.0, 0.0, 1.0, 1.0]),  # fits cutoff (len 4), supervised -> kept
        _s([0.0, 0.0, 0.0, 0.0, 1.0, 1.0]),  # len 6 > 4, supervision only beyond cutoff -> dropped
        _s([1.0, 1.0]),  # short, fully supervised -> kept
        _s([0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),  # len 6 > 4 but supervision within cutoff -> kept
    ]
    kept = BatchGenerator._drop_unsupervised(gen, samples)
    assert kept == [samples[0], samples[2], samples[3]]
    assert gen._warned_truncation is True
