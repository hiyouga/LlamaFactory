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

"""Batching utils supports stateful dataloader.

1. Init stateful dataloader (tokenize)
2. Add to buffer
3. Yield batch indexes (micro batch * grad acc)
    a) non pack + non dynamic
    b) non pack + dynamic
    c) pack + non dynamic
    d) pack + dynamic
"""

from collections.abc import Iterator
from typing import Any

import torch
from torch.utils.data import default_collate
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from ...accelerator.helper import ReduceOp
from ...accelerator.interface import Dim, DistributedInterface
from ...config import BatchingStrategy
from ...utils import logging
from ...utils.constants import IGNORE_INDEX
from ...utils.helper import is_tokenizer
from ...utils.objects import StatefulBuffer
from ...utils.types import BatchInfo, BatchInput, ModelInput, Tensor, TorchDataset
from ..rendering import Renderer
from .collation import _MULTIMODAL_PASSTHROUGH_KEYS, pad_and_truncate


logger = logging.get_logger(__name__)

__all__ = ["BatchGenerator"]


# (modality, presence/feature key, grid key, mm_token_type_ids marker) for encoder-tower alignment.
# The presence key is what survives collation when the modality is present; the grid key is unused
# here (kept for parity with the collation specs). Audio carries no grid -- feature_attention_mask
# rides along as a passthrough feature.
_ALIGN_MODALITIES = (
    ("image", "pixel_values", "image_grid_thw", 1),
    ("video", "pixel_values_videos", "video_grid_thw", 2),
    ("audio", "input_features", "feature_attention_mask", 3),
)


def _collate_micro_batch(micro_batch: list[ModelInput], cutoff_len: int) -> BatchInput:
    """Pad/truncate then collate one micro batch (text fields stacked, MM features dim-0 concat)."""
    padded = pad_and_truncate(micro_batch, cutoff_len)
    standard_samples = [{k: v for k, v in s.items() if k not in _MULTIMODAL_PASSTHROUGH_KEYS} for s in padded]
    collated = default_collate(standard_samples)
    for key in _MULTIMODAL_PASSTHROUGH_KEYS:
        tensors = [s[key] for s in padded if key in s]
        if tensors:
            collated[key] = torch.cat(tensors, dim=0)
    return collated


def _inject_dummy_into_collated(collated: BatchInput, fragment: dict, marker: int) -> None:
    """Append a zero-loss dummy media fragment to an already-collated micro batch, in place.

    Operates *after* pad_and_truncate so it reflects post-truncation presence: an image whose
    placeholder tokens were partially cut is deleted by ``_align_multimodal_on_truncation``,
    turning that sample text-only -- which must be detected here (not before truncation) or the
    vision-tower call count still desyncs across ranks.

    The dummy tokens are appended (extra columns) into row 0 only; other rows get padding there.
    Causal attention keeps every real token's logits unchanged; the dummy carries IGNORE_INDEX
    labels and zero loss weight, so it contributes nothing to the loss while forcing the (FSDP-
    sharded) vision tower to run.
    """
    bsz, seqlen = collated["input_ids"].shape
    frag_ids = torch.tensor(fragment["input_ids"], dtype=collated["input_ids"].dtype)
    frag_len = frag_ids.numel()
    frag_mm = torch.tensor(fragment["mm_token_type_ids"], dtype=torch.long)
    new_len = seqlen + frag_len

    def _grow(tensor: Tensor, pad_value, row0_tail=None) -> Tensor:
        out = torch.full((bsz, new_len), pad_value, dtype=tensor.dtype)
        out[:, :seqlen] = tensor
        if row0_tail is not None:
            out[0, seqlen:] = row0_tail.to(tensor.dtype)
        return out

    collated["input_ids"] = _grow(collated["input_ids"], 0, frag_ids)
    collated["attention_mask"] = _grow(collated["attention_mask"], 0)
    collated["attention_mask"][0, seqlen:] = 1
    collated["labels"] = _grow(collated["labels"], IGNORE_INDEX)  # dummy region stays ignored
    collated["loss_weights"] = _grow(collated["loss_weights"], 0.0)
    if "position_ids" in collated:
        pos = _grow(collated["position_ids"], 0)
        pos[0, seqlen:] = torch.arange(seqlen + 1, new_len + 1, dtype=pos.dtype)
        collated["position_ids"] = pos

    mm = collated.get("mm_token_type_ids")
    if mm is not None:
        collated["mm_token_type_ids"] = _grow(mm, 0, frag_mm)
    else:
        mm = torch.zeros((bsz, new_len), dtype=torch.long)
        mm[0, seqlen:] = frag_mm
        collated["mm_token_type_ids"] = mm

    for key, value in fragment.items():
        if key in ("input_ids", "mm_token_type_ids"):
            continue
        collated[key] = torch.cat([collated[key], value], dim=0) if key in collated else value


def default_collate_fn(
    buffer: StatefulBuffer, batch_info: BatchInfo, renderer: Renderer | None = None
) -> list[BatchInput] | None:
    micro_batch_size = batch_info["micro_batch_size"]
    num_micro_batch = batch_info["num_micro_batch"]
    cutoff_len = batch_info["cutoff_len"]
    batch_size = micro_batch_size * num_micro_batch
    if len(buffer) < batch_size:
        return None

    samples = buffer.get(batch_size)
    micro_batches = [samples[i * micro_batch_size : (i + 1) * micro_batch_size] for i in range(num_micro_batch)]

    # Collate first; presence is judged on the *post-truncation* result, since truncation can
    # delete a partially-cut image and turn a sample text-only (see _inject_dummy_into_collated).
    batch = [_collate_micro_batch(mb, cutoff_len) for mb in micro_batches]

    if renderer is not None and not is_tokenizer(renderer.processor):
        present = torch.zeros((num_micro_batch, len(_ALIGN_MODALITIES)), dtype=torch.int64)
        for i, collated in enumerate(batch):
            for m, (_, pixel_key, _, _) in enumerate(_ALIGN_MODALITIES):
                present[i, m] = int(pixel_key in collated)

        present = DistributedInterface().all_reduce(present, op=ReduceOp.MAX, dim=Dim.DP)

        for i, collated in enumerate(batch):
            for m, (modality, pixel_key, _, marker) in enumerate(_ALIGN_MODALITIES):
                if present[i, m] and pixel_key not in collated:
                    _inject_dummy_into_collated(collated, renderer.get_dummy_media_fragment(modality), marker)

    return batch


class BatchGenerator(Iterator):
    def __init__(
        self,
        dataset: TorchDataset,
        renderer: Renderer,
        micro_batch_size: int = 1,
        global_batch_size: int | None = None,
        cutoff_len: int = 2048,
        batching_workers: int = 0,
        batching_strategy: BatchingStrategy = BatchingStrategy.NORMAL,
        pin_memory: bool = True,
        drop_last: bool = True,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.renderer = renderer

        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.cutoff_len = cutoff_len
        self.batching_workers = batching_workers
        self.batching_strategy = batching_strategy
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.seed = seed
        self._warned_truncation = False  # warn once when dropping fully-truncated (zero-loss) samples
        # TODO: support length and infinity
        dp_size = DistributedInterface().get_world_size(Dim.DP)

        if self.global_batch_size is None:
            self.global_batch_size = dp_size * micro_batch_size
            self.num_micro_batch = 1
        elif self.global_batch_size % (dp_size * micro_batch_size) == 0:
            self.num_micro_batch = global_batch_size // dp_size // micro_batch_size
        else:
            raise ValueError(
                "Global batch size must be divisible by DP size and micro batch size. "
                f"Got {global_batch_size} % ({dp_size} * {micro_batch_size}) != 0."
            )

        if not self.drop_last:
            raise ValueError("Drop last must be True.")

        self._batch_info: BatchInfo = {
            "micro_batch_size": self.micro_batch_size,
            "num_micro_batch": self.num_micro_batch,
            "cutoff_len": self.cutoff_len,
        }

        self._init_data_provider()

        self._is_resuming: bool = False
        self._data_iter = iter(self._data_provider)
        self._buffer = StatefulBuffer()

        logger.info_rank0(
            f"Init unified data loader with global batch size {self.global_batch_size}, "
            f"micro batch size {self.micro_batch_size}, "
            f"num micro batch {self.num_micro_batch}, "
            f"cutoff len {self.cutoff_len}, "
            f"batching workers {self.batching_workers}, "
            f"batching strategy {self.batching_strategy}."
        )

    def _init_data_provider(self) -> None:
        if len(self.dataset) != -1:
            sampler = StatefulDistributedSampler(
                self.dataset,
                num_replicas=DistributedInterface().get_world_size(Dim.DP),
                rank=DistributedInterface().get_rank(Dim.DP),
                shuffle=True,
                seed=self.seed,
                drop_last=self.drop_last,
            )
        else:
            raise NotImplementedError("Iterable dataset is not supported yet.")

        if self.batching_strategy == BatchingStrategy.NORMAL:
            batch_size = self.micro_batch_size * self.num_micro_batch
        else:
            from ...plugins.trainer_plugins.batching import BatchingPlugin

            batch_size = BatchingPlugin(self.batching_strategy).get_data_provider_batch_size(self._batch_info)

        generator_seed = torch.Generator()
        generator_seed.manual_seed(self.seed)

        self._data_provider = StatefulDataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.batching_workers,
            collate_fn=self.renderer.process_samples,
            pin_memory=self.pin_memory,
            pin_memory_device=DistributedInterface().current_device.type,
            drop_last=self.drop_last,
            generator=generator_seed,
        )
        if self.batching_strategy == BatchingStrategy.NORMAL:
            self._length = len(self._data_provider)
        else:
            from ...plugins.trainer_plugins.batching import BatchingPlugin

            self._length = BatchingPlugin(self.batching_strategy).compute_length(self._data_provider, self._batch_info)

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        if not self._is_resuming:
            self._buffer.clear()

        self._data_iter = iter(self._data_provider)
        self._is_resuming = False
        return self

    def __next__(self):
        self._fill_buffer()
        batch = self._generate_batch()
        if batch is None:
            raise StopIteration

        return batch

    def _drop_unsupervised(self, samples: list[ModelInput]) -> list[ModelInput]:
        """Drop samples whose supervised span is entirely beyond ``cutoff_len``.

        Prefix-split puts the supervised tokens at the tail, and truncation keeps ``[:cutoff_len]``,
        so a sample longer than ``cutoff_len`` loses all supervision and would contribute a zero-loss
        (wasted) step. Only such over-length samples are at risk -- samples that fit within
        ``cutoff_len`` are never truncated and always keep supervision -- so they pass an O(1) length
        test before any ``loss_weights`` scan. Drop the at-risk, fully-masked ones and warn once.
        """
        kept = []
        for sample in samples:
            if len(sample["input_ids"]) > self.cutoff_len and not any(
                w > 1e-6 for w in sample["loss_weights"][: self.cutoff_len]
            ):
                if not self._warned_truncation:
                    self._warned_truncation = True
                    logger.warning_rank0(
                        f"Dropping training sample(s) whose supervised tokens fall entirely beyond "
                        f"cutoff_len={self.cutoff_len} (all loss masked after truncation). "
                        "Increase cutoff_len to keep them."
                    )
                continue
            kept.append(sample)
        return kept

    def _fill_buffer(self) -> None:
        if self.batching_strategy == BatchingStrategy.NORMAL:
            while len(self._buffer) < self.micro_batch_size * self.num_micro_batch:
                try:
                    samples: list[ModelInput] = next(self._data_iter)
                except StopIteration:
                    break

                self._buffer.put(self._drop_unsupervised(samples))
        else:
            from ...plugins.trainer_plugins.batching import BatchingPlugin

            BatchingPlugin(self.batching_strategy).fill_buffer(self._buffer, self._batch_info, self._next_samples)

    def _generate_batch(self) -> list[BatchInput] | None:
        if self.batching_strategy == BatchingStrategy.NORMAL:
            return default_collate_fn(self._buffer, self._batch_info, self.renderer)
        else:
            # Non-NORMAL strategies (dynamic / padding_free) collate ragged pixel tensors with a
            # bare default_collate and have no vision-tower alignment, so multimodal data would
            # crash or hang. Fail loud instead of silently mishandling it.
            if any(k in s for s in self._buffer.samples for k in _MULTIMODAL_PASSTHROUGH_KEYS):
                raise NotImplementedError(
                    f"batching_strategy={self.batching_strategy.value!r} does not support multimodal data; "
                    "use the NORMAL strategy for image/video training."
                )

            from ...plugins.trainer_plugins.batching import BatchingPlugin

            return BatchingPlugin(self.batching_strategy).generate_batch(self._buffer, self._batch_info)

    def _next_samples(self, restart: bool) -> list[ModelInput] | None:
        try:
            return next(self._data_iter)
        except StopIteration:
            if not restart:
                return None

            # Dynamic batching may restart the provider to fill one token-budgeted batch.
            self._data_iter = iter(self._data_provider)
            try:
                return next(self._data_iter)
            except StopIteration:
                return None

    def state_dict(self) -> dict[str, Any]:
        return {
            "buffer": self._buffer.state_dict(),
            "data_provider": self._data_provider.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._buffer.load_state_dict(state["buffer"])
        self._data_provider.load_state_dict(state["data_provider"])
        self._is_resuming = True

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self._data_provider.sampler, "set_epoch"):
            self._data_provider.sampler.set_epoch(epoch)


if __name__ == "__main__":
    """
    python -m llamafactory.v1.core.utils.batching \
        --model llamafactory/tiny-random-qwen2.5 \
        --train_dataset data/v1_sft_demo.yaml \
        --micro_batch_size 2 \
        --global_batch_size 4 \
        --batching_workers 0
    """
    from ...config.arg_parser import get_args
    from ..data_engine import DataEngine
    from ..model_engine import ModelEngine

    model_args, data_args, training_args, _ = get_args()
    data_engine = DataEngine(data_args.train_dataset)
    model_engine = ModelEngine(model_args=model_args)
    batch_generator = BatchGenerator(
        data_engine,
        model_engine.renderer,
        micro_batch_size=training_args.micro_batch_size,
        global_batch_size=training_args.global_batch_size,
        cutoff_len=training_args.cutoff_len,
        batching_workers=training_args.batching_workers,
        batching_strategy=training_args.batching_strategy,
    )
    for batch in batch_generator:
        print(batch)
        print(len(batch))
        print(batch[0]["input_ids"].shape)
        break
