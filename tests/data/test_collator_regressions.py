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

import torch

from llamafactory.data import collator as collator_module
from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq


def test_unpacked_rope_path_passes_batch_index(monkeypatch):
    """The non-packed path must use the helper's batch_idx keyword (issue #10497)."""
    seen_indices: list[int] = []

    def record_slice(
        mm_inputs,
        batch_imglens,
        batch_vidlens,
        batch_idx,
        images_per_subseq=None,
        videos_per_subseq=None,
        subseq_idx=None,
    ):
        del (
            mm_inputs,
            batch_imglens,
            batch_vidlens,
            images_per_subseq,
            videos_per_subseq,
            subseq_idx,
        )
        seen_indices.append(batch_idx)
        return {}

    monkeypatch.setattr(collator_module, "_slice_mm_inputs_for_sample", record_slice)
    collator = object.__new__(MultiModalDataCollatorForSeq2Seq)

    def fake_compute(features, mm_inputs):
        del mm_inputs
        seq_len = features["input_ids"].shape[-1]
        features["position_ids"] = torch.arange(seq_len).view(1, seq_len)
        features["rope_deltas"] = torch.zeros(features["input_ids"].shape[0])

    collator._compute_rope_position_ids = fake_compute
    features = {
        "input_ids": torch.tensor([[10, 11, 12]]),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
    }

    collator._compute_rope_position_ids_with_packing(
        features,
        {},
        packing_params_list=[{}],
        batch_imglens=[0],
        batch_vidlens=[0],
        batch_audlens=[0],
        has_dummy_image=False,
    )

    assert seen_indices == [0]
    assert features["position_ids"].shape == (1, 3)
