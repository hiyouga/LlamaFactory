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

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_moss_vl_training_configs_are_unpacked_and_additive():
    lora = yaml.safe_load((ROOT / "examples/train_lora/mossvl_lora_sft.yaml").read_text())
    full = yaml.safe_load((ROOT / "examples/train_full/mossvl_full_sft.yaml").read_text())

    assert lora["template"] == full["template"] == "moss_vl"
    assert lora["packing"] is full["packing"] is False
    assert lora["per_device_train_batch_size"] == 2
    assert full["per_device_train_batch_size"] == 1
    assert full["use_reentrant_gc"] is False
    assert full["gradient_checkpointing"] is True
    assert full["gradient_checkpointing_kwargs"] == {"use_reentrant": False}
    for config in (lora, full):
        assert config["model_name_or_path"] == "OpenMOSS-Team/MOSS-VL-Instruct-0708"
        assert not any("/inspire/" in str(value) or "/tmp/" in str(value) for value in config.values())
        assert config["freeze_vision_tower"] is True
        assert config["freeze_multi_modal_projector"] is True
        assert config["freeze_language_model"] is False
