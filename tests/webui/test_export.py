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

from unittest.mock import Mock

import pytest

from llamafactory.webui.components import export


@pytest.mark.parametrize("private", [False, True])
def test_save_model_hub_privacy(monkeypatch, tmp_path, private):
    export_model = Mock()
    monkeypatch.setattr(export, "export_model", export_model)
    monkeypatch.setattr(export, "load_config", lambda: {})
    monkeypatch.setattr(export, "torch_gc", lambda: None)

    messages = list(
        export.save_model(
            lang="en",
            model_name="test-model",
            model_path="test-model",
            finetuning_type="full",
            checkpoint_path="checkpoint",
            template="empty",
            export_size=5,
            export_quantization_bit="none",
            export_quantization_dataset="",
            export_device="cpu",
            export_legacy_format=False,
            export_dir=str(tmp_path),
            export_hub_model_id="test-user/test-model",
            extra_args="{}",
            export_hub_private=private,
        )
    )

    export_model.assert_called_once()
    args = export_model.call_args.args[0]
    assert args["export_hub_model_id"] == "test-user/test-model"
    assert args["export_hub_private"] is private
    assert messages[-1] == export.ALERTS["info_exported"]["en"]
