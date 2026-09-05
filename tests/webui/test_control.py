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

import json
import os

from llamafactory.webui.control import list_datasets


def test_list_datasets_warns_on_malformed_json(tmp_path, monkeypatch):
    r"""The dataset dropdown must surface a user-facing warning instead of failing silently (see #9060)."""
    (tmp_path / "dataset_info.json").write_text("{ invalid json,,, }", encoding="utf-8")

    warnings = []
    monkeypatch.setattr("llamafactory.webui.control.gr.Warning", lambda msg, *args, **kwargs: warnings.append(msg))

    result = list_datasets("en", str(tmp_path), "Supervised Fine-Tuning")

    assert len(warnings) == 1
    assert result.choices == []


def test_list_datasets_valid(tmp_path):
    dataset_info = {"alpaca": {"file_name": "alpaca.json"}}
    with open(os.path.join(tmp_path, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_info, f)

    result = list_datasets("en", str(tmp_path), "Supervised Fine-Tuning")
    assert result.choices == [("alpaca", "alpaca")]
