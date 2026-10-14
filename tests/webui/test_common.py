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

import pytest

from llamafactory.webui.common import load_dataset_info


def test_load_dataset_info_valid(tmp_path):
    dataset_info = {"alpaca": {"file_name": "alpaca.json"}}
    (tmp_path / "dataset_info.json").write_text(json.dumps(dataset_info), encoding="utf-8")
    assert load_dataset_info(str(tmp_path)) == dataset_info


def test_load_dataset_info_missing_file_returns_empty(tmp_path):
    r"""A dataset dir without a dataset_info.json is not an error, just no datasets to list."""
    assert load_dataset_info(str(tmp_path)) == {}


def test_load_dataset_info_malformed_json_raises(tmp_path):
    r"""Malformed JSON must raise instead of failing silently (see #9060)."""
    (tmp_path / "dataset_info.json").write_text("{ invalid json,,, }", encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset_info(str(tmp_path))


def test_load_dataset_info_invalid_entry_raises(tmp_path):
    r"""A dataset entry that is not a JSON object must raise instead of failing silently (see #9060)."""
    (tmp_path / "dataset_info.json").write_text(json.dumps({"alpaca": "not_an_object"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset_info(str(tmp_path))


def test_load_dataset_info_non_dict_root_raises(tmp_path):
    (tmp_path / "dataset_info.json").write_text(json.dumps(["alpaca"]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset_info(str(tmp_path))


def test_load_dataset_info_non_utf8_raises(tmp_path):
    r"""A dataset_info.json that is not valid UTF-8 must raise, not fall through as a silent empty dict."""
    (tmp_path / "dataset_info.json").write_bytes(b'{"alpaca": {"file_name": "\xff\xfe.json"}}')
    with pytest.raises(ValueError):
        load_dataset_info(str(tmp_path))
