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

from pathlib import Path

from llamafactory.webui import common


def test_load_config_with_empty_file(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(common, "DEFAULT_CACHE_DIR", str(tmp_path))
    (tmp_path / common.USER_CONFIG).write_text("", encoding="utf-8")

    assert common.load_config() == {
        "lang": None,
        "hub_name": None,
        "last_model": None,
        "path_dict": {},
        "cache_dir": None,
    }


def test_save_config_with_incomplete_file(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(common, "DEFAULT_CACHE_DIR", str(tmp_path))
    (tmp_path / common.USER_CONFIG).write_text("lang: zh\n", encoding="utf-8")

    common.save_config("", model_name="custom", model_path="path/to/model")

    assert common.load_config() == {
        "lang": "zh",
        "hub_name": None,
        "last_model": "custom",
        "path_dict": {"custom": "path/to/model"},
        "cache_dir": None,
    }
