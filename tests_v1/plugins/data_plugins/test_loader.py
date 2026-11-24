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

from llamafactory.v1.plugins.data_plugins.loader import DataLoaderPlugin


@pytest.mark.parametrize("streaming", [False, True])
def test_local_directory_uses_named_splits(tmp_path, streaming):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "README.md").write_text("# Dataset fixture", encoding="utf-8")
    (root / ".keep").touch()
    for split in ("train", "test"):
        folder = root / split
        folder.mkdir()
        (folder / "data.jsonl").write_text(json.dumps({"text": split}) + "\n", encoding="utf-8")

    dataset = DataLoaderPlugin("local").load({"path": str(root), "split": "test", "streaming": streaming})

    assert list(dataset) == [{"text": "test"}]


def test_local_file_keeps_existing_loading(tmp_path):
    path = tmp_path / "examples.jsonl"
    path.write_text('{"text": "example"}\n', encoding="utf-8")

    dataset = DataLoaderPlugin("local").load({"path": str(path)})

    assert list(dataset) == [{"text": "example"}]
