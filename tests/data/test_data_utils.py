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

import fsspec
import pytest

from llamafactory.data import data_utils


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
def test_read_cloud_json_directory(monkeypatch, tmp_path):
    filesystem = fsspec.filesystem("memory")
    directory = f"/{tmp_path.name}"
    filesystem.pipe(f"{directory}/data.json", b'[{"value": 1}]')
    filesystem.pipe(f"{directory}/data.jsonl", b'{"value": 2}\n{"value": 3}\n')
    filesystem.pipe(f"{directory}/ignored.txt", b"not json")
    monkeypatch.setattr(data_utils, "setup_fs", lambda *args, **kwargs: filesystem)

    records = data_utils.read_cloud_json(directory)

    assert sorted(record["value"] for record in records) == [1, 2, 3]
