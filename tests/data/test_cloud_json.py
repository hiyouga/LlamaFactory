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
from fsspec.implementations.memory import MemoryFileSystem

from llamafactory.data import data_utils


@pytest.fixture
def cloud_fs(monkeypatch, tmp_path):
    fs = MemoryFileSystem()
    root = f"bucket/{tmp_path.name}"
    monkeypatch.setattr(data_utils, "setup_fs", lambda _path, anon=False: fs)
    yield fs, root
    for protocol in ("s3", "gs", "gcs"):
        path = f"{protocol}://{root}"
        if fs.exists(path):
            fs.rm(path, recursive=True)


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
@pytest.mark.parametrize("protocol", ["s3", "gs", "gcs"])
def test_read_cloud_json_directory_uses_filesystem_listing_names(cloud_fs, protocol):
    fs, root = cloud_fs
    directory = f"{protocol}://{root}"
    fs.pipe(f"{directory}/first.json", json.dumps([{"id": 1}]).encode())
    fs.pipe(f"{directory}/second.jsonl", b'{"id": 2}\n\n{"id": 3}\n')
    fs.pipe(f"{directory}/ignored.txt", b"not json")

    assert all("name" in entry and "Key" not in entry for entry in fs.ls(directory))
    result = data_utils.read_cloud_json(directory)
    assert sorted(row["id"] for row in result) == [1, 2, 3]


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
@pytest.mark.parametrize("protocol", ["s3", "gs", "gcs"])
@pytest.mark.parametrize("extension,content", [("json", b'[{"id": 1}]'), ("jsonl", b'{"id": 1}\n')])
def test_read_cloud_json_still_accepts_a_single_file(cloud_fs, protocol, extension, content):
    fs, root = cloud_fs
    path = f"{protocol}://{root}/data.{extension}"
    fs.pipe(path, content)
    assert data_utils.read_cloud_json(path) == [{"id": 1}]


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
def test_read_cloud_json_reports_directory_without_json_files(cloud_fs):
    fs, root = cloud_fs
    directory = f"gs://{root}"
    fs.pipe(f"{directory}/ignored.txt", b"not json")
    with pytest.raises(ValueError, match="No JSON/JSONL files found"):
        data_utils.read_cloud_json(directory)
