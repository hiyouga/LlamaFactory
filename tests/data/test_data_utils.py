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

from llamafactory.data import data_utils


def test_read_cloud_json_from_gcs_directory(monkeypatch):
    fs = fsspec.filesystem("memory", skip_instance_cache=True)
    cloud_dir = "gs://llamafactory-test/dataset"
    with fs.open(f"{cloud_dir}/train.json", "w") as f:
        f.write('[{"id": 1}]')

    with fs.open(f"{cloud_dir}/eval.jsonl", "w") as f:
        f.write('{"id": 2}\n{"id": 3}\n')

    with fs.open(f"{cloud_dir}/README.txt", "w") as f:
        f.write("not a dataset")

    monkeypatch.setattr(data_utils, "setup_fs", lambda path, anon=False: fs)

    records = data_utils.read_cloud_json(cloud_dir)
    assert sorted(records, key=lambda record: record["id"]) == [{"id": 1}, {"id": 2}, {"id": 3}]
