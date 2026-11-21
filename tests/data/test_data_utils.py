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

from unittest.mock import MagicMock, patch

import pytest

from llamafactory.data.data_utils import read_cloud_json


@pytest.mark.runs_on(["cpu", "mps"])
def test_read_cloud_json_directory_listing_uses_fsspec_standard_name_field():
    # Both s3fs and gcsfs populate "name" on every listdir() entry; only s3fs also sets the
    # S3-specific "Key" alias. Using "Key" breaks GCS directory listings with a bare KeyError.
    mock_fs = MagicMock()
    mock_fs.isdir.return_value = True
    mock_fs.listdir.return_value = [
        {"name": "my-bucket/data/train.jsonl", "size": 123, "type": "file"},
        {"name": "my-bucket/data/README.md", "size": 45, "type": "file"},
    ]

    with (
        patch("llamafactory.data.data_utils.setup_fs", return_value=mock_fs),
        patch("llamafactory.data.data_utils._read_json_with_fs", return_value=[{"text": "hello"}]) as mock_read,
    ):
        result = read_cloud_json("gs://my-bucket/data/")

    mock_read.assert_called_once_with(mock_fs, "my-bucket/data/train.jsonl")
    assert result == [{"text": "hello"}]
