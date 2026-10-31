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

import json
import os
import subprocess
import sys
import textwrap


# "数据.json" and "你好" as escapes: the child interpreter below runs with a non-UTF-8 locale, where
# a non-ASCII command line argument cannot be decoded reliably.
FILE_NAME = "数据.json"
GREETING = "你好"


def _run_with_legacy_locale(code: str, *args: str) -> str:
    r"""Run a snippet in a child interpreter whose default text encoding is not UTF-8.

    This is the default on Windows (cp1252, cp1254, cp936, ...) and on POSIX under LC_ALL=C. The parent keeps its
    own encoding, so the test behaves the same on every platform.
    """
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "LC_ALL": "C", "PYTHONCOERCECLOCALE": "0"}
    env.pop("PYTHONUTF8", None)
    result = subprocess.run(
        [sys.executable, "-X", "utf8=0", "-c", textwrap.dedent(code), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_dataset_info_is_read_as_utf8(tmp_path):
    # a user-edited dataset_info.json is UTF-8 and may name a non-ASCII file
    info = {"my_data": {"file_name": FILE_NAME}}
    (tmp_path / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    code = """
        import sys
        from llamafactory.data.parser import get_dataset_list

        print(get_dataset_list(["my_data"], sys.argv[1])[0].dataset_name)
    """
    assert _run_with_legacy_locale(code, str(tmp_path)) == FILE_NAME


def test_cloud_json_is_read_as_utf8():
    code = """
        import json

        import fsspec

        from llamafactory.data.data_utils import _read_json_with_fs

        fs = fsspec.filesystem("memory")
        with fs.open("/cases/data.json", "wb") as f:
            f.write(json.dumps([{"instruction": "\\u4f60\\u597d"}], ensure_ascii=False).encode("utf-8"))

        print(_read_json_with_fs(fs, "/cases/data.json")[0]["instruction"])
    """
    assert _run_with_legacy_locale(code) == GREETING
