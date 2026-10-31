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
from dataclasses import dataclass
from typing import Any, Literal

from huggingface_hub import hf_hub_download

from ..extras.constants import DATA_CONFIG
from ..extras.misc import use_modelscope, use_openmind


@dataclass
class DatasetAttr:
    r"""Dataset attributes."""

    # basic configs
    load_from: Literal["hf_hub", "ms_hub", "om_hub", "script", "file"]
    dataset_name: str
    formatting: Literal["alpaca", "sharegpt", "openai"] = "alpaca"
    ranking: bool = False
    # extra configs
    subset: str | None = None
    split: str = "train"
    folder: str | None = None
    num_samples: int | None = None
    # common columns
    system: str | None = None
    tools: str | None = None
    images: str | None = None
    videos: str | None = None
    audios: str | None = None
    # dpo columns
    chosen: str | None = None
    rejected: str | None = None
    kto_tag: str | None = None
    # alpaca columns
    prompt: str | None = "instruction"
    query: str | None = "input"
    response: str | None = "output"
    history: str | None = None
    # sharegpt columns
    messages: str | None = "conversations"
    # sharegpt tags
    role_tag: str | None = "from"
    content_tag: str | None = "value"
    user_tag: str | None = "human"
    assistant_tag: str | None = "gpt"
    observation_tag: str | None = "observation"
    function_tag: str | None = "function_call"
    system_tag: str | None = "system"

    def __repr__(self) -> str:
        return self.dataset_name

    def set_attr(self, key: str, obj: dict[str, Any], default: Any | None = None) -> None:
        setattr(self, key, obj.get(key, default))

    def join(self, attr: dict[str, Any]) -> None:
        self.set_attr("formatting", attr, default="alpaca")
        self.set_attr("ranking", attr, default=False)
        self.set_attr("subset", attr)
        self.set_attr("split", attr, default="train")
        self.set_attr("folder", attr)
        self.set_attr("num_samples", attr)

        if "columns" in attr:
            column_names = ["prompt", "query", "response", "history", "messages", "system", "tools"]
            column_names += ["images", "videos", "audios", "chosen", "rejected", "kto_tag"]
            for column_name in column_names:
                self.set_attr(column_name, attr["columns"])

        if "tags" in attr:
            tag_names = ["role_tag", "content_tag"]
            tag_names += ["user_tag", "assistant_tag", "observation_tag", "function_tag", "system_tag"]
            for tag in tag_names:
                self.set_attr(tag, attr["tags"])


def get_dataset_list(dataset_names: list[str] | None, dataset_dir: str | dict) -> list["DatasetAttr"]:
    r"""Get the attributes of the datasets."""
    if dataset_names is None:
        dataset_names = []

    if isinstance(dataset_dir, dict):
        dataset_info = dataset_dir
    elif dataset_dir == "ONLINE":
        dataset_info = None
    else:
        if dataset_dir.startswith("REMOTE:"):
            config_path = hf_hub_download(repo_id=dataset_dir[7:], filename=DATA_CONFIG, repo_type="dataset")
        else:
            config_path = os.path.join(dataset_dir, DATA_CONFIG)

        try:
            with open(config_path, encoding="utf-8") as f:
                dataset_info = json.load(f)
        except Exception as err:
            if len(dataset_names) != 0:
                raise ValueError(f"Cannot open {config_path} due to {str(err)}.")

            dataset_info = None

    dataset_list: list[DatasetAttr] = []
    for name in dataset_names:
        if dataset_info is None:  # dataset_dir is ONLINE
            load_from = "ms_hub" if use_modelscope() else "om_hub" if use_openmind() else "hf_hub"
            dataset_attr = DatasetAttr(load_from, dataset_name=name)
            dataset_list.append(dataset_attr)
            continue

        if name not in dataset_info:
            raise ValueError(f"Undefined dataset {name} in {DATA_CONFIG}.")

        has_hf_url = "hf_hub_url" in dataset_info[name]
        has_ms_url = "ms_hub_url" in dataset_info[name]
        has_om_url = "om_hub_url" in dataset_info[name]

        if has_hf_url or has_ms_url or has_om_url:
            if has_ms_url and (use_modelscope() or not has_hf_url):
                dataset_attr = DatasetAttr("ms_hub", dataset_name=dataset_info[name]["ms_hub_url"])
            elif has_om_url and (use_openmind() or not has_hf_url):
                dataset_attr = DatasetAttr("om_hub", dataset_name=dataset_info[name]["om_hub_url"])
            else:
                dataset_attr = DatasetAttr("hf_hub", dataset_name=dataset_info[name]["hf_hub_url"])
        elif "script_url" in dataset_info[name]:
            dataset_attr = DatasetAttr("script", dataset_name=dataset_info[name]["script_url"])
        elif "cloud_file_name" in dataset_info[name]:
            dataset_attr = DatasetAttr("cloud_file", dataset_name=dataset_info[name]["cloud_file_name"])
        else:
            dataset_attr = DatasetAttr("file", dataset_name=dataset_info[name]["file_name"])

        dataset_attr.join(dataset_info[name])
        dataset_list.append(dataset_attr)

    return dataset_list
