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

"""The definition of data engine.

How to use:
data_engine = DataEngine(data_args.train_dataset)
data_engine[i]: Get the sample via index.

Init workflow:
1. Parse dataset info from arguments.
2. Load datasets according to dataset info.
3. Build data index (and reweight samples if necessary).

Get data sample:
1. Get sample from data index.
2. Convert sample to standard format.
3. Return sample.

Note:
1. The data engine is equivalent to the torch dataset.
2. The data engine is agnostic to the model used.
"""

import os
from collections.abc import Iterable
from typing import Any

from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from ..utils.types import DatasetInfo, HFDataset, Sample


class DataEngine(Dataset):
    """Data engine.

    Args:
        data_args: Data arguments.
    """

    def __init__(self, dataset_path: str) -> None:
        self.path = dataset_path
        """Dataset path."""
        self.datasets: dict[str, HFDataset] = {}
        """Dict of (dataset_name, dataset)"""
        self.dataset_infos: dict[str, DatasetInfo] = {}
        """Dict of (dataset_name, dataset_info)"""
        self.data_index: list[tuple[str, int, int | None]] = []
        """List of (dataset_name, sample_index, cut). ``cut`` is the prefix length for multi-turn
        split (messages[:cut], ending at one supervised assistant turn), or None for a whole sample."""
        self.streaming: bool = False
        """Whether dataset is streaming."""
        self._get_dataset_info()
        self._load_dataset()
        self._build_data_index()

    def _get_dataset_info(self) -> None:
        """Get dataset info from data arguments."""
        if self.path.endswith(".yaml") and os.path.isfile(self.path):  # local file
            self.dataset_infos = OmegaConf.load(self.path)
        elif self.path.endswith(".yaml"):  # hf hub uri, e.g. llamafactory/v1-sft-demo/dataset_info.yaml
            repo_id, filename = os.path.split(self.path)
            filepath = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
            self.dataset_infos = OmegaConf.load(filepath)
        elif os.path.exists(self.path):  # local file(s)
            self.dataset_infos = {"default": {"path": self.path, "source": "local"}}
        else:  # hf hub dataset, e.g. llamafactory/v1-sft-demo
            self.dataset_infos = {"default": {"path": self.path}}

    def _load_dataset(self) -> None:
        """Load datasets according to dataset info."""
        is_streaming = [dataset_info.get("streaming", False) for dataset_info in self.dataset_infos.values()]
        self.streaming = any(is_streaming)
        if all(is_streaming) != any(is_streaming):
            raise ValueError("All datasets must be streaming or non-streaming.")

        for dataset_name, dataset_info in self.dataset_infos.items():
            split = dataset_info.get("split", "train")
            if dataset_info.get("source", "hf_hub") == "hf_hub":
                from datasets import load_dataset

                self.datasets[dataset_name] = load_dataset(dataset_info["path"], split=split, streaming=self.streaming)
            else:  # data loader plugin
                from ..plugins.data_plugins.loader import DataLoaderPlugin

                self.datasets[dataset_name] = DataLoaderPlugin(dataset_info["source"]).load(dataset_info)

    def _build_data_index(self) -> None:
        """Build dataset index.

        Multi-turn SFT conversations are prefix-expanded: one index entry per supervised assistant
        turn, so ``len()`` reflects the true number of training samples (each trained on its last
        turn). Entries are ``(dataset_name, sample_index, cut)``; ``cut`` is the prefix length
        ``messages[:cut]`` ending at one supervised turn, or ``None`` for a whole sample (DPO,
        streaming, or no supervised turn).
        """
        for dataset_name, dataset in self.datasets.items():
            if self.streaming:  # cannot pre-count turns -> keep whole, unsplit
                data_index = [(dataset_name, -1, None) for _ in range(1000)]
            else:
                data_index = []
                for sample_index in range(len(dataset)):
                    sample = self._convert_data_sample(dataset[sample_index], dataset_name)
                    for cut in self._prefix_cuts(sample):
                        data_index.append((dataset_name, sample_index, cut))

            size = self.dataset_infos[dataset_name].get("size")
            weight = self.dataset_infos[dataset_name].get("weight")
            if size or weight:
                from ..plugins.data_plugins.loader import adjust_data_index

                data_index = adjust_data_index(data_index, size, weight)

            self.data_index.extend(data_index)

    @staticmethod
    def _prefix_cuts(sample: Sample) -> list[int | None]:
        """Prefix lengths to split a multi-turn conversation on: one per supervised assistant turn.

        ``u1 a1 u2 a2`` -> ``[2, 4]`` (samples ``messages[:2]`` and ``messages[:4]``, each trained on
        its last assistant turn). Non-SFT samples (no ``messages``) or those with no supervised turn
        are kept whole (``[None]``).
        """
        messages = sample.get("messages")
        if not messages:
            return [None]
        cuts = [i + 1 for i, m in enumerate(messages) if m["role"] == "assistant" and m.get("loss_weight", 1.0) > 1e-6]
        return cuts or [None]

    def _convert_data_sample(self, raw_sample: dict[str, Any], dataset_name: str) -> Sample:
        """Convert dataset sample.

        Args:
            raw_sample (dict[str, Any]): Raw dataset sample.
            dataset_name (str): Dataset name.

        Returns:
            Sample: Dataset sample.
        """
        converter = self.dataset_infos[dataset_name].get("converter")
        if converter is not None:
            from ..plugins.data_plugins.converter import DataConverterPlugin

            return {"_dataset_name": dataset_name, **DataConverterPlugin(converter)(raw_sample)}
        else:
            return {"_dataset_name": dataset_name, **raw_sample}

    def __len__(self) -> int:
        """Get dataset length.

        Returns:
            int: Dataset length.
        """
        if self.streaming:
            return -1
        else:
            return len(self.data_index)

    def __getitem__(self, index: int | Any) -> Sample | list[Sample]:
        """Get dataset item.

        Args:
            index (int): Dataset index.

        Returns:
            Sample: Dataset item.
        """
        if self.streaming:
            raise ValueError("Streaming dataset does not support index access.")

        if isinstance(index, int):
            return self._get(*self.data_index[index])
        else:  # data selector plugin
            from ..plugins.data_plugins.loader import select_data_sample

            selected_index = select_data_sample(self.data_index, index)
            if isinstance(selected_index, list):
                return [self._get(*entry) for entry in selected_index]
            else:
                return self._get(*selected_index)

    def _get(self, dataset_name: str, sample_index: int, cut: int | None = None) -> Sample:
        """Convert one raw row, truncating to a multi-turn prefix when ``cut`` is set."""
        sample = self._convert_data_sample(self.datasets[dataset_name][sample_index], dataset_name)
        if cut is not None and "messages" in sample:
            sample = {**sample, "messages": sample["messages"][:cut]}
        return sample

    def __iter__(self) -> Iterable[Sample]:
        """Get dataset iterator.

        Returns:
            Iterable[Sample]: Dataset iterator.
        """
        # NOTE: hf iterable dataset uses worker ids while map dataset does not
        # NOTE: add worker id and shuffle to the map dataset
        # https://github.com/huggingface/datasets/blob/4.0.0/src/datasets/iterable_dataset.py#L2214

        raise NotImplementedError()


if __name__ == "__main__":
    """
    python -m llamafactory.v1.core.data_engine --train_dataset data/v1_sft_demo.yaml
    python -m llamafactory.v1.core.data_engine --train_dataset data/v1_dpo_demo.yaml
    """
    from ..config.arg_parser import get_args

    _, data_args, *_ = get_args()
    data_engine = DataEngine(data_args.train_dataset)
    print(data_engine[0])
