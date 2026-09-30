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

from enum import StrEnum, unique
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, NotRequired, TypedDict, Union


if TYPE_CHECKING:
    import datasets
    import numpy as np
    import torch
    import torch.utils.data
    import transformers
    from torch.distributed import ProcessGroup
    from torch.distributed.fsdp import FullyShardedDataParallel

    Tensor = torch.Tensor
    TensorLike = Union[int, float, list[int], list[float], np.ndarray, Tensor]
    TorchDataset = Union[torch.utils.data.Dataset, torch.utils.data.IterableDataset]
    HFDataset = Union[datasets.Dataset, datasets.IterableDataset]
    DataCollator = transformers.DataCollator
    DataLoader = torch.utils.data.DataLoader
    HFConfig = transformers.PretrainedConfig
    HFModel = transformers.PreTrainedModel
    DistModel = Union[torch.nn.parallel.DistributedDataParallel, FullyShardedDataParallel]
    Processor = Union[transformers.PreTrainedTokenizer, transformers.ProcessorMixin]
    Optimizer = torch.optim.Optimizer
    Scheduler = torch.optim.lr_scheduler.LRScheduler
    ProcessGroup = ProcessGroup
else:
    Tensor = None
    TensorLike = None
    TorchDataset = None
    HFDataset = None
    DataCollator = None
    DataLoader = None
    HFConfig = None
    HFModel = None
    DistModel = None
    Processor = None
    Optimizer = None
    Scheduler = None
    ProcessGroup = None


@unique
class AttentionFunction(StrEnum):
    EAGER = "eager"
    SDPA = "sdpa"
    FLASH_ATTENTION_2 = "flash_attention_2"


class DatasetInfo(TypedDict, total=False):
    path: str
    """Local file path."""
    source: NotRequired[Literal["hf_hub", "ms_hub", "local"]]
    """Dataset source, default to "hf_hub"."""
    split: NotRequired[str]
    """Dataset split, default to "train"."""
    converter: NotRequired[str]
    """Dataset converter, default to None."""
    size: NotRequired[int]
    """Number of samples, default to all samples."""
    weight: NotRequired[float]
    """Dataset weight, default to 1.0."""
    streaming: NotRequired[bool]
    """Is streaming dataset, default to False."""


class Content(TypedDict):
    type: Literal["text", "reasoning", "tool_call", "image_url", "video_url", "audio_url"]
    """Type of the content."""
    value: str
    """Value of the content."""


class Message(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    """Role of the message."""
    content: list[Content]
    """Content of the message."""
    loss_weight: NotRequired[float]
    """Loss weight for this message, default to 1.0. Required in training."""


class SFTSample(TypedDict):
    messages: list[Message]
    """Messages in the sample."""
    tools: NotRequired[str]
    """Tools for the sample in JSON string format."""
    extra_info: NotRequired[str]
    """Extra information for the sample, e.g. kto_labels."""
    _dataset_name: NotRequired[str]
    """Dataset name for the sample."""


class DPOSample(TypedDict):
    chosen_messages: list[Message]
    """Chosen messages in the sample."""
    rejected_messages: list[Message]
    """Rejected messages in the sample."""
    tools: NotRequired[str]
    """Tools for the sample in JSON string format."""
    extra_info: NotRequired[str]
    """Extra information for the sample, e.g. kto_labels."""
    _dataset_name: NotRequired[str]
    """Dataset name for the sample."""


Sample = Union[SFTSample, DPOSample]


class ToolCall(TypedDict):
    name: str
    """Function name."""
    arguments: dict[str, Any]
    """Function arguments."""


class ModelInput(TypedDict, total=False):
    input_ids: list[int]
    """Input ids for the model."""
    attention_mask: list[int]
    """Attention mask for the model."""
    labels: list[int]
    """Labels for the model."""
    loss_weights: list[float]
    """Loss weight for each token, default to 1.0."""
    position_ids: NotRequired[list[int] | list[list[int]]]
    """Position ids for the model (optional)."""
    token_type_ids: NotRequired[list[int]]
    """Token type ids used in DPO, 1 represents the chosen messages, 2 represents the rejected messages."""
    pixel_values: NotRequired[Any]
    """Pixel values for vision models."""
    image_grid_thw: NotRequired[Any]
    """Image grid (temporal, height, width) for vision models."""
    pixel_values_videos: NotRequired[Any]
    """Pixel values for video inputs."""
    video_grid_thw: NotRequired[Any]
    """Video grid (temporal, height, width) for video models."""
    input_features: NotRequired[Any]
    """Audio input features (e.g. mel spectrogram) for audio models."""
    feature_attention_mask: NotRequired[Any]
    """Attention mask over the audio input features."""
    mm_token_type_ids: NotRequired[list[int]]
    """Multimodal token type ids: 0=text, 1=image, 2=video, 3=audio."""


class BatchInput(TypedDict, total=False):
    input_ids: Tensor
    """Input ids for the model."""
    attention_mask: Tensor
    """Attention mask for the model."""
    labels: Tensor
    """Labels for the model."""
    loss_weights: Tensor
    """Loss weight for each token, default to 1.0."""
    position_ids: NotRequired[Tensor]
    """Position ids for the model (optional)."""
    token_type_ids: NotRequired[Tensor]
    """Token type ids used in DPO, 1 represents the chosen messages, 2 represents the rejected messages."""
    pixel_values: NotRequired[Tensor]
    """Pixel values for vision models."""
    image_grid_thw: NotRequired[Tensor]
    """Image grid (temporal, height, width) for vision models."""
    pixel_values_videos: NotRequired[Tensor]
    """Pixel values for video inputs."""
    video_grid_thw: NotRequired[Tensor]
    """Video grid (temporal, height, width) for video models."""
    input_features: NotRequired[Tensor]
    """Audio input features (e.g. mel spectrogram) for audio models."""
    feature_attention_mask: NotRequired[Tensor]
    """Attention mask over the audio input features."""
    mm_token_type_ids: NotRequired[Tensor]
    """Multimodal token type ids: 0=text, 1=image, 2=video, 3=audio."""


class BatchInfo(TypedDict):
    micro_batch_size: int
    """Micro batch size."""
    num_micro_batch: int
    """Number of micro batches."""
    cutoff_len: int
    """Cutoff length."""


class ModelOutput(NamedTuple):
    logits: Tensor
    """Logits for the model."""
