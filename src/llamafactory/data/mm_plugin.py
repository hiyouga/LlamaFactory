# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's Transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llava/processing_llava.py
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

import inspect
import math
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, BinaryIO, Literal, NotRequired, Optional, TypedDict, Union

import numpy as np
import torch
import torchaudio
from transformers.image_utils import get_image_size, is_valid_image, make_flat_list_of_images, to_numpy_array
from transformers.models.mllama.processing_mllama import (
    convert_sparse_cross_attention_mask_to_dense,
    get_cross_attention_token_mask,
)
from transformers.video_utils import make_batched_videos
from typing_extensions import override

from ..extras.constants import AUDIO_PLACEHOLDER, IGNORE_INDEX, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER
from ..extras.packages import is_pillow_available, is_pyav_available, is_transformers_version_greater_than


if is_pillow_available():
    from PIL import Image
    from PIL.Image import Image as ImageObject


if is_pyav_available():
    import av


if TYPE_CHECKING:
    from av.stream import Stream
    from numpy.typing import NDArray
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
    from transformers.image_processing_utils import BaseImageProcessor
    from transformers.video_processing_utils import BaseVideoProcessor

    class EncodedImage(TypedDict):
        path: str | None
        bytes: bytes | None

    ImageInput = Union[str, bytes, EncodedImage, BinaryIO, ImageObject]
    VideoInput = Union[str, BinaryIO, list[list[ImageInput]]]
    AudioInput = Union[str, BinaryIO, NDArray]

    class RegularizedImageOutput(TypedDict):
        images: list[ImageObject]

    class RegularizedVideoOutput(TypedDict):
        videos: list[list[ImageObject]]
        durations: list[float]
        fps_per_video: NotRequired[list[float]]

    class RegularizedAudioOutput(TypedDict):
        audios: list[NDArray]
        sampling_rates: list[float]

    class MMProcessor(ProcessorMixin):
        patch_size: int
        image_seq_length: int
        num_additional_image_tokens: int
        vision_feature_select_strategy: Literal["default", "full"]

        def _get_number_of_features(self, orig_height: int, orig_width: int, height: int, width: int) -> int:
            pass


def _get_paligemma_token_type_ids(imglens: list[int], seqlens: list[int], processor: "MMProcessor") -> list[list[int]]:
    r"""Get paligemma token type ids for computing loss.

    It is slightly different with the original token type ids where the prompt part is 0.

    Returns:
        batch_token_type_ids: shape (batch_size, seq_length)

    """
    batch_token_type_ids = []
    for imglen, seqlen in zip(imglens, seqlens):
        image_seqlen = imglen * processor.image_seq_length
        batch_token_type_ids.append([0] * image_seqlen + [1] * (seqlen - image_seqlen))

    return batch_token_type_ids


def _get_gemma3_token_type_ids(batch_ids: list[list[int]], processor: "MMProcessor"):
    r"""Get gemma3 token type ids for computing loss.

    Returns:
        batch_token_type_ids: shape (batch_size, seq_length)

    """
    image_token_id: int = getattr(processor, "image_token_id")
    batch_token_type_ids = []
    for token_ids in batch_ids:
        token_ids = np.array(token_ids)
        token_type_ids = np.zeros_like(token_ids)
        token_type_ids[token_ids == image_token_id] = 1
        batch_token_type_ids.append(token_type_ids.tolist())

    return batch_token_type_ids


def _make_batched_images(images: list["ImageObject"], imglens: list[int]) -> list[list["ImageObject"]]:
    r"""Make nested list of images."""
    batch_images = []
    for imglen in imglens:
        batch_images.append(images[:imglen])
        images = images[imglen:]

    return batch_images


def _check_video_is_nested_images(video: "VideoInput") -> bool:
    r"""Check if the video is nested images."""
    return isinstance(video, list) and all(isinstance(frame, (str, BinaryIO, dict, ImageObject)) for frame in video)


@dataclass
class MMPluginMixin:
    image_token: str | None
    video_token: str | None
    audio_token: str | None
    expand_mm_tokens: bool = True

    def _validate_input(
        self,
        processor: Optional["MMProcessor"],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> None:
        r"""Validate if this model accepts the input modalities."""
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        video_processor: BaseImageProcessor = getattr(
            processor, "video_processor", getattr(processor, "image_processor", None)
        )
        feature_extractor: SequenceFeatureExtractor = getattr(processor, "feature_extractor", None) or getattr(
            processor, "audio_processor", None
        )
        if len(images) != 0 and self.image_token is None:
            raise ValueError(
                "This model does not support image input. Please check whether the correct `template` is used."
            )

        if len(videos) != 0 and self.video_token is None:
            raise ValueError(
                "This model does not support video input. Please check whether the correct `template` is used."
            )

        if len(audios) != 0 and self.audio_token is None:
            raise ValueError(
                "This model does not support audio input. Please check whether the correct `template` is used."
            )

        if (
            self.image_token is not None or self.video_token is not None or self.audio_token is not None
        ) and processor is None:
            raise ValueError(
                "Processor was not found. The selected `template` expects a multi-modal processor, "
                "but none could be loaded for this model. Please verify that the model files are complete "
                "(e.g. `preprocessor_config.json`), upgrade `transformers` to a version that supports this "
                "model, or choose a text-only template if you do not need multi-modal inputs. "
                "See the warning emitted by `load_tokenizer` for the underlying error."
            )

        if self.image_token is not None and image_processor is None:
            raise ValueError(
                "Image processor was not found. Please verify that the model files include the image "
                "preprocessor configuration and that your `transformers` version supports this model."
            )

        if self.video_token is not None and video_processor is None:
            raise ValueError(
                "Video processor was not found. Please verify that the model files include the video "
                "preprocessor configuration and that your `transformers` version supports this model."
            )

        if self.audio_token is not None and feature_extractor is None:
            raise ValueError(
                "Audio feature extractor was not found. Please verify that the model files include the "
                "audio feature extractor configuration and that your `transformers` version supports this model."
            )

    def _validate_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ):
        r"""Validate if the number of images, videos and audios match the number of placeholders in messages."""
        num_image_tokens, num_video_tokens, num_audio_tokens = 0, 0, 0
        for message in messages:
            num_image_tokens += message["content"].count(IMAGE_PLACEHOLDER)
            num_video_tokens += message["content"].count(VIDEO_PLACEHOLDER)
            num_audio_tokens += message["content"].count(AUDIO_PLACEHOLDER)

        if len(images) != num_image_tokens:
            raise ValueError(
                f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens in {messages}."
            )

        if len(videos) != num_video_tokens:
            raise ValueError(
                f"The number of videos does not match the number of {VIDEO_PLACEHOLDER} tokens in {messages}."
            )

        if len(audios) != num_audio_tokens:
            raise ValueError(
                f"The number of audios does not match the number of {AUDIO_PLACEHOLDER} tokens in {messages}."
            )

    def _preprocess_image(
        self, image: "ImageObject", image_max_pixels: int, image_min_pixels: int, **kwargs
    ) -> "ImageObject":
        r"""Pre-process a single image."""
        if (image.width * image.height) > image_max_pixels:
            resize_factor = math.sqrt(image_max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < image_min_pixels:
            resize_factor = math.sqrt(image_min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def _get_video_sample_indices(
        self, video_stream: "Stream", video_fps: float, video_maxlen: int, **kwargs
    ) -> list[int]:
        r"""Compute video sample indices according to fps."""
        total_frames = video_stream.frames
        if total_frames == 0:  # infinite video
            return np.linspace(0, video_maxlen - 1, video_maxlen).astype(np.int32)

        sample_frames = max(1, math.floor(float(video_stream.duration * video_stream.time_base) * video_fps))
        sample_frames = min(total_frames, video_maxlen, sample_frames)
        return np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)

    def _get_video_token_metadata(
        self,
        videos: list["VideoInput"],
        processor: "MMProcessor",
    ) -> Optional[dict[str, Any]]:
        r"""Build metadata used to expand video tokens without decoding frames."""
        return None

    def _regularize_images(self, images: list["ImageInput"], **kwargs) -> "RegularizedImageOutput":
        r"""Regularize images to avoid error. Including reading and pre-processing."""
        results = []
        for image in images:
            if isinstance(image, (str, BinaryIO)):
                image = Image.open(image)
            elif isinstance(image, bytes):
                image = Image.open(BytesIO(image))
            elif isinstance(image, dict):
                if image["bytes"] is not None:
                    image = Image.open(BytesIO(image["bytes"]))
                else:
                    image = Image.open(image["path"])

            if not isinstance(image, ImageObject):
                raise ValueError(f"Expect input is a list of images, but got {type(image)}.")

            results.append(self._preprocess_image(image, **kwargs))

        return {"images": results}

    def _regularize_videos(self, videos: list["VideoInput"], **kwargs) -> "RegularizedVideoOutput":
        r"""Regularizes videos to avoid error. Including reading, resizing and converting."""
        results = []
        durations = []
        for video in videos:
            frames: list[ImageObject] = []
            if _check_video_is_nested_images(video):
                for frame in video:
                    if not is_valid_image(frame) and not isinstance(frame, dict) and not os.path.exists(frame):
                        raise ValueError("Invalid image found in video frames.")
                frames = video
                durations.append(len(frames) / kwargs.get("video_fps", 2.0))
            else:
                container = av.open(video, "r")
                video_stream = next(stream for stream in container.streams if stream.type == "video")
                sample_indices = self._get_video_sample_indices(video_stream, **kwargs)
                container.seek(0)
                for frame_idx, frame in enumerate(container.decode(video_stream)):
                    if frame_idx in sample_indices:
                        frames.append(frame.to_image())

                if video_stream.duration is None:
                    durations.append(len(frames) / kwargs.get("video_fps", 2.0))
                else:
                    durations.append(float(video_stream.duration * video_stream.time_base))

            frames = self._regularize_images(frames, **kwargs)["images"]
            results.append(frames)

        return {"videos": results, "durations": durations}

    def _regularize_audios(
        self, audios: list["AudioInput"], sampling_rate: float, **kwargs
    ) -> "RegularizedAudioOutput":
        r"""Regularizes audios to avoid error. Including reading and resampling."""
        results, sampling_rates = [], []
        for audio in audios:
            if not isinstance(audio, np.ndarray):
                audio, sr = torchaudio.load(audio)
                if audio.shape[0] > 1:
                    audio = audio.mean(dim=0, keepdim=True)

                if sr != sampling_rate:
                    audio = torchaudio.functional.resample(audio, sr, sampling_rate)

                audio = audio.squeeze(0).numpy()

            results.append(audio)
            sampling_rates.append(sampling_rate)

        return {"audios": results, "sampling_rates": sampling_rates}

    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "MMProcessor",
        imglens: list[int] | None = None,
    ) -> dict[str, "torch.Tensor"]:
        r"""Process visual inputs.

        Returns: (llava and paligemma)
            pixel_values: tensor with shape (B, C, H, W)

        Returns: (qwen2-vl)
            pixel_values: tensor with shape (num_patches, patch_dim)
            image_grid_thw: tensor with shape (num_images, 3), where the three numbers are time, width, height
                            where num_patches == torch.prod(image_grid_thw)

        Returns: (mllama)
            pixel_values: tensor with shape
                          (batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width)
                          For example, (2, 1, 4, 3, 560, 560).
            aspect_ratio_ids: tensor with shape (batch_size, max_num_images). For example, (2, 1).
            aspect_ratio_mask: tensor with shape (batch_size, max_num_images, max_image_tiles). For example, (2, 1, 4).
            num_tiles: List[List[int]] with shape (batch_size, num_images_in_batch). For example, (2, 1).

        """
        mm_inputs = {}
        if len(images) != 0:
            image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            if imglens is not None:  # if imglens are provided, make batched images
                images = _make_batched_images(images, imglens)

            image_processor_kwargs = {}
            if getattr(processor, "image_do_pan_and_scan", False):  # gemma3 image processor
                image_processor_kwargs.update(
                    {
                        "do_pan_and_scan": True,
                        "pan_and_scan_min_crop_size": 256,
                        "pan_and_scan_max_num_crops": 4,
                        "pan_and_scan_min_ratio_to_activate": 1.2,
                    }
                )

            mm_inputs.update(image_processor(images, return_tensors="pt", **image_processor_kwargs))

        if len(videos) != 0:
            video_processor: BaseImageProcessor = getattr(
                processor, "video_processor", getattr(processor, "image_processor", None)
            )
            videos = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )["videos"]
            if "videos" in inspect.signature(video_processor.preprocess).parameters:  # for qwen2_vl and video_llava
                mm_inputs.update(video_processor(images=None, videos=videos, return_tensors="pt"))
            else:  # for llava_next_video
                mm_inputs.update(video_processor(videos, return_tensors="pt"))

        if len(audios) != 0:
            feature_extractor: SequenceFeatureExtractor = getattr(processor, "feature_extractor", None) or getattr(
                processor, "audio_processor", None
            )
            audios = self._regularize_audios(
                audios,
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            )["audios"]
            mm_inputs.update(
                feature_extractor(
                    audios,
                    sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
                    return_attention_mask=True,
                    padding="max_length",
                    return_tensors="pt",
                )
            )
            mm_inputs["feature_attention_mask"] = mm_inputs.pop("attention_mask", None)  # prevent conflicts

        return mm_inputs


@dataclass
class BasePlugin(MMPluginMixin):
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        r"""Pre-process input messages before tokenization for VLMs."""
        self._validate_input(processor, images, videos, audios)
        return messages

    def process_token_ids(
        self,
        input_ids: list[int],
        labels: list[int] | None,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["MMProcessor"],
    ) -> tuple[list[int], list[int] | None]:
        r"""Pre-process token ids after tokenization for VLMs."""
        self._validate_input(processor, images, videos, audios)
        return input_ids, labels

    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        r"""Build batched multimodal inputs for VLMs.

        Arguments:
            images: a list of image inputs, shape (num_images,)
            videos: a list of video inputs, shape (num_videos,)
            audios: a list of audio inputs, shape (num_audios,)
            imglens: number of images in each sample, shape (batch_size,)
            vidlens: number of videos in each sample, shape (batch_size,)
            audlens: number of audios in each sample, shape (batch_size,)
            batch_ids: token ids of input samples, shape (batch_size, seq_len)
            processor: a processor for pre-processing images and videos

        """
        self._validate_input(processor, images, videos, audios)
        return self._get_mm_inputs(images, videos, audios, processor)


@dataclass
class MossVLPlugin(BasePlugin):
    vision_bos_token: str = "<|vision_start|>"
    vision_eos_token: str = "<|vision_end|>"
    time_bos_token: str = "<|time_start|>"
    time_eos_token: str = "<|time_end|>"

    @staticmethod
    def _split_pixel_values(
        pixel_values: "torch.Tensor",
        grid_thw: "torch.Tensor",
    ) -> list["torch.Tensor"]:
        patch_counts = [int(grid.prod().item()) for grid in grid_thw]
        return list(torch.split(pixel_values, patch_counts))

    @staticmethod
    def _create_cross_attention_mask(
        input_ids: Union[list[list[int]], "torch.Tensor"],
        grid_thw: "torch.Tensor",
        media_nums_per_sample: list[int],
        image_token_id: int,
        attention_mask: Optional["torch.Tensor"] = None,
        padding_side: Literal["left", "right"] = "right",
    ) -> "torch.Tensor":
        r"""Create the native MOSS-VL frame-level causal cross-attention mask."""
        if isinstance(input_ids, list):
            max_text_len = max(len(token_ids) for token_ids in input_ids)
            input_ids_tensor = torch.full((len(input_ids), max_text_len), -1, dtype=torch.long)
            attention_mask_tensor = torch.zeros_like(input_ids_tensor, dtype=torch.bool)
            for batch_index, token_ids in enumerate(input_ids):
                seq_len = len(token_ids)
                start = max_text_len - seq_len if padding_side == "left" else 0
                input_ids_tensor[batch_index, start : start + seq_len] = torch.tensor(token_ids, dtype=torch.long)
                attention_mask_tensor[batch_index, start : start + seq_len] = True
        else:
            input_ids_tensor = input_ids
            attention_mask_tensor = (
                torch.ones_like(input_ids_tensor, dtype=torch.bool)
                if attention_mask is None
                else attention_mask.bool()
            )

        total_frames_per_sample = []
        media_index = 0
        for num_media in media_nums_per_sample:
            sample_grid = grid_thw[media_index : media_index + num_media]
            total_frames_per_sample.append(int(sample_grid[:, 0].sum().item()))
            media_index += num_media

        max_num_frames = max(total_frames_per_sample)
        frame_indices = torch.arange(max_num_frames, device=input_ids_tensor.device).view(1, 1, -1)
        visible_mask = (input_ids_tensor == image_token_id).cumsum(dim=1).unsqueeze(-1) > frame_indices
        visible_mask &= attention_mask_tensor.unsqueeze(-1)
        valid_frames = frame_indices < torch.tensor(
            total_frames_per_sample,
            device=input_ids_tensor.device,
        ).view(-1, 1, 1)
        visible_mask &= valid_frames
        return (~visible_mask).unsqueeze(1)

    def _get_video_inputs(
        self,
        videos: list["VideoInput"],
        processor: "MMProcessor",
        return_metadata: bool,
    ) -> dict[str, Any]:
        video_kwargs = {"return_tensors": "pt", "return_metadata": return_metadata}
        if getattr(processor, "video_fps", None) is not None:
            video_kwargs["video_fps"] = processor.video_fps
        if getattr(processor, "video_maxlen", None) is not None:
            video_kwargs["max_frames"] = processor.video_maxlen

        video_min_pixels = getattr(processor, "video_min_pixels", None)
        video_max_pixels = getattr(processor, "video_max_pixels", None)
        if video_min_pixels is not None and video_max_pixels is not None:
            video_kwargs["size"] = {
                "shortest_edge": video_min_pixels,
                "longest_edge": video_max_pixels,
            }

        return dict(processor.video_processor(videos=videos, **video_kwargs))

    def _get_media_order_from_ids(
        self,
        input_ids: list[int],
        processor: "MMProcessor",
        num_images: int,
        num_videos: int,
        expected_video_frames: Optional[list[int]] = None,
    ) -> list[str]:
        media_order = []
        video_frame_counts = []
        in_video = False
        current_video_frames = 0
        for token_id in input_ids:
            if token_id == processor.vision_start_token_id:
                if in_video:
                    raise ValueError(
                        "MOSS-VL encountered nested video token blocks after tokenization. "
                        "Please increase `cutoff_len` if a video placeholder was truncated."
                    )

                media_order.append("video")
                in_video = True
                current_video_frames = 0
            elif token_id == processor.vision_end_token_id:
                if not in_video:
                    raise ValueError(
                        "MOSS-VL encountered a video end token without a matching start token after tokenization. "
                        "Please increase `cutoff_len` if a video placeholder was truncated."
                    )

                video_frame_counts.append(current_video_frames)
                in_video = False
            elif token_id == processor.image_token_id:
                if in_video:
                    current_video_frames += 1
                else:
                    media_order.append("image")

        if in_video:
            raise ValueError(
                "MOSS-VL encountered an incomplete video token block after tokenization. "
                "Please increase `cutoff_len` or reduce `video_maxlen`."
            )

        if media_order.count("image") != num_images or media_order.count("video") != num_videos:
            raise ValueError(
                "MOSS-VL media tokens do not match the provided media after tokenization: "
                f"order={media_order}, images={num_images}, videos={num_videos}. "
                "Please increase `cutoff_len` if a visual placeholder was truncated."
            )

        if expected_video_frames is not None and video_frame_counts != expected_video_frames:
            raise ValueError(
                "MOSS-VL video frame tokens do not match the processed video after tokenization: "
                f"tokens={video_frame_counts}, frames={expected_video_frames}. "
                "Please increase `cutoff_len` or reduce `video_maxlen`."
            )

        return media_order

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)

        video_inputs = self._get_video_inputs(videos, processor, return_metadata=True) if videos else {}
        video_grid_thw = video_inputs.get("video_grid_thw", [])
        video_metadata = video_inputs.get("video_metadata", [])
        video_index = 0
        for message in messages:
            content = message["content"]
            content = content.replace(IMAGE_PLACEHOLDER, self.image_token)
            while VIDEO_PLACEHOLDER in content:
                metadata = video_metadata[video_index]
                if metadata.fps is None:
                    metadata.fps = 24

                timestamps = processor._calculate_timestamps(
                    metadata.frames_indices,
                    metadata.total_num_frames,
                    metadata.fps,
                    metadata.duration,
                    processor.video_processor.temporal_patch_size,
                    actual_timestamps=getattr(metadata, "actual_timestamps", None),
                )
                num_frames = int(video_grid_thw[video_index][0].item())
                frame_tokens = [
                    f"{self.time_bos_token}{timestamps[frame_idx]:.1f} seconds{self.time_eos_token}{self.image_token}"
                    for frame_idx in range(num_frames)
                ]
                video_tokens = f"{self.vision_bos_token}{''.join(frame_tokens)}{self.vision_eos_token}"
                content = content.replace(VIDEO_PLACEHOLDER, video_tokens, 1)
                video_index += 1

            message["content"] = content

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        if audios:
            raise ValueError("MOSS-VL does not support audio inputs.")

        if not (len(imglens) == len(vidlens) == len(batch_ids)):
            raise ValueError("MOSS-VL batch metadata must have one entry per sample.")
        final_pixel_values = []
        final_grid_thw = []
        media_nums_per_sample = []
        image_offset = 0
        video_offset = 0
        for imglen, vidlen, input_ids in zip(imglens, vidlens, batch_ids):
            sample_images = images[image_offset : image_offset + imglen]
            sample_videos = videos[video_offset : video_offset + vidlen]
            image_offset += imglen
            video_offset += vidlen
            image_chunks, image_grids = [], []
            if sample_images:
                regularized_images = self._regularize_images(
                    sample_images,
                    image_max_pixels=2**63 - 1,
                    image_min_pixels=1,
                )["images"]
                image_kwargs = {"return_tensors": "pt"}
                if getattr(processor, "image_min_pixels", None) is not None:
                    image_kwargs["min_pixels"] = processor.image_min_pixels
                if getattr(processor, "image_max_pixels", None) is not None:
                    image_kwargs["max_pixels"] = processor.image_max_pixels

                image_inputs = processor.image_processor(images=regularized_images, **image_kwargs)
                image_grids = list(image_inputs["image_grid_thw"])
                image_chunks = self._split_pixel_values(image_inputs["pixel_values"], image_inputs["image_grid_thw"])

            video_chunks, video_grids = [], []
            if sample_videos:
                video_inputs = self._get_video_inputs(sample_videos, processor, return_metadata=False)
                video_grids = list(video_inputs["video_grid_thw"])
                video_chunks = self._split_pixel_values(
                    video_inputs["pixel_values_videos"],
                    video_inputs["video_grid_thw"],
                )

            media_order = self._get_media_order_from_ids(
                input_ids,
                processor,
                imglen,
                vidlen,
                expected_video_frames=[int(grid[0].item()) for grid in video_grids],
            )

            if not media_order:
                patch_size = getattr(processor.image_processor, "patch_size", None)
                if patch_size is None:  # lightweight/test processors without the native MOSS-VL contract
                    blank_image = Image.new("RGB", (128, 128), (255, 255, 255))
                    blank_inputs = processor.image_processor(images=[blank_image], return_tensors="pt")
                    final_pixel_values.append(blank_inputs["pixel_values"])
                    final_grid_thw.append(blank_inputs["image_grid_thw"][0])
                else:
                    temporal_patch_size = getattr(processor.image_processor, "temporal_patch_size", None) or 1
                    merge_size = getattr(processor.image_processor, "merge_size", None) or 2
                    factor = patch_size * merge_size
                    side = math.ceil(128 / factor) * factor
                    grid_thw = torch.tensor([1, side // patch_size, side // patch_size])
                    feature_dim = 3 * temporal_patch_size * patch_size * patch_size
                    final_pixel_values.append(torch.zeros((int(grid_thw.prod()), feature_dim), dtype=torch.float32))
                    final_grid_thw.append(grid_thw)
                media_nums_per_sample.append(1)
                continue

            image_index = 0
            video_index = 0
            for modality in media_order:
                if modality == "image":
                    final_pixel_values.append(image_chunks[image_index])
                    final_grid_thw.append(image_grids[image_index])
                    image_index += 1
                else:
                    final_pixel_values.append(video_chunks[video_index])
                    final_grid_thw.append(video_grids[video_index])
                    video_index += 1

            media_nums_per_sample.append(len(media_order))

        if image_offset != len(images) or video_offset != len(videos):
            raise ValueError("MOSS-VL media lengths do not consume all provided inputs.")

        mm_inputs = {
            "pixel_values": torch.cat(final_pixel_values, dim=0),
            "grid_thw": torch.stack(final_grid_thw),
            "media_nums_per_sample": media_nums_per_sample,
        }
        mm_inputs["cross_attention_mask"] = self._create_cross_attention_mask(
            batch_ids,
            mm_inputs["grid_thw"],
            media_nums_per_sample,
            processor.image_token_id,
            padding_side=processor.tokenizer.padding_side,
        )
        return mm_inputs

    def post_process_mossvl_inputs(
        self,
        features: dict[str, "torch.Tensor"],
        mm_inputs: dict[str, Any],
        processor: "MMProcessor",
    ) -> dict[str, Any]:
        r"""Create MOSS-VL batch-only inputs after the text batch has been padded."""
        input_ids = features["input_ids"]
        attention_mask = features["attention_mask"].bool()
        mm_inputs["cross_attention_mask"] = self._create_cross_attention_mask(
            input_ids,
            mm_inputs["grid_thw"],
            mm_inputs["media_nums_per_sample"],
            processor.image_token_id,
            attention_mask,
        )

        dummy_image_tokens = (input_ids == processor.image_token_id) & ~attention_mask
        input_ids.masked_fill_(dummy_image_tokens, processor.tokenizer.pad_token_id)

        labels = features.get("labels")
        if labels is not None:
            control_token_ids = {
                processor.image_token_id,
                processor.video_token_id,
                processor.vision_start_token_id,
                processor.vision_end_token_id,
                processor.tokenizer.convert_tokens_to_ids(self.time_bos_token),
                processor.tokenizer.convert_tokens_to_ids(self.time_eos_token),
            }
            for batch_index, token_ids in enumerate(input_ids):
                in_vision = False
                for token_index, token_id in enumerate(token_ids.tolist()):
                    if token_id == processor.vision_start_token_id:
                        in_vision = True
                    if in_vision or token_id in control_token_ids:
                        labels[batch_index, token_index] = IGNORE_INDEX
                    if token_id == processor.vision_end_token_id:
                        in_vision = False

            # Native MOSS-VL labels_spans supervise through <|im_end|>, but not its trailing newline.
            im_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
            labels[:, 1:].masked_fill_(input_ids[:, :-1] == im_end_token_id, IGNORE_INDEX)

        features.pop("position_ids", None)
        return mm_inputs


@dataclass
class ErnieVLPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)

        image_processor: BaseImageProcessor = getattr(processor, "image_processor")

        merge_length: int = getattr(image_processor, "merge_size") ** 2
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
        else:
            image_grid_thw = [None] * len(images)
            video_grid_thw = [None] * len(videos)

        image_idx, video_idx = 0, 0
        for message in messages:
            content = message["content"]
            image_token = self.image_token or "<|IMAGE_PLACEHOLDER|>"
            video_token = self.video_token or "<|VIDEO_PLACEHOLDER|>"
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = image_grid_thw[image_idx].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"Picture {image_idx + 1}:<|IMAGE_START|>{image_token * image_seqlen}<|IMAGE_END|>",
                    1,
                )
                image_idx += 1
            while VIDEO_PLACEHOLDER in content:
                video_seqlen = video_grid_thw[video_idx].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    VIDEO_PLACEHOLDER,
                    f"Video {video_idx + 1}:<|VIDEO_START|>{video_token * video_seqlen}<|VIDEO_END|>",
                    1,
                )
                video_idx += 1
            message["content"] = content
        return messages


@dataclass
class Gemma3Plugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens = 0
        messages = deepcopy(messages)
        boi_token: str = getattr(processor, "boi_token")
        full_image_sequence: str = getattr(processor, "full_image_sequence")
        image_str = full_image_sequence if self.expand_mm_tokens else boi_token

        do_pan_and_scan: bool = getattr(processor, "image_do_pan_and_scan", False)
        if do_pan_and_scan:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if do_pan_and_scan:
                    image_placeholder_str = (
                        "Here is the original image {{image}} and here are some crops to help you see better "
                        + " ".join(["{{image}}"] * mm_inputs["num_crops"][0][num_image_tokens])
                    )
                else:
                    image_placeholder_str = "{{image}}"

                content = content.replace(IMAGE_PLACEHOLDER, image_placeholder_str, 1)
                num_image_tokens += 1

            message["content"] = content.replace("{{image}}", image_str)

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        mm_inputs.pop("num_crops", None)
        mm_inputs["token_type_ids"] = _get_gemma3_token_type_ids(batch_ids, processor)
        return mm_inputs


class Gemma3nPlugin(Gemma3Plugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)
        boi_token: str = getattr(processor, "boi_token")
        boa_token: str = getattr(processor, "boa_token")
        full_image_sequence: str = getattr(processor, "full_image_sequence")
        full_audio_sequence: str = getattr(processor, "full_audio_sequence")
        image_str = full_image_sequence if self.expand_mm_tokens else boi_token
        audio_str = full_audio_sequence if self.expand_mm_tokens else boa_token

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(IMAGE_PLACEHOLDER, image_str, 1)

            while AUDIO_PLACEHOLDER in content:
                content = content.replace(AUDIO_PLACEHOLDER, audio_str, 1)

            message["content"] = content

        return messages


@dataclass
class Gemma4Plugin(BasePlugin):
    r"""Plugin for the Gemma4 multimodal model."""

    @override
    def _regularize_videos(self, videos: list["VideoInput"], **kwargs) -> "RegularizedVideoOutput":
        r"""Regularize videos, also tracking per-video FPS and frame indices for timestamp generation."""
        results, fps_per_video, durations, frames_indices = [], [], [], []
        for video in videos:
            frames: list[ImageObject] = []
            if _check_video_is_nested_images(video):
                frames = video
                fps_per_video.append(kwargs.get("video_fps", 2.0))
                durations.append(len(frames) / kwargs.get("video_fps", 2.0))
                frames_indices.append(list(range(len(frames))))
            else:
                container = av.open(video, "r")
                video_stream = next(stream for stream in container.streams if stream.type == "video")
                sample_indices = self._get_video_sample_indices(video_stream, **kwargs)
                original_fps = float(video_stream.average_rate)
                # for correctly calculate timestamps
                frames_indices.append([idx / original_fps * kwargs.get("video_fps", 2.0) for idx in sample_indices])
                container.seek(0)
                for frame_idx, frame in enumerate(container.decode(video_stream)):
                    if frame_idx in sample_indices:
                        frames.append(frame.to_image())

                if video_stream.duration is None:
                    durations.append(len(frames) / kwargs.get("video_fps", 2.0))
                else:
                    durations.append(float(video_stream.duration * video_stream.time_base))

            frames = self._regularize_images(frames, **kwargs)["images"]
            results.append(frames)

        return {
            "videos": results,
            "fps_per_video": fps_per_video,
            "durations": durations,
            "frames_indices": frames_indices,
        }

    @override
    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "MMProcessor",
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        image_processor = getattr(processor, "image_processor", None)
        video_processor = getattr(processor, "video_processor", None)
        feature_extractor = getattr(processor, "feature_extractor", None)
        mm_inputs = {}

        if len(images) != 0:
            regularized = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            mm_inputs.update(image_processor(regularized, return_tensors="pt"))

        if len(videos) != 0:
            video_data = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )
            video_metadata = [
                {
                    "fps": getattr(processor, "video_fps", 2.0),
                    "duration": duration,
                    "total_num_frames": len(video),
                    "frames_indices": sample_indices,
                }
                for video, duration, sample_indices in zip(
                    video_data["videos"], video_data["durations"], video_data["frames_indices"]
                )
            ]
            mm_inputs.update(
                video_processor(
                    videos=video_data["videos"],
                    video_metadata=video_metadata,
                    return_tensors="pt",
                    return_metadata=True,
                    do_sample_frames=False,
                )
            )

        if len(audios) != 0:  # only for gemma4n
            audios = self._regularize_audios(
                audios,
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            )["audios"]

            mm_inputs.update(
                feature_extractor(
                    audios,
                    padding="max_length",
                    return_tensors="pt",
                )
            )

        return mm_inputs

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)

        boi_token: str = getattr(processor, "boi_token")
        eoi_token: str = getattr(processor, "eoi_token")
        boa_token: str = getattr(processor, "boa_token")
        eoa_token: str = getattr(processor, "eoa_token")
        image_token: str = getattr(processor, "image_token")
        video_token: str = getattr(processor, "video_token")
        audio_token: str = getattr(processor, "audio_token")

        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            num_image_soft_tokens: list[int] = list(
                mm_inputs.get("num_soft_tokens_per_image", [getattr(processor, "image_seq_length", 256)] * len(images))
            )
            num_video_soft_tokens: list[int] = list(mm_inputs.get("num_soft_tokens_per_video", [1] * len(videos)))
            video_metadata = mm_inputs.get("video_metadata", [])
        else:
            num_image_soft_tokens = [1] * len(images)
            num_video_soft_tokens = [1] * len(videos)
            video_metadata = [None] * len(videos)

        audio_iter = iter(audios)
        image_iter = iter(num_image_soft_tokens)
        video_iter = iter(zip(num_video_soft_tokens, video_metadata))

        for message in messages:
            content = message["content"]

            while IMAGE_PLACEHOLDER in content:
                n = next(image_iter)
                content = content.replace(IMAGE_PLACEHOLDER, f"{boi_token}{image_token * n}{eoi_token}", 1)

            while VIDEO_PLACEHOLDER in content:
                num_soft_tokens_per_frame, metadata = next(video_iter)
                if self.expand_mm_tokens:
                    timestamp_strs = [f"{int(t // 60):02d}:{int(t % 60):02d}" for t in metadata.timestamps]
                    frame_strs = [
                        f"{ts} {boi_token}{video_token * num_soft_tokens_per_frame}{eoi_token}"
                        for ts in timestamp_strs
                    ]
                    video_str = " ".join(frame_strs)
                else:
                    video_str = f"{boi_token}{video_token * num_soft_tokens_per_frame}{eoi_token}"
                content = content.replace(VIDEO_PLACEHOLDER, video_str, 1)

            while AUDIO_PLACEHOLDER in content:
                current_audio = next(audio_iter)
                if self.expand_mm_tokens:
                    num_audio_tokens = processor._compute_audio_num_tokens(
                        current_audio, processor.feature_extractor.sampling_rate
                    )
                    audio_str = f"{boa_token}{audio_token * num_audio_tokens}{eoa_token}"
                else:
                    audio_str = f"{boa_token}{audio_token}{eoa_token}"

                content = content.replace(AUDIO_PLACEHOLDER, audio_str, 1)

            message["content"] = content

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        # Pop metadata keys that must not be passed to the model.
        for key in (
            "num_soft_tokens_per_image",
            "num_soft_tokens_per_video",
            "video_metadata",
            "_gemma4_fps_per_video",
            "_gemma4_frames_indices",
            "_gemma4_num_audio_soft_tokens",
        ):
            mm_inputs.pop(key, None)

        mm_inputs["mm_token_type_ids"] = processor.create_mm_token_type_ids(batch_ids)

        return mm_inputs


@dataclass
class InternVLPlugin(BasePlugin):
    @override
    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "ProcessorMixin",
        **kwargs,
    ) -> dict[str, "torch.Tensor"]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")
        image_processor_kwargs = {}
        if getattr(processor, "crop_to_patches", False):
            image_processor_kwargs.update(
                {
                    "crop_to_patches": True,
                    "max_patches": 12,
                    "min_patches": 1,
                }
            )

        mm_inputs = {}
        image_video_patches = []

        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 1024 * 1024),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]

        if len(videos) != 0:
            videos = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )["videos"]

        if len(images) != 0:
            images = make_flat_list_of_images(images)
            image_inputs = image_processor(images=images, return_tensors="pt", **image_processor_kwargs)
            image_num_patches = image_inputs.pop("num_patches")
            image_pixel_values = image_inputs.pop("pixel_values")
            image_num_patches_indices = np.cumsum(image_num_patches)

        if len(videos) != 0:
            videos = make_batched_videos(videos)
            num_frames_per_video = [len(video) for video in videos]
            patch_indices = np.cumsum(num_frames_per_video)
            image_processor_kwargs["crop_to_patches"] = False
            video_inputs = image_processor(images=videos, return_tensors="pt", **image_processor_kwargs)
            video_num_patches = video_inputs.pop("num_patches")
            video_pixel_values = video_inputs.pop("pixel_values")
            video_num_patches_indices = np.cumsum(video_num_patches)

        # NOT SUPPORT IMAGE VIDEO INTERLEAVED
        if len(images) != 0 and image_pixel_values is not None:
            for i in range(len(images)):
                start_index = image_num_patches_indices[i - 1] if i > 0 else 0
                end_index = image_num_patches_indices[i]
                image_video_patches.append(image_pixel_values[start_index:end_index])

        if len(videos) != 0 and video_pixel_values is not None:
            patch_indices_with_prefix = [0] + list(patch_indices)
            for i in range(len(videos)):
                current_patch_index = patch_indices_with_prefix[i]
                end_patch_index = patch_indices_with_prefix[i + 1]
                start_index = video_num_patches_indices[current_patch_index - 1] if i > 0 else 0
                end_index = video_num_patches_indices[end_patch_index - 1]
                image_video_patches.append(video_pixel_values[start_index:end_index])

        if len(images) != 0 or len(videos) != 0:
            mm_inputs["pixel_values"] = torch.cat(image_video_patches, dim=0)

        if len(images) != 0:
            mm_inputs.update({"image_num_patches": image_num_patches})

        if len(videos) != 0:
            mm_inputs.update({"video_patch_indices": patch_indices})
            mm_inputs.update({"video_num_patches": video_num_patches})

        return mm_inputs

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["ProcessorMixin"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens = 0, 0
        image_seqlen = getattr(processor, "image_seq_length") if self.expand_mm_tokens else 1
        messages = deepcopy(messages)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)

        image_pixel_patch_list = mm_inputs.get("image_num_patches")  # patches of images
        video_num_patches = mm_inputs.get("video_num_patches")  # all patches for frames of videos
        video_patch_indices = mm_inputs.get("video_patch_indices")  # num frames of per video

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"<img>{'<IMG_CONTEXT>' * image_seqlen * image_pixel_patch_list[num_image_tokens]}</img>",
                    1,
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                current_patch_index = video_patch_indices[num_video_tokens - 1] if num_video_tokens > 0 else 0
                end_patch_index = video_patch_indices[num_video_tokens]
                num_patches = list(video_num_patches[current_patch_index:end_patch_index])
                video_replaced_prompt = "\n".join(
                    f"Frame{i + 1}: <img>{'<IMG_CONTEXT>' * image_seqlen * num_patches[i]}</img>"
                    for i in range(len(num_patches))
                )
                content = content.replace(VIDEO_PLACEHOLDER, video_replaced_prompt, 1)
                num_video_tokens += 1

            message["content"] = content

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["ProcessorMixin"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        mm_inputs.pop("image_num_patches", None)
        mm_inputs.pop("video_patch_indices", None)
        mm_inputs.pop("video_num_patches", None)
        return mm_inputs


class KimiVLPlugin(BasePlugin):
    @override
    def process_messages(self, messages, images, videos, audios, processor):
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            image_grid_hws = mm_inputs.get("image_grid_hws", [])
        else:
            image_grid_hws = [None] * len(images)

        num_image_tokens = 0
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")
        merge_length = math.prod(image_processor.merge_kernel_size)
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = image_grid_hws[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"<|media_start|>image<|media_content|>{self.image_token * image_seqlen}<|media_end|>",
                    1,
                )
                num_image_tokens += 1

            message["content"] = content

        return messages


@dataclass
class Llama4Plugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if "pixel_values" in mm_inputs:
                image_height, image_width = mm_inputs["pixel_values"][0].shape[-2:]
                num_patches_per_chunk = int(
                    (image_height // processor.patch_size)
                    * (image_width // processor.patch_size)
                    // processor.downsample_ratio
                )
                aspect_ratios = mm_inputs.pop("aspect_ratios")

        num_image_tokens = 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            if self.expand_mm_tokens:
                placeholder_count = content.count(IMAGE_PLACEHOLDER)
                prompt_splits = content.split(IMAGE_PLACEHOLDER)
                new_content = []
                for local_image_index, split_part in enumerate(prompt_splits):
                    new_content.append(split_part)
                    if local_image_index < placeholder_count:
                        tokens_for_this_image = processor._prompt_split_image(
                            aspect_ratios[num_image_tokens], num_patches_per_chunk
                        )
                        num_image_tokens += 1
                        new_content.append(tokens_for_this_image)

                content = "".join(new_content)
            else:
                content = content.replace(IMAGE_PLACEHOLDER, self.image_token)

            message["content"] = content

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        mm_inputs.pop("aspect_ratios", None)
        return mm_inputs


@dataclass
class LlavaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if "pixel_values" in mm_inputs:
                height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values"][0]))
                image_seqlen = (height // processor.patch_size) * (
                    width // processor.patch_size
                ) + processor.num_additional_image_tokens
                if processor.vision_feature_select_strategy == "default":
                    image_seqlen -= 1
        else:
            image_seqlen = 1

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)

            message["content"] = content.replace("{{image}}", self.image_token)

        return messages


@dataclass
class LlavaNextPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens = 0
        messages = deepcopy(messages)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if "pixel_values" in mm_inputs:
                image_sizes = iter(mm_inputs["image_sizes"].tolist())
                height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values"][0][0]))

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if self.expand_mm_tokens:
                    orig_height, orig_width = next(image_sizes)
                    image_seqlen = processor._get_number_of_features(orig_height, orig_width, height, width)
                    if processor.vision_feature_select_strategy == "default":
                        image_seqlen -= 1
                else:
                    image_seqlen = 1

                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)
                num_image_tokens += 1

            message["content"] = content.replace("{{image}}", self.image_token)

        return messages


@dataclass
class LlavaNextVideoPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if "pixel_values" in mm_inputs:
                image_sizes = iter(mm_inputs["image_sizes"].tolist())
                height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values"][0][0]))

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if self.expand_mm_tokens:
                    orig_height, orig_width = next(image_sizes)
                    image_seqlen = processor._get_number_of_features(orig_height, orig_width, height, width)
                    if processor.vision_feature_select_strategy == "default":
                        image_seqlen -= 1
                else:
                    image_seqlen = 1

                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)

            message["content"] = content.replace("{{image}}", self.image_token)

        if self.expand_mm_tokens:
            if "pixel_values_videos" in mm_inputs:
                one_video = to_numpy_array(mm_inputs.get("pixel_values_videos")[0])
                height, width = get_image_size(one_video[0])
                num_frames = one_video.shape[0]  # frame dim is always after batch dim
                image_seqlen = (height // processor.patch_size) * (width // processor.patch_size)
                video_seqlen = image_seqlen // 4 * num_frames  # divide by 4 needed for avg pooling layer
        else:
            video_seqlen = 1

        for message in messages:
            content = message["content"]
            while VIDEO_PLACEHOLDER in content:
                content = content.replace(VIDEO_PLACEHOLDER, "{{video}}" * video_seqlen, 1)

            message["content"] = content.replace("{{video}}", self.video_token)

        return messages


@dataclass
class MiniCPMVPlugin(BasePlugin):
    @override
    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "MMProcessor",
        **kwargs,
    ) -> dict[str, "torch.Tensor"]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            if "valid_image_nums_ls" in kwargs:
                valid_image_nums_ls = kwargs["valid_image_nums_ls"]
                new_images = []
                idx = 0
                for valid_image_nums in valid_image_nums_ls:
                    new_images.append(images[idx : idx + valid_image_nums])
                    idx += valid_image_nums

                images = new_images

            image_inputs = image_processor(
                images, do_pad=True, max_slice_nums=image_processor.max_slice_nums, return_tensors="pt"
            )
            mm_inputs.update(image_inputs)

        if len(videos) != 0:
            videos = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )["videos"]
            video_inputs = image_processor(videos, do_pad=True, max_slice_nums=2, return_tensors="pt")
            mm_inputs.update(video_inputs)

        if len(audios) != 0:
            audios = self._regularize_audios(
                audios,
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            )["audios"]
            if "valid_audio_nums_ls" in kwargs:
                valid_audio_nums_ls = kwargs["valid_audio_nums_ls"]
                audios_ls = []
                idx = 0
                for valid_audio_nums in valid_audio_nums_ls:
                    audios_ls.append(audios[idx : idx + valid_audio_nums])
                    idx += valid_audio_nums
            else:
                audios_ls = [audios]

            audio_features, audio_feature_lens, audio_phs = processor.audio_feature_extract(
                audios_ls,
                chunk_input=True,
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            )
            audio_feature_lens = [
                x.clone().detach() if isinstance(x, torch.Tensor) else torch.tensor(x) for x in audio_feature_lens
            ]
            mm_inputs.update({"audio_features": audio_features, "audio_feature_lens": audio_feature_lens})
            if kwargs.get("ret_phs", False):
                mm_inputs.update({"audio_phs": audio_phs})

        return mm_inputs

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens, num_audio_tokens = 0, 0, 0
        messages = deepcopy(messages)
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")
        mm_inputs, audio_inputs = {}, {}
        if len(images) != 0 and len(videos) != 0:
            raise ValueError("MiniCPM-V model does not support input images and videos at the same time.")

        if len(videos) != 0:
            max_slice_nums = 2
            use_image_id = False
            mm_inputs = self._get_mm_inputs([], videos, [], processor)
        else:
            max_slice_nums = image_processor.max_slice_nums
            use_image_id = image_processor.use_image_id

        for i, message in enumerate(messages):
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}", 1)
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                video_seqlen = len(mm_inputs["image_sizes"][num_video_tokens]) if self.expand_mm_tokens else 1
                content = content.replace(VIDEO_PLACEHOLDER, "{{image}}" * video_seqlen, 1)
                num_video_tokens += 1

            while AUDIO_PLACEHOLDER in content:
                content = content.replace(AUDIO_PLACEHOLDER, "{{audio}}", 1)
                num_audio_tokens += 1

            message["content"] = content.replace("{{image}}", "(<image>./</image>)").replace(
                "{{audio}}", "(<audio>./</audio>)"
            )

        if len(images):
            mm_inputs = self._get_mm_inputs(images, [], [], processor)

        if len(audios):
            audio_inputs = self._get_mm_inputs([], [], audios, processor, ret_phs=True)

        if self.expand_mm_tokens and mm_inputs:
            pattern = "(<image>./</image>)"
            image_sizes = mm_inputs["image_sizes"]
            idx = 0
            for index, message in enumerate(messages):
                text = message["content"]
                image_tags = re.findall(pattern, text)
                text_chunks = text.split(pattern)
                final_text = ""
                for i in range(len(image_tags)):
                    final_text = (
                        final_text
                        + text_chunks[i]
                        + image_processor.get_slice_image_placeholder(
                            image_sizes[0][idx], idx, max_slice_nums, use_image_id
                        )
                    )
                    idx += 1

                final_text += text_chunks[-1]
                messages[index]["content"] = final_text

        if self.expand_mm_tokens and audio_inputs:
            pattern = "(<audio>./</audio>)"
            idx = 0
            for index, message in enumerate(messages):
                text = message["content"]
                audio_tags = re.findall(pattern, text)
                text_chunks = text.split(pattern)
                final_text = ""
                for i in range(len(audio_tags)):
                    audio_placeholder = audio_inputs["audio_phs"][0][idx]
                    final_text = final_text + text_chunks[i] + audio_placeholder
                    idx += 1

                final_text += text_chunks[-1]
                messages[index]["content"] = final_text

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        # image bound
        image_bounds_list = []
        valid_image_nums_ls = []
        for i, input_ids in enumerate(batch_ids):
            input_ids_ = torch.tensor(input_ids)
            start_cond = (input_ids_ == processor.tokenizer.im_start_id) | (
                input_ids_ == processor.tokenizer.slice_start_id
            )
            end_cond = (input_ids_ == processor.tokenizer.im_end_id) | (input_ids_ == processor.tokenizer.slice_end_id)
            image_start_tokens = torch.where(start_cond)[0]
            image_start_tokens += 1
            image_end_tokens = torch.where(end_cond)[0]
            valid_image_nums_ls.append(imglens[i])
            image_bounds = torch.hstack(
                [
                    image_start_tokens.unsqueeze(-1),
                    image_end_tokens.unsqueeze(-1),
                ]
            )
            image_bounds_list.append(image_bounds)

        mm_inputs = self._get_mm_inputs(images, videos, [], processor, valid_image_nums_ls=valid_image_nums_ls)
        if "tgt_sizes" not in mm_inputs:
            dummy_data = [torch.empty(0) for _ in range(len(batch_ids))]
            mm_inputs.update({"tgt_sizes": dummy_data, "pixel_values": dummy_data, "image_sizes": dummy_data})

        mm_inputs.update({"image_bound": image_bounds_list})

        if len(audios) > 0:
            # audio bound
            audio_bounds_ls = []
            spk_bounds_ls = []
            valid_audio_nums_ls = []

            for input_ids, audiolen in zip(batch_ids, audlens):
                input_ids_ = torch.tensor(input_ids)
                audio_start_idx = torch.where(input_ids_ == processor.tokenizer.audio_start_id)[0]
                audio_end_idx = torch.where(input_ids_ == processor.tokenizer.audio_end_id)[0]
                assert len(audio_start_idx) == len(audio_end_idx)
                audio_bounds = torch.hstack([(audio_start_idx + 1).unsqueeze(-1), audio_end_idx.unsqueeze(-1)])
                audio_bounds_ls.append(audio_bounds)
                valid_audio_nums_ls.append(audiolen)

                spk_start_idx = torch.where(input_ids_ == processor.tokenizer.spk_start_id)[0]
                spk_end_idx = torch.where(input_ids_ == processor.tokenizer.spk_end_id)[0]
                assert len(spk_start_idx) == len(spk_end_idx)
                spk_bounds = torch.hstack([(spk_start_idx + 1).unsqueeze(-1), spk_end_idx.unsqueeze(-1)])
                spk_bounds_ls.append(spk_bounds)

            audio_inputs = self._get_mm_inputs([], [], audios, processor, valid_audio_nums_ls=valid_audio_nums_ls)
            mm_inputs.update(audio_inputs)
            mm_inputs.update({"audio_bounds": audio_bounds_ls, "spk_bounds": spk_bounds_ls})

        return mm_inputs


@dataclass
class MiniCPMV4_6Plugin(BasePlugin):
    """Plugin for MiniCPM-V-4.6 with new transformers (NaViT vision + get_placeholder_mask API)."""

    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "MMProcessor",
        **kwargs,
    ) -> dict[str, "torch.Tensor"]:
        image_processor = getattr(processor, "image_processor")
        video_processor = getattr(processor, "video_processor", None)
        mm_inputs = {}

        if len(images) != 0:
            # The image_processor ignores downsample_mode; target_sizes are always based on patch_size.
            # downsample_mode only affects the token divisor in _build_v4_6_placeholder and model forward.
            mm_inputs.update(image_processor(images, return_tensors="pt"))

        if len(videos) != 0:
            if video_processor is not None:
                video_inputs = video_processor(videos, return_tensors="pt")
                mm_inputs["pixel_values_videos"] = video_inputs["pixel_values_videos"]
                mm_inputs["target_sizes_videos"] = video_inputs["target_sizes_videos"]
            else:
                video_inputs = image_processor(videos, return_tensors="pt")
                mm_inputs["pixel_values_videos"] = video_inputs["pixel_values"]
                mm_inputs["target_sizes_videos"] = video_inputs["target_sizes"]

        if len(audios) != 0:
            audio_features, audio_feature_lens, audio_phs = processor.audio_feature_extract(
                [audios],
                chunk_input=True,
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            )
            audio_feature_lens = [
                x.clone().detach() if isinstance(x, torch.Tensor) else torch.tensor(x) for x in audio_feature_lens
            ]
            mm_inputs.update({"audio_features": audio_features, "audio_feature_lens": audio_feature_lens})
            if kwargs.get("ret_phs", False):
                mm_inputs.update({"audio_phs": audio_phs})

        return mm_inputs

    def _build_v4_6_placeholder(
        self,
        image_inputs: dict[str, Any],
        image_idx: int,
        use_image_id: bool,
        processor: "MMProcessor",
    ) -> str:
        """Build image placeholder for MiniCPM-V-4.6 using NaViT token count computation."""
        grids = image_inputs.get("grids", [[0, 0]])
        num_patches_per_image = image_inputs.get("num_patches_per_image", [1])
        target_sizes = image_inputs.get("target_sizes")

        downsample_mode = os.getenv("DOWNSAMPLE_MODE")
        if downsample_mode is None:
            image_processor = getattr(processor, "image_processor")
            downsample_mode = getattr(image_processor, "downsample_mode", "16x")
        token_divisor = 4 if downsample_mode == "4x" else 16

        flat_index = 0
        for idx in range(image_idx):
            flat_index += num_patches_per_image[idx]
        n_patches = num_patches_per_image[image_idx]

        img_target_sizes = target_sizes[flat_index : flat_index + n_patches]
        num_tokens_per_patch = img_target_sizes.prod(-1) // token_divisor
        num_rows, num_cols = grids[image_idx]

        image_start = getattr(processor, "image_start_token", "<image>")
        image_end = getattr(processor, "image_end_token", "</image>")
        slice_start = getattr(processor, "slice_start_token", "<slice>")
        slice_end = getattr(processor, "slice_end_token", "</slice>")
        image_id_start = getattr(processor, "image_id_start_token", "<image_id>")
        image_id_end = getattr(processor, "image_id_end_token", "</image_id>")
        image_token = (
            getattr(processor, "image_token", None)
            or getattr(getattr(processor, "tokenizer", None), "image_token", None)
            or "<image>"
        )

        image_placeholder = image_start + "<|ph|>" * int(num_tokens_per_patch[0]) + image_end
        if use_image_id:
            image_placeholder = f"{image_id_start}{image_idx}{image_id_end}" + image_placeholder

        slice_mode = getattr(processor, "slice_mode", True)
        if slice_mode and num_rows > 0 and num_cols > 0:
            per_slice_tokens = int(num_tokens_per_patch[1]) if len(num_tokens_per_patch) > 1 else 0
            slice_placeholder = slice_start + "<|ph|>" * per_slice_tokens + slice_end
            slices = [slice_placeholder * num_cols for _ in range(num_rows)]
            image_placeholder += "\n".join(slices)

        return image_placeholder.replace("<|ph|>", image_token)

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens, num_audio_tokens = 0, 0, 0
        messages = deepcopy(messages)
        mm_inputs, audio_inputs = {}, {}
        if len(images) != 0 and len(videos) != 0:
            raise ValueError("MiniCPM-V model does not support input images and videos at the same time.")

        use_image_id = getattr(processor, "default_use_image_id", True)

        if len(videos) != 0:
            use_image_id = False
            mm_inputs = self._get_mm_inputs([], videos, [], processor)

        for i, message in enumerate(messages):
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}", 1)
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                num_frames = 1
                if "num_frames_per_video" in mm_inputs:
                    num_frames = sum(mm_inputs["num_frames_per_video"])
                content = content.replace(VIDEO_PLACEHOLDER, "{{image}}" * num_frames, 1)
                num_video_tokens += 1

            while AUDIO_PLACEHOLDER in content:
                content = content.replace(AUDIO_PLACEHOLDER, "{{audio}}", 1)
                num_audio_tokens += 1

            message["content"] = content.replace("{{image}}", "(<image>./</image>)").replace(
                "{{audio}}", "(<audio>./</audio>)"
            )

        if len(images):
            mm_inputs = self._get_mm_inputs(images, [], [], processor)

        if len(audios):
            audio_inputs = self._get_mm_inputs([], [], audios, processor, ret_phs=True)

        if self.expand_mm_tokens and mm_inputs:
            pattern = "(<image>./</image>)"
            idx = 0
            for index, message in enumerate(messages):
                text = message["content"]
                image_tags = re.findall(pattern, text)
                text_chunks = text.split(pattern)
                final_text = ""
                for i in range(len(image_tags)):
                    image_placeholder = self._build_v4_6_placeholder(mm_inputs, idx, use_image_id, processor)
                    final_text = final_text + text_chunks[i] + image_placeholder
                    idx += 1
                final_text += text_chunks[-1]
                messages[index]["content"] = final_text

        if self.expand_mm_tokens and audio_inputs:
            pattern = "(<audio>./</audio>)"
            idx = 0
            for index, message in enumerate(messages):
                text = message["content"]
                audio_tags = re.findall(pattern, text)
                text_chunks = text.split(pattern)
                final_text = ""
                for i in range(len(audio_tags)):
                    audio_placeholder = audio_inputs["audio_phs"][0][idx]
                    final_text = final_text + text_chunks[i] + audio_placeholder
                    idx += 1
                final_text += text_chunks[-1]
                messages[index]["content"] = final_text

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)

        # v4.6 does NOT use image_bound — the model finds image tokens via get_placeholder_mask
        # Ensure target_sizes key name matches the model's expected input
        if "target_sizes" not in mm_inputs and "tgt_sizes" in mm_inputs:
            mm_inputs["target_sizes"] = mm_inputs.pop("tgt_sizes")

        if "target_sizes" not in mm_inputs:
            mm_inputs["target_sizes"] = torch.empty(0, 2, dtype=torch.int32)

        if "pixel_values" not in mm_inputs:
            mm_inputs["pixel_values"] = torch.empty(1, 3, 14, 0)

        # Pass downsample_mode to model forward so it matches the placeholder divisor
        _ds = os.getenv("DOWNSAMPLE_MODE")
        if _ds is None:
            _ds = getattr(getattr(processor, "image_processor", None), "downsample_mode", "16x")
        mm_inputs["downsample_mode"] = _ds

        if len(audios) > 0:
            audio_inputs = self._get_mm_inputs([], [], audios, processor)
            mm_inputs.update(audio_inputs)

        return mm_inputs


@dataclass
class MllamaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens = 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            num_image_tokens += content.count(IMAGE_PLACEHOLDER)
            message["content"] = content.replace(IMAGE_PLACEHOLDER, self.image_token)

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor, imglens)
        if mm_inputs:
            num_tiles = mm_inputs.pop("num_tiles")
            image_token_id: int = getattr(processor, "image_token_id")
            max_image_tiles: int = getattr(processor.image_processor, "max_image_tiles")
            cross_attention_token_mask = [
                get_cross_attention_token_mask(input_ids, image_token_id) for input_ids in batch_ids
            ]
            mm_inputs["cross_attention_mask"] = torch.from_numpy(
                convert_sparse_cross_attention_mask_to_dense(
                    cross_attention_token_mask,
                    num_tiles=num_tiles,
                    max_num_tiles=max_image_tiles,
                    length=max(len(input_ids) for input_ids in batch_ids),
                )
            )  # shape: (batch_size, length, max_num_images, max_num_tiles)

        return mm_inputs


@dataclass
class PaliGemmaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens = 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(IMAGE_PLACEHOLDER, "", 1)
                num_image_tokens += 1

            message["content"] = content

        return messages

    @override
    def process_token_ids(
        self,
        input_ids: list[int],
        labels: list[int] | None,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["MMProcessor"],
    ) -> tuple[list[int], list[int] | None]:
        self._validate_input(processor, images, videos, audios)
        num_images = len(images)
        image_seqlen = processor.image_seq_length if self.expand_mm_tokens else 0  # skip mm token
        image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        input_ids = [image_token_id] * num_images * image_seqlen + input_ids
        if labels is not None:
            labels = [IGNORE_INDEX] * num_images * image_seqlen + labels

        return input_ids, labels

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        seqlens = [len(input_ids) for input_ids in batch_ids]
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        mm_inputs["token_type_ids"] = _get_paligemma_token_type_ids(imglens, seqlens, processor)
        return mm_inputs


@dataclass
class PixtralPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if "pixel_values" in mm_inputs:
                # BC for transformers < 4.49.0
                if isinstance(mm_inputs["image_sizes"], list):
                    image_sizes = iter(mm_inputs["image_sizes"][0])
                else:
                    image_sizes = iter(mm_inputs["image_sizes"].tolist())

                image_break_token: str = getattr(processor, "image_break_token")
                image_end_token: str = getattr(processor, "image_end_token")

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if self.expand_mm_tokens:
                    patch_size = processor.patch_size * getattr(processor, "spatial_merge_size", 1)
                    height, width = next(image_sizes)
                    num_height_tokens = height // patch_size
                    num_width_tokens = width // patch_size
                    replace_tokens = [[self.image_token] * num_width_tokens + [image_break_token]] * num_height_tokens
                    replace_tokens = [item for sublist in replace_tokens for item in sublist]  # flatten list
                    replace_tokens[-1] = image_end_token
                    replace_str = "".join(replace_tokens)
                else:
                    replace_str = self.image_token

                content = content.replace(IMAGE_PLACEHOLDER, replace_str, 1)

            message["content"] = content

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        # ref to this commit https://github.com/huggingface/transformers/pull/35122
        # after transformers 4.49.0, the `image_sizes` is mandatory as an input parameter for Pixtral VisionEncoder forwarding.
        # it can be passed into `LlavaConditionalGeneration` as a parameter.
        if not is_transformers_version_greater_than("4.49.0"):
            mm_inputs.pop("image_sizes", None)
        return mm_inputs


@dataclass
class Qwen2AudioPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        bos_token: str = getattr(processor, "audio_bos_token")
        eos_token: str = getattr(processor, "audio_eos_token")
        messages = deepcopy(messages)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs([], [], audios, processor)
            if "feature_attention_mask" in mm_inputs:
                audio_lengths = mm_inputs["feature_attention_mask"].sum(-1).tolist()

        for message in messages:
            content = message["content"]
            while AUDIO_PLACEHOLDER in content:
                if self.expand_mm_tokens:
                    audio_length = audio_lengths.pop(0)
                    input_length = (audio_length - 1) // 2 + 1
                    audio_seqlen = (input_length - 2) // 2 + 1
                else:
                    audio_seqlen = 1

                content = content.replace(
                    AUDIO_PLACEHOLDER, f"{bos_token}{self.audio_token * audio_seqlen}{eos_token}", 1
                )

            message["content"] = content

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        return self._get_mm_inputs(images, videos, audios, processor)


@dataclass
class Qwen2VLPlugin(BasePlugin):
    vision_bos_token: str = "<|vision_start|>"
    vision_eos_token: str = "<|vision_end|>"

    @override
    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        image = super()._preprocess_image(image, **kwargs)
        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height))

        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height))

        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height))

        return image

    @override
    def _regularize_videos(self, videos: list["VideoInput"], **kwargs) -> "RegularizedVideoOutput":
        results, fps_per_video, durations, frames_indices = [], [], [], []
        for video in videos:
            frames: list[ImageObject] = []
            if _check_video_is_nested_images(video):
                # we assume already sample frames from videos
                for frame in video:
                    if not is_valid_image(frame) and not isinstance(frame, dict) and not os.path.exists(frame):
                        raise ValueError("Invalid image found in video frames.")

                frames = video
                fps_per_video.append(kwargs.get("video_fps", 2.0))
                durations.append(len(frames) / kwargs.get("video_fps", 2.0))
                frames_indices.append(list(range(len(frames))))
            else:
                container = av.open(video, "r")
                video_stream = next(stream for stream in container.streams if stream.type == "video")
                sample_indices = self._get_video_sample_indices(video_stream, **kwargs)
                original_fps = float(video_stream.average_rate)
                # for qwen3vl video timestamp calculation
                frames_indices.append(
                    [idx / original_fps * kwargs.get("video_fps", 2.0) for idx in sample_indices]
                )  # hack usage when do_sample_frames=False
                container.seek(0)
                for frame_idx, frame in enumerate(container.decode(video_stream)):
                    if frame_idx in sample_indices:
                        frames.append(frame.to_image())

                if video_stream.duration is None:
                    fps_per_video.append(kwargs.get("video_fps", 2.0))
                    durations.append(len(frames) / kwargs.get("video_fps", 2.0))
                else:
                    fps_per_video.append(len(sample_indices) / float(video_stream.duration * video_stream.time_base))
                    durations.append(float(video_stream.duration * video_stream.time_base))

            if len(frames) % 2 != 0:
                frames.append(frames[-1])

            frames = self._regularize_images(frames, **kwargs)["images"]
            results.append(frames)

        return {
            "videos": results,
            "fps_per_video": fps_per_video,
            "durations": durations,
            "frames_indices": frames_indices,
        }

    def _get_qwen_video_size_after_regularization(
        self, width: int, height: int, image_max_pixels: int, image_min_pixels: int
    ) -> tuple[int, int]:
        r"""Compute the frame size produced by Qwen-VL image regularization."""
        if (width * height) > image_max_pixels:
            resize_factor = math.sqrt(image_max_pixels / (width * height))
            width, height = int(width * resize_factor), int(height * resize_factor)

        if (width * height) < image_min_pixels:
            resize_factor = math.sqrt(image_min_pixels / (width * height))
            width, height = int(width * resize_factor), int(height * resize_factor)

        if min(width, height) < 28:
            width, height = max(width, 28), max(height, 28)

        if width / height > 200:
            width, height = height * 180, height

        if height / width > 200:
            width, height = width, width * 180

        return width, height

    def _get_qwen_video_stream_metadata(
        self,
        video: "VideoInput",
        video_fps: float,
        video_maxlen: int,
    ) -> Optional[dict[str, Any]]:
        if not is_pyav_available() or not isinstance(video, (str, os.PathLike)):
            return None

        try:
            container = av.open(video, "r")
        except (av.FFmpegError, OSError):
            return None

        try:
            video_stream = next((stream for stream in container.streams if stream.type == "video"), None)
            if video_stream is None:
                return None

            if video_stream.duration is None or video_stream.average_rate is None:
                return None

            average_fps = float(video_stream.average_rate)
            if average_fps <= 0:
                return None

            sample_indices = self._get_video_sample_indices(
                video_stream, video_fps=video_fps, video_maxlen=video_maxlen
            )
            return {
                "width": video_stream.width,
                "height": video_stream.height,
                "average_fps": average_fps,
                "sample_indices": sample_indices,
            }
        finally:
            container.close()

    def _get_qwen_video_resize(
        self,
        num_frames: int,
        height: int,
        width: int,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        min_pixels: int,
        max_pixels: int,
    ) -> tuple[int, int]:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

        return smart_resize(
            height=height,
            width=width,
            factor=patch_size * merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    def _get_qwen_video_grid_metadata(
        self,
        videos: list["VideoInput"],
        processor: "MMProcessor",
    ) -> Optional[dict[str, Any]]:
        if len(videos) == 0:
            return {"video_grid_thw": torch.empty((0, 3), dtype=torch.long), "frames_indices": [], "fps": 2.0}

        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        video_processor: BaseVideoProcessor = getattr(processor, "video_processor", None) or image_processor
        if image_processor is None or video_processor is None:
            return None

        patch_size = getattr(video_processor, "patch_size", None)
        temporal_patch_size = getattr(video_processor, "temporal_patch_size", None)
        merge_size = getattr(video_processor, "merge_size", None)
        size = getattr(video_processor, "size", None)
        if patch_size is None or temporal_patch_size is None or merge_size is None or size is None:
            return None

        if isinstance(size, dict):
            min_pixels = size.get("shortest_edge")
            max_pixels = size.get("longest_edge")
        else:
            min_pixels = getattr(size, "shortest_edge", None)
            max_pixels = getattr(size, "longest_edge", None)

        if min_pixels is None or max_pixels is None:
            return None

        video_fps = getattr(processor, "video_fps", 2.0)
        video_maxlen = getattr(processor, "video_maxlen", 128)
        image_max_pixels = getattr(processor, "video_max_pixels", 256 * 256)
        image_min_pixels = getattr(processor, "video_min_pixels", 16 * 16)

        video_grid_thw = []
        frames_indices = []
        for video in videos:
            metadata = self._get_qwen_video_stream_metadata(video, video_fps, video_maxlen)
            if metadata is None:
                return None

            width, height = self._get_qwen_video_size_after_regularization(
                metadata["width"], metadata["height"], image_max_pixels, image_min_pixels
            )
            num_frames = len(metadata["sample_indices"])
            if num_frames % 2 != 0:
                num_frames += 1

            resized_size = self._get_qwen_video_resize(
                num_frames,
                height,
                width,
                patch_size,
                temporal_patch_size,
                merge_size,
                min_pixels,
                max_pixels,
            )

            resized_height, resized_width = resized_size
            video_grid_thw.append(
                [
                    math.ceil(num_frames / temporal_patch_size),
                    resized_height // patch_size,
                    resized_width // patch_size,
                ]
            )
            frames_indices.append([idx / metadata["average_fps"] * video_fps for idx in metadata["sample_indices"]])

        return {
            "video_grid_thw": torch.tensor(video_grid_thw, dtype=torch.long),
            "frames_indices": frames_indices,
            "fps": video_fps,
        }

    @override
    def _get_video_token_metadata(
        self,
        videos: list["VideoInput"],
        processor: "MMProcessor",
    ) -> Optional[dict[str, Any]]:
        video_metadata = self._get_qwen_video_grid_metadata(videos, processor)
        if video_metadata is None:
            return None

        return {"video_grid_thw": video_metadata["video_grid_thw"]}

    def _get_mm_token_metadata(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "MMProcessor",
    ) -> Optional[dict[str, Any]]:
        if len(audios) != 0:
            return None

        mm_inputs = {}
        if len(images) != 0:
            mm_inputs.update(self._get_mm_inputs(images, [], [], processor))

        if len(videos) != 0:
            video_inputs = self._get_video_token_metadata(videos, processor)
            if video_inputs is None:
                return None

            mm_inputs.update(video_inputs)

        return mm_inputs

    @override
    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "MMProcessor",
    ) -> dict[str, "torch.Tensor"]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        video_processor: BaseVideoProcessor = getattr(processor, "video_processor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pt"))

        if len(videos) != 0:
            video_data = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )
            mm_inputs.update(video_processor(videos=video_data["videos"], return_tensors="pt"))
            temporal_patch_size: int = getattr(image_processor, "temporal_patch_size", 2)
            if "second_per_grid_ts" in processor.model_input_names:
                mm_inputs["second_per_grid_ts"] = [temporal_patch_size / fps for fps in video_data["fps_per_video"]]

        return mm_inputs

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")

        merge_length: int = getattr(image_processor, "merge_size") ** 2
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_token_metadata(images, videos, audios, processor)
            if mm_inputs is None:
                mm_inputs = self._get_mm_inputs(images, videos, audios, processor)

            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
        else:
            image_grid_thw = [None] * len(images)
            video_grid_thw = [None] * len(videos)

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = image_grid_thw[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"{self.vision_bos_token}{self.image_token * image_seqlen}{self.vision_eos_token}",
                    1,
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                video_seqlen = video_grid_thw[num_video_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    VIDEO_PLACEHOLDER,
                    f"{self.vision_bos_token}{self.video_token * video_seqlen}{self.vision_eos_token}",
                    1,
                )
                num_video_tokens += 1

            message["content"] = content

        return messages


@dataclass
class Qwen3VLPlugin(Qwen2VLPlugin):
    @override
    def _get_qwen_video_resize(
        self,
        num_frames: int,
        height: int,
        width: int,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        min_pixels: int,
        max_pixels: int,
    ) -> tuple[int, int]:
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize

        return smart_resize(
            num_frames=num_frames,
            height=height,
            width=width,
            temporal_factor=temporal_patch_size,
            factor=patch_size * merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    @override
    def _get_video_token_metadata(
        self,
        videos: list["VideoInput"],
        processor: "MMProcessor",
    ) -> Optional[dict[str, Any]]:
        video_metadata = self._get_qwen_video_grid_metadata(videos, processor)
        if video_metadata is None:
            return None

        return {
            "video_grid_thw": video_metadata["video_grid_thw"],
            "video_metadata": [
                SimpleNamespace(
                    frames_indices=frames_indices,
                    fps=video_metadata["fps"],
                )
                for frames_indices in video_metadata["frames_indices"]
            ],
        }

    @override
    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "MMProcessor",
    ) -> dict[str, "torch.Tensor"]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        video_processor: BaseImageProcessor = getattr(processor, "video_processor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pt"))

        if len(videos) != 0:
            videos = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )
            video_metadata = [
                {
                    "fps": getattr(processor, "video_fps", 2.0),
                    "duration": duration,
                    "total_num_frames": len(video),
                    "frames_indices": sample_indices,
                }
                for video, duration, sample_indices in zip(
                    videos["videos"], videos["durations"], videos["frames_indices"]
                )
            ]
            mm_inputs.update(
                video_processor(
                    videos=videos["videos"],
                    video_metadata=video_metadata,
                    fps=getattr(processor, "video_fps", 2.0),
                    return_metadata=True,
                    do_sample_frames=False,  # avoid changing frames_indices
                )
            )
            temporal_patch_size: int = getattr(image_processor, "temporal_patch_size", 2)
            if "second_per_grid_ts" in processor.model_input_names:
                mm_inputs["second_per_grid_ts"] = [temporal_patch_size / fps for fps in videos["fps_per_video"]]

        return mm_inputs

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")
        video_processor: BaseImageProcessor = getattr(processor, "video_processor")

        image_merge_length: int = getattr(image_processor, "merge_size") ** 2
        video_merge_length: int = getattr(video_processor, "merge_size") ** 2
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_token_metadata(images, videos, audios, processor)
            if mm_inputs is None:
                mm_inputs = self._get_mm_inputs(images, videos, audios, processor)

            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
            video_metadata = mm_inputs.get("video_metadata", [])

        else:
            image_grid_thw = [None] * len(images)
            video_grid_thw = [None] * len(videos)

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = (
                    image_grid_thw[num_image_tokens].prod() // image_merge_length if self.expand_mm_tokens else 1
                )
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"{self.vision_bos_token}{self.image_token * image_seqlen}{self.vision_eos_token}",
                    1,
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                if self.expand_mm_tokens:
                    video_grid = video_grid_thw[num_video_tokens]
                    num_frames = int(video_grid[0].item())
                    metadata = video_metadata[num_video_tokens]
                    timestamps = processor._calculate_timestamps(
                        metadata.frames_indices,
                        metadata.fps,
                        getattr(video_processor, "temporal_patch_size", 2),
                    )
                    video_structure = ""
                    video_seqlen = int((video_grid[1:].prod() // video_merge_length).item())
                    for frame_index in range(num_frames):
                        timestamp_sec = timestamps[frame_index]
                        frame_structure = (
                            f"<{timestamp_sec:.1f} seconds>"
                            f"{self.vision_bos_token}{self.video_token * video_seqlen}{self.vision_eos_token}"
                        )
                        video_structure += frame_structure
                else:
                    video_structure = f"{self.vision_bos_token}{self.video_token}{self.vision_eos_token}"

                content = content.replace(VIDEO_PLACEHOLDER, video_structure, 1)
                num_video_tokens += 1

            message["content"] = content

        return messages


@dataclass
class GLM4VPlugin(Qwen2VLPlugin):
    @override
    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "MMProcessor",
    ) -> dict[str, "torch.Tensor"]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        video_processor: BaseImageProcessor = getattr(processor, "video_processor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pt"))

        if len(videos) != 0:
            video_data = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )
            # prepare video metadata
            video_metadata = [
                {"fps": 2, "duration": duration, "total_frames": len(video)}
                for video, duration in zip(video_data["videos"], video_data["durations"])
            ]
            mm_inputs.update(video_processor(images=None, videos=video_data["videos"], video_metadata=video_metadata))

        return mm_inputs

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")

        merge_length: int = getattr(image_processor, "merge_size") ** 2
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
            num_frames = video_grid_thw[0][0] if len(video_grid_thw) > 0 else 0  # hard code for now
            timestamps = mm_inputs.get("timestamps", [])

            if hasattr(timestamps, "tolist"):
                timestamps = timestamps.tolist()

            if not timestamps:
                timestamps_list = []
            elif isinstance(timestamps[0], list):
                timestamps_list = timestamps[0]
            else:
                timestamps_list = timestamps

            unique_timestamps = timestamps_list.copy()
            selected_timestamps = unique_timestamps[:num_frames]
            while len(selected_timestamps) < num_frames:
                selected_timestamps.append(selected_timestamps[-1] if selected_timestamps else 0)

        else:
            image_grid_thw = [None] * len(images)
            video_grid_thw = [None] * len(videos)
            num_frames = 0
            selected_timestamps = [0]

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = image_grid_thw[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER, f"<|begin_of_image|>{self.image_token * image_seqlen}<|end_of_image|>", 1
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                video_structure = ""
                for frame_index in range(num_frames):
                    video_seqlen = (
                        video_grid_thw[num_video_tokens][1:].prod() // merge_length if self.expand_mm_tokens else 1
                    )
                    timestamp_sec = selected_timestamps[frame_index]
                    frame_structure = (
                        f"<|begin_of_image|>{self.image_token * video_seqlen}<|end_of_image|>{timestamp_sec}"
                    )
                    video_structure += frame_structure

                if not self.expand_mm_tokens:
                    video_structure = self.video_token

                content = content.replace(VIDEO_PLACEHOLDER, f"<|begin_of_video|>{video_structure}<|end_of_video|>", 1)
                num_video_tokens += 1

            message["content"] = content

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["ProcessorMixin"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        mm_inputs.pop("timestamps", None)
        return mm_inputs


@dataclass
class Qwen2OmniPlugin(Qwen2VLPlugin):
    audio_bos_token: str = "<|audio_start|>"
    audio_eos_token: str = "<|audio_end|>"

    @override
    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "MMProcessor",
    ) -> dict[str, "torch.Tensor"]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        video_processor: BaseVideoProcessor = getattr(processor, "video_processor", None)
        feature_extractor: SequenceFeatureExtractor = getattr(processor, "feature_extractor", None) or getattr(
            processor, "audio_processor", None
        )
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pt"))

        if len(videos) != 0:
            video_dict = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )
            mm_inputs.update(video_processor(videos=video_dict["videos"], return_tensors="pt"))
            temporal_patch_size: int = getattr(image_processor, "temporal_patch_size", 2)
            mm_inputs["video_second_per_grid"] = torch.tensor(
                [temporal_patch_size / fps for fps in video_dict["fps_per_video"]]
            )

        if len(audios) != 0:
            audios = self._regularize_audios(
                audios,
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            )["audios"]
            mm_inputs.update(
                feature_extractor(
                    audios,
                    sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
                    return_attention_mask=True,
                    padding="max_length",
                    return_tensors="pt",
                )
            )
            mm_inputs["feature_attention_mask"] = mm_inputs.pop("attention_mask")  # prevent conflicts

        return mm_inputs

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens, num_audio_tokens = 0, 0, 0
        messages = deepcopy(messages)
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)

        merge_length = processor.image_processor.merge_size**2
        use_audio_in_video = getattr(processor, "use_audio_in_video", False)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
            if "feature_attention_mask" in mm_inputs:
                if processor.__class__.__name__ == "Qwen3OmniMoeProcessor":  # for qwen3omni
                    input_lengths = mm_inputs["feature_attention_mask"].sum(-1)
                    input_lengths_leave = input_lengths % 100
                    feature_lengths = (input_lengths_leave - 1) // 2 + 1
                    audio_lengths = ((feature_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
                else:
                    input_lengths = (mm_inputs["feature_attention_mask"].sum(-1).numpy() - 1) // 2 + 1
                    audio_lengths = (input_lengths - 2) // 2 + 1
        else:
            mm_inputs = {}
            image_grid_thw = [None] * len(images)
            video_grid_thw = [None] * len(videos)
            audio_lengths = [None] * len(audios)

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = image_grid_thw[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"{self.vision_bos_token}{self.image_token * image_seqlen}{self.vision_eos_token}",
                    1,
                )
                num_image_tokens += 1

            if (
                use_audio_in_video and len(audios) and len(videos)
            ):  # if use the audio of video # deal video token and audio token together
                if len(videos) != len(audios):
                    raise ValueError(
                        f"Number of videos ({len(videos)}) must match number of audios ({len(audios)}) when using audio in video."
                    )

                while VIDEO_PLACEHOLDER in content:
                    video_pos = content.find(VIDEO_PLACEHOLDER)
                    audio_pos = content.find(AUDIO_PLACEHOLDER, video_pos)
                    if audio_pos == -1 or audio_pos < video_pos:
                        raise ValueError(
                            f"Each {VIDEO_PLACEHOLDER} must be followed by an {AUDIO_PLACEHOLDER} when using audio in video."
                        )

                    position_id_per_seconds: int = getattr(processor, "position_id_per_seconds", 25)
                    audio_t_index = torch.arange(audio_lengths[num_audio_tokens])
                    video_t_index = (
                        torch.arange(video_grid_thw[num_video_tokens][0])
                        .view(-1, 1, 1)
                        .expand(
                            -1,
                            video_grid_thw[num_video_tokens][1] // image_processor.merge_size,
                            video_grid_thw[num_video_tokens][2] // image_processor.merge_size,
                        )
                        .flatten()
                        * mm_inputs["video_second_per_grid"][num_video_tokens]
                        * position_id_per_seconds
                    ).long()
                    t_ntoken_per_chunk = position_id_per_seconds * 2
                    video_chunk_indices = processor.get_chunked_index(video_t_index, t_ntoken_per_chunk)
                    audio_chunk_indices = processor.get_chunked_index(audio_t_index, t_ntoken_per_chunk)
                    placeholder_string = ""
                    placeholder_string += self.vision_bos_token + self.audio_bos_token
                    for j in range(max(len(video_chunk_indices), len(audio_chunk_indices))):
                        video_chunk_index = video_chunk_indices[j] if j < len(video_chunk_indices) else None
                        audio_chunk_index = audio_chunk_indices[j] if j < len(audio_chunk_indices) else None
                        if video_chunk_index is not None:
                            placeholder_string += self.video_token * (video_chunk_index[1] - video_chunk_index[0])

                        if audio_chunk_index is not None:
                            placeholder_string += self.audio_token * (audio_chunk_index[1] - audio_chunk_index[0])

                    placeholder_string += self.audio_eos_token + self.vision_eos_token
                    content = content.replace(VIDEO_PLACEHOLDER, placeholder_string, 1)
                    content = content.replace(AUDIO_PLACEHOLDER, "", 1)
                    num_audio_tokens += 1
                    num_video_tokens += 1
            else:
                while AUDIO_PLACEHOLDER in content:
                    audio_seqlen = audio_lengths[num_audio_tokens] if self.expand_mm_tokens else 1
                    content = content.replace(
                        AUDIO_PLACEHOLDER,
                        f"{self.audio_bos_token}{self.audio_token * audio_seqlen}{self.audio_eos_token}",
                        1,
                    )
                    num_audio_tokens += 1

                while VIDEO_PLACEHOLDER in content:
                    video_seqlen = (
                        video_grid_thw[num_video_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                    )
                    content = content.replace(
                        VIDEO_PLACEHOLDER,
                        f"{self.vision_bos_token}{self.video_token * video_seqlen}{self.vision_eos_token}",
                        1,
                    )
                    num_video_tokens += 1

            message["content"] = content

        return messages


@dataclass
class VideoLlavaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        num_frames = 0
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if "pixel_values_images" in mm_inputs:
                height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values_images"][0]))
                num_frames = 1

            if "pixel_values_videos" in mm_inputs:
                one_video = to_numpy_array(mm_inputs["pixel_values_videos"][0])
                height, width = get_image_size(one_video[0])
                num_frames = one_video.shape[0]  # frame dim is always after batch dim

            if "pixel_values_images" in mm_inputs or "pixel_values_videos" in mm_inputs:
                image_seqlen = (height // processor.patch_size) * (
                    width // processor.patch_size
                ) + processor.num_additional_image_tokens
                video_seqlen = image_seqlen * num_frames
                if processor.vision_feature_select_strategy == "default":
                    image_seqlen -= 1
        else:
            image_seqlen, video_seqlen = 1, 1

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                content = content.replace(VIDEO_PLACEHOLDER, "{{video}}" * video_seqlen, 1)
                num_video_tokens += 1

            content = content.replace("{{image}}", self.image_token)
            message["content"] = content.replace("{{video}}", self.video_token)

        return messages


@dataclass
class LFMVLPlugin(BasePlugin):
    r"""Plugin for LFM2.5-VL vision-language models.

    LFM2.5-VL uses dynamic image token counts based on image resolution.
    The image processor returns spatial_shapes tensor with [height, width] grid dimensions.
    Token count per image = (spatial_h * spatial_w) / (downsample_factor^2)
    """

    @override
    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "MMProcessor",
    ) -> dict[str, "torch.Tensor"]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pt"))
        return mm_inputs

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens = 0
        messages = deepcopy(messages)
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")
        downsample_factor: int = getattr(image_processor, "downsample_factor", 2)

        if self.expand_mm_tokens and len(images) > 0:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            spatial_shapes = mm_inputs.get("spatial_shapes", [])
        else:
            spatial_shapes = []

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if self.expand_mm_tokens and len(spatial_shapes) > num_image_tokens:
                    h, w = spatial_shapes[num_image_tokens].tolist()
                    image_seqlen = (h * w) // (downsample_factor * downsample_factor)
                else:
                    image_seqlen = 1

                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)
                num_image_tokens += 1

            message["content"] = content.replace("{{image}}", self.image_token)

        return messages


@dataclass
class YoutuVLPlugin(BasePlugin):
    r"""Plugin for Youtu-VL vision-language models."""

    vision_bos_token: str = "<|vision_start|>"
    vision_eos_token: str = "<|vision_end|>"

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)

        for message in messages:
            content = message["content"]
            content = content.replace(
                IMAGE_PLACEHOLDER, f"{self.vision_bos_token}{self.image_token}{self.vision_eos_token}"
            )
            content = content.replace(
                VIDEO_PLACEHOLDER, f"{self.vision_bos_token}{self.video_token}{self.vision_eos_token}"
            )

            message["content"] = content

        return messages


PLUGINS = {
    "base": BasePlugin,
    "ernie_vl": ErnieVLPlugin,
    "gemma3": Gemma3Plugin,
    "gemma3n": Gemma3nPlugin,
    "gemma4": Gemma4Plugin,
    "glm4v": GLM4VPlugin,
    "intern_vl": InternVLPlugin,
    "kimi_vl": KimiVLPlugin,
    "llama4": Llama4Plugin,
    "llava": LlavaPlugin,
    "llava_next": LlavaNextPlugin,
    "llava_next_video": LlavaNextVideoPlugin,
    "lfm2_vl": LFMVLPlugin,
    "minicpm_v": MiniCPMVPlugin,
    "minicpm_v_4_6": MiniCPMV4_6Plugin,
    "mllama": MllamaPlugin,
    "moss_vl": MossVLPlugin,
    "paligemma": PaliGemmaPlugin,
    "pixtral": PixtralPlugin,
    "qwen2_audio": Qwen2AudioPlugin,
    "qwen2_omni": Qwen2OmniPlugin,
    "qwen2_vl": Qwen2VLPlugin,
    "qwen3_vl": Qwen3VLPlugin,
    "video_llava": VideoLlavaPlugin,
    "youtu_vl": YoutuVLPlugin,
}


def register_mm_plugin(name: str, plugin_class: type["BasePlugin"]) -> None:
    r"""Register a multimodal plugin."""
    if name in PLUGINS:
        raise ValueError(f"Multimodal plugin {name} already exists.")

    PLUGINS[name] = plugin_class


def get_mm_plugin(
    name: str,
    image_token: str | None = None,
    video_token: str | None = None,
    audio_token: str | None = None,
    **kwargs,
) -> "BasePlugin":
    r"""Get plugin for multimodal inputs."""
    if name not in PLUGINS:
        raise ValueError(f"Multimodal plugin `{name}` not found.")

    return PLUGINS[name](image_token, video_token, audio_token, **kwargs)
