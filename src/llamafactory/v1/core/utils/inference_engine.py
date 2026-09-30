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

import asyncio
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from threading import Thread

import torch
from transformers import AsyncTextIteratorStreamer

from ...accelerator.interface import DistributedInterface
from ...config import ModelArguments, SampleArguments
from ...utils.helper import get_tokenizer
from ...utils.types import HFModel, Message, Sample, TorchDataset
from ..rendering import Renderer


class BaseEngine(ABC):
    @abstractmethod
    def __init__(
        self,
        args: SampleArguments,
        model_args: ModelArguments,
        model: HFModel,
        renderer: Renderer,
    ) -> None:
        """Initialize the engine.

        Args:
            args: Sample arguments.
            model_args: Model arguments.
            model: Model.
            renderer: Renderer.
        """
        ...

    @abstractmethod
    async def generate(self, messages: list[Message], tools: str | None = None) -> AsyncGenerator[str, None]:
        """Generate tokens asynchronously.

        Args:
            messages: List of messages.
            tools: Tools string.

        Yields:
            Generated tokens.
        """
        ...

    @abstractmethod
    async def batch_infer(self, dataset: TorchDataset) -> list[Sample]:
        """Batch infer samples.

        Args:
            dataset: Torch dataset.

        Returns:
            List of samples.
        """
        ...


class HuggingFaceEngine(BaseEngine):
    def __init__(
        self,
        args: SampleArguments,
        model_args: ModelArguments,
        model: HFModel,
        renderer: Renderer,
    ) -> None:
        self.args = args
        self.model_args = model_args
        self.model = model
        self.renderer = renderer
        self.semaphore = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT", "1")))

    @torch.inference_mode()
    async def generate(self, messages: list[Message], tools: str | None = None) -> AsyncGenerator[str, None]:
        async with self.semaphore:
            model_inputs = self.renderer.render_messages(messages, tools, is_generate=True)
            streamer = AsyncTextIteratorStreamer(
                tokenizer=get_tokenizer(self.renderer.processor),
                skip_prompt=True,
                skip_special_tokens=True,  # TODO: configurable
            )
            device = DistributedInterface().current_device
            kwargs = {
                "input_ids": torch.tensor([model_inputs["input_ids"]]).to(device),
                "attention_mask": torch.tensor([model_inputs["attention_mask"]]).to(device),
                "max_new_tokens": self.args.max_new_tokens,
                "streamer": streamer,
            }
            thread = Thread(target=self.model.generate, kwargs=kwargs, daemon=True)
            thread.start()

            async for token in streamer:
                yield token

    async def batch_infer(self, dataset: TorchDataset) -> list[Sample]:
        """Batch infer samples.

        Args:
            dataset: Torch dataset.

        Returns:
            List of samples.
        """
        raise NotImplementedError("Batch infer is not implemented.")
