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

from collections.abc import AsyncGenerator

from ..config import ModelArguments, SampleArguments, SampleBackend
from ..utils.types import HFModel, Message, Sample, TorchDataset
from .rendering import Renderer
from .utils.inference_engine import HuggingFaceEngine


class BaseSampler:
    """Base sampler.

    Args:
        args: Sample arguments.
        model_args: Model arguments.
        model: Model.
        renderer: Renderer.
    """

    def __init__(
        self,
        args: SampleArguments,
        model_args: ModelArguments,
        model: HFModel,
        renderer: Renderer,
    ) -> None:
        if args.sample_backend == SampleBackend.HF:
            self.engine = HuggingFaceEngine(args, model_args, model, renderer)
        else:
            raise ValueError(f"Unknown sample backend: {args.sample_backend}")

    async def generate(self, messages: list[Message], tools: str | None = None) -> AsyncGenerator[str, None]:
        """Generate tokens asynchronously.

        Args:
            messages: List of messages.
            tools: Tools string.

        Yields:
            Generated tokens.
        """
        async for token in self.engine.generate(messages, tools):
            yield token

    async def batch_infer(self, dataset: TorchDataset) -> list[Sample]:
        """Batch infer samples.

        Args:
            dataset: Torch dataset.

        Returns:
            List of samples.
        """
        return await self.engine.batch_infer(dataset)
