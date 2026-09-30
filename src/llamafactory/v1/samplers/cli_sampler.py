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
from collections.abc import Generator
from threading import Thread

from ..config import InputArgument, ModelArguments, SampleArguments, SampleBackend, get_args
from ..core.base_sampler import BaseSampler
from ..core.data_engine import DataEngine
from ..core.model_engine import ModelEngine
from ..core.rendering import Renderer
from ..utils.types import HFModel, Message, Sample, TorchDataset


class SyncSampler(BaseSampler):
    def __init__(
        self,
        args: SampleArguments,
        model_args: ModelArguments,
        model: HFModel,
        renderer: Renderer,
    ) -> None:
        def _start_background_loop(loop: asyncio.AbstractEventLoop) -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        super().__init__(args, model_args, model, renderer)
        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=_start_background_loop, args=(self._loop,), daemon=True)
        self._thread.start()

    def generate(self, messages: list[Message], tools: str | None = None) -> Generator[str, None, None]:
        """Generate tokens synchronously.

        Args:
            messages: List of messages.
            tools: Tools string.

        Yields:
            Generated tokens.
        """
        generator = super().generate(messages, tools)
        while True:
            try:
                token = asyncio.run_coroutine_threadsafe(generator.__anext__(), self._loop).result()
                yield token
            except StopAsyncIteration:
                break

    def batch_infer(self, dataset: TorchDataset) -> list[Sample]:
        """Batch infer samples synchronously.

        Args:
            dataset: Torch dataset.

        Returns:
            List of samples.
        """
        return asyncio.run_coroutine_threadsafe(super().batch_infer(dataset), self._loop).result()


def run_chat(args: InputArgument = None):
    model_args, data_args, _, sample_args = get_args(args)
    if sample_args.sample_backend != SampleBackend.HF:
        model_args.init_plugin = {"name": "init_on_meta"}

    model_engine = ModelEngine(model_args)
    sampler = SyncSampler(sample_args, model_args, model_engine.model, model_engine.renderer)
    if data_args.train_dataset is not None:
        dataset = DataEngine(data_args.train_dataset)
        sampler.batch_infer(dataset)
    else:
        if os.name != "nt":
            try:
                import readline  # noqa: F401
            except ImportError:
                print("Install `readline` for a better experience.")

        messages = []
        print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")

        while True:
            try:
                query = input("\nUser: ")
            except UnicodeDecodeError:
                print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
                continue
            except Exception:
                raise

            if query.strip() == "exit":
                break

            if query.strip() == "clear":
                messages = []
                print("History has been removed.")
                continue

            messages.append({"role": "user", "content": [{"type": "text", "value": query}]})
            print("Assistant: ", end="", flush=True)

            response = ""
            for new_text in sampler.generate(messages):
                print(new_text, end="", flush=True)
                response += new_text

            print()
            messages.append({"role": "assistant", "content": [{"type": "text", "value": response}]})


if __name__ == "__main__":
    run_chat()
