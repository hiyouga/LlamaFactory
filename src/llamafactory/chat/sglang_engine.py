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
import atexit
import json
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any, Optional, Union

import requests
from typing_extensions import override

from ..data import get_template_and_fix_tokenizer
from ..extras import logging
from ..extras.constants import AUDIO_PLACEHOLDER, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER, EngineName
from ..extras.misc import get_device_count, torch_gc
from ..extras.packages import is_sglang_available
from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments
from ..model import load_config, load_tokenizer
from ..model.model_utils.quantization import QuantizationMethod
from .base_engine import BaseEngine, Response


if is_sglang_available():
    from sglang.utils import launch_server_cmd, terminate_process, wait_for_server  # type: ignore


if TYPE_CHECKING:
    from ..data.mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


class SGLangEngine(BaseEngine):
    """Inference engine for SGLang models.

    This class wraps the SGLang engine to provide a consistent interface for text generation
    that matches LLaMA Factory's requirements. It uses the SGLang HTTP server approach for
    better interaction and performance. The engine launches a server process and communicates
    with it via HTTP requests.

    For more details on the SGLang HTTP server approach, see:
    https://docs.sglang.ai/backend/send_request.html
    """

    def __init__(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
    ) -> None:
        self.name = EngineName.SGLANG
        self.model_args = model_args
        config = load_config(model_args)  # may download model from ms hub
        if getattr(config, "quantization_config", None):  # gptq models should use float16
            quantization_config: dict[str, Any] = getattr(config, "quantization_config", None)
            quant_method = quantization_config.get("quant_method", "")
            if quant_method == QuantizationMethod.GPTQ and model_args.infer_dtype == "auto":
                model_args.infer_dtype = "float16"

        self.can_generate = finetuning_args.stage == "sft"
        tokenizer_module = load_tokenizer(model_args)
        self.tokenizer = tokenizer_module["tokenizer"]
        self.processor = tokenizer_module["processor"]
        self.tokenizer.padding_side = "left"
        self.template = get_template_and_fix_tokenizer(self.tokenizer, data_args)
        self.template.mm_plugin.expand_mm_tokens = False  # for sglang generate
        self.generating_args = generating_args.to_dict()
        if model_args.adapter_name_or_path is not None:
            self.lora_request = True
        else:
            self.lora_request = False

        launch_cmd = [
            "python3 -m sglang.launch_server",
            f"--model-path {model_args.model_name_or_path}",
            f"--dtype {model_args.infer_dtype}",
            f"--context-length {model_args.sglang_maxlen}",
            f"--mem-fraction-static {model_args.sglang_mem_fraction}",
            f"--tp-size {model_args.sglang_tp_size if model_args.sglang_tp_size != -1 else get_device_count() or 1}",
            f"--download-dir {model_args.cache_dir}",
            "--log-level error",
        ]
        if self.lora_request:
            launch_cmd.extend(
                [
                    "--max-loras-per-batch 1",
                    f"--lora-backend {model_args.sglang_lora_backend}",
                    f"--lora-paths lora0={model_args.adapter_name_or_path[0]}",
                    "--disable-radix-cache",
                ]
            )
        launch_cmd = " ".join(launch_cmd)
        logger.info_rank0(f"Starting SGLang server with command: {launch_cmd}")
        try:
            torch_gc()
            self.server_process, port = launch_server_cmd(launch_cmd)
            self.base_url = f"http://localhost:{port}"
            atexit.register(self._cleanup_server)

            logger.info_rank0(f"Waiting for SGLang server to be ready at {self.base_url}")
            wait_for_server(self.base_url, timeout=300)
            logger.info_rank0(f"SGLang server initialized successfully at {self.base_url}")
            try:
                response = requests.get(f"{self.base_url}/get_model_info", timeout=5)
                if response.status_code == 200:
                    model_info = response.json()
                    logger.info(f"SGLang server model info: {model_info}")
            except Exception as e:
                logger.debug(f"Note: could not get model info: {str(e)}")

        except Exception as e:
            logger.error(f"Failed to start SGLang server: {str(e)}")
            self._cleanup_server()  # make sure to clean up any started process
            raise RuntimeError(f"SGLang server initialization failed: {str(e)}.")

    def _cleanup_server(self):
        r"""Clean up the server process when the engine is destroyed."""
        if hasattr(self, "server_process") and self.server_process:
            try:
                logger.info("Terminating SGLang server process")
                terminate_process(self.server_process)
                logger.info("SGLang server process terminated")
            except Exception as e:
                logger.warning(f"Error terminating SGLang server: {str(e)}")

    async def _generate(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[list["ImageInput"]] = None,
        videos: Optional[list["VideoInput"]] = None,
        audios: Optional[list["AudioInput"]] = None,
        **input_kwargs,
    ) -> AsyncIterator[dict[str, Any]]:
        if images is not None and not any(IMAGE_PLACEHOLDER in message["content"] for message in messages):
            messages[0]["content"] = IMAGE_PLACEHOLDER * len(images) + messages[0]["content"]

        if videos is not None and not any(VIDEO_PLACEHOLDER in message["content"] for message in messages):
            messages[0]["content"] = VIDEO_PLACEHOLDER * len(videos) + messages[0]["content"]

        if audios is not None and not any(AUDIO_PLACEHOLDER in message["content"] for message in messages):
            messages[0]["content"] = AUDIO_PLACEHOLDER * len(audios) + messages[0]["content"]

        messages = self.template.mm_plugin.process_messages(
            messages, images or [], videos or [], audios or [], self.processor
        )
        paired_messages = messages + [{"role": "assistant", "content": ""}]
        prompt_ids, _ = self.template.encode_oneturn(self.tokenizer, paired_messages, system, tools)
        prompt_length = len(prompt_ids)

        temperature: Optional[float] = input_kwargs.pop("temperature", None)
        top_p: Optional[float] = input_kwargs.pop("top_p", None)
        top_k: Optional[float] = input_kwargs.pop("top_k", None)
        num_return_sequences: int = input_kwargs.pop("num_return_sequences", 1)
        repetition_penalty: Optional[float] = input_kwargs.pop("repetition_penalty", None)
        skip_special_tokens: Optional[bool] = input_kwargs.pop("skip_special_tokens", None)
        max_length: Optional[int] = input_kwargs.pop("max_length", None)
        max_new_tokens: Optional[int] = input_kwargs.pop("max_new_tokens", None)
        seed: Optional[int] = input_kwargs.pop("seed", None)
        stop: Optional[Union[str, list[str]]] = input_kwargs.pop("stop", None)

        if num_return_sequences != 1:
            raise NotImplementedError("SGLang only supports n=1.")

        if "max_new_tokens" in self.generating_args:
            max_tokens = self.generating_args["max_new_tokens"]
        elif "max_length" in self.generating_args:
            if self.generating_args["max_length"] > prompt_length:
                max_tokens = self.generating_args["max_length"] - prompt_length
            else:
                max_tokens = 1

        if max_length:
            max_tokens = max_length - prompt_length if max_length > prompt_length else 1

        if max_new_tokens:
            max_tokens = max_new_tokens

        sampling_params = {
            "temperature": temperature if temperature is not None else self.generating_args["temperature"],
            "top_p": (top_p if top_p is not None else self.generating_args["top_p"]) or 1.0,  # top_p must > 0
            "top_k": (top_k if top_k is not None else self.generating_args["top_k"]) or -1,  # top_k must > 0
            "stop": stop,
            "stop_token_ids": self.template.get_stop_token_ids(self.tokenizer),
            "max_new_tokens": max_tokens,
            "repetition_penalty": (
                repetition_penalty if repetition_penalty is not None else self.generating_args["repetition_penalty"]
            )
            or 1.0,  # repetition_penalty must > 0
            "skip_special_tokens": skip_special_tokens
            if skip_special_tokens is not None
            else self.generating_args["skip_special_tokens"],
        }
        if seed is not None:
            sampling_params["seed"] = seed

        def stream_request():
            json_data = {
                "input_ids": prompt_ids,
                "sampling_params": sampling_params,
                "stream": True,
            }
            if self.lora_request:
                json_data["lora_request"] = ["lora0"]
            response = requests.post(f"{self.base_url}/generate", json=json_data, stream=True)
            if response.status_code != 200:
                raise RuntimeError(f"SGLang server error: {response.status_code}, {response.text}")

            for chunk in response.iter_lines(decode_unicode=False):
                chunk = str(chunk.decode("utf-8"))
                if chunk == "data: [DONE]":
                    break

                if chunk and chunk.startswith("data:"):
                    yield json.loads(chunk[5:].strip("\n"))

        return await asyncio.to_thread(stream_request)

    @override
    async def chat(
        self,
        messages: Sequence[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[Sequence["ImageInput"]] = None,
        videos: Optional[Sequence["VideoInput"]] = None,
        audios: Optional[Sequence["AudioInput"]] = None,
        **input_kwargs,
    ) -> list["Response"]:
        final_output = None
        generator = await self._generate(messages, system, tools, images, videos, audios, **input_kwargs)
        for request_output in generator:
            final_output = request_output

        results = [
            Response(
                response_text=final_output["text"],
                response_length=final_output["meta_info"]["completion_tokens"],
                prompt_length=final_output["meta_info"]["prompt_tokens"],
                finish_reason="stop" if final_output["meta_info"]["finish_reason"] == "stop" else "length",
            )
        ]
        return results

    @override
    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[list["ImageInput"]] = None,
        videos: Optional[list["VideoInput"]] = None,
        audios: Optional[list["AudioInput"]] = None,
        **input_kwargs,
    ) -> AsyncGenerator[str, None]:
        generated_text = ""
        generator = await self._generate(messages, system, tools, images, videos, audios, **input_kwargs)
        for result in generator:
            delta_text = result["text"][len(generated_text) :]
            generated_text = result["text"]
            yield delta_text

    @override
    async def get_scores(
        self,
        batch_input: list[str],
        **input_kwargs,
    ) -> list[float]:
        raise NotImplementedError("SGLang engine does not support `get_scores`.")

    def __del__(self):
        r"""Ensure server is cleaned up when object is deleted."""
        self._cleanup_server()
        try:
            atexit.unregister(self._cleanup_server)
        except Exception:
            pass
