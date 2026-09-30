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
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from transformers.utils import is_torch_npu_available

from ..chat import ChatModel
from ..data import Role
from ..extras.constants import PEFT_METHODS
from ..extras.misc import torch_gc
from ..extras.packages import is_gradio_available
from .common import get_save_dir, load_config
from .locales import ALERTS


if TYPE_CHECKING:
    from ..chat import BaseEngine
    from .manager import Manager


if is_gradio_available():
    import gradio as gr


def _escape_html(text: str) -> str:
    r"""Escape HTML characters."""
    return text.replace("<", "&lt;").replace(">", "&gt;")


def _format_response(text: str, lang: str, escape_html: bool, thought_words: tuple[str, str]) -> str:
    r"""Post-process the response text.

    Based on: https://huggingface.co/spaces/Lyte/DeepSeek-R1-Distill-Qwen-1.5B-Demo-GGUF/blob/main/app.py
    """
    if thought_words[0] not in text:
        return _escape_html(text) if escape_html else text

    text = text.replace(thought_words[0], "")
    result = text.split(thought_words[1], maxsplit=1)
    if len(result) == 1:
        summary = ALERTS["info_thinking"][lang]
        thought, answer = text, ""
    else:
        summary = ALERTS["info_thought"][lang]
        thought, answer = result

    if escape_html:
        thought, answer = _escape_html(thought), _escape_html(answer)

    return (
        f"<details open><summary class='thinking-summary'><span>{summary}</span></summary>\n\n"
        f"<div class='thinking-container'>\n{thought}\n</div>\n</details>{answer}"
    )


@contextmanager
def update_attr(obj: Any, name: str, value: Any):
    old_value = getattr(obj, name, None)
    setattr(obj, name, value)
    yield
    setattr(obj, name, old_value)


class WebChatModel(ChatModel):
    def __init__(self, manager: "Manager", demo_mode: bool = False, lazy_init: bool = True) -> None:
        self.manager = manager
        self.demo_mode = demo_mode
        self.engine: BaseEngine | None = None

        if not lazy_init:  # read arguments from command line
            super().__init__()

        if demo_mode and os.getenv("DEMO_MODEL") and os.getenv("DEMO_TEMPLATE"):  # load demo model
            model_name_or_path = os.getenv("DEMO_MODEL")
            template = os.getenv("DEMO_TEMPLATE")
            infer_backend = os.getenv("DEMO_BACKEND", "huggingface")
            super().__init__(
                dict(model_name_or_path=model_name_or_path, template=template, infer_backend=infer_backend)
            )

    @property
    def loaded(self) -> bool:
        return self.engine is not None

    def load_model(self, data) -> Generator[str, None, None]:
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
        lang, model_name, model_path = get("top.lang"), get("top.model_name"), get("top.model_path")
        finetuning_type, checkpoint_path = get("top.finetuning_type"), get("top.checkpoint_path")
        user_config = load_config()

        error = ""
        if self.loaded:
            error = ALERTS["err_exists"][lang]
        elif not model_name:
            error = ALERTS["err_no_model"][lang]
        elif not model_path:
            error = ALERTS["err_no_path"][lang]
        elif self.demo_mode:
            error = ALERTS["err_demo"][lang]

        try:
            json.loads(get("infer.extra_args"))
        except json.JSONDecodeError:
            error = ALERTS["err_json_schema"][lang]

        if error:
            gr.Warning(error)
            yield error
            return

        yield ALERTS["info_loading"][lang]
        args = dict(
            model_name_or_path=model_path,
            cache_dir=user_config.get("cache_dir", None),
            finetuning_type=finetuning_type,
            template=get("top.template"),
            rope_scaling=get("top.rope_scaling") if get("top.rope_scaling") != "none" else None,
            flash_attn="fa2" if get("top.booster") == "flashattn2" else "auto",
            use_unsloth=(get("top.booster") == "unsloth"),
            enable_liger_kernel=(get("top.booster") == "liger_kernel"),
            infer_backend=get("infer.infer_backend"),
            infer_dtype=get("infer.infer_dtype"),
            trust_remote_code=True,
        )
        args.update(json.loads(get("infer.extra_args")))

        # checkpoints
        if checkpoint_path:
            if finetuning_type in PEFT_METHODS:  # list
                args["adapter_name_or_path"] = ",".join(
                    [get_save_dir(model_name, finetuning_type, adapter) for adapter in checkpoint_path]
                )
            else:  # str
                args["model_name_or_path"] = get_save_dir(model_name, finetuning_type, checkpoint_path)

        # quantization
        if get("top.quantization_bit") != "none":
            args["quantization_bit"] = int(get("top.quantization_bit"))
            args["quantization_method"] = get("top.quantization_method")
            args["double_quantization"] = not is_torch_npu_available()

        super().__init__(args)
        yield ALERTS["info_loaded"][lang]

    def unload_model(self, data) -> Generator[str, None, None]:
        lang = data[self.manager.get_elem_by_id("top.lang")]

        if self.demo_mode:
            gr.Warning(ALERTS["err_demo"][lang])
            yield ALERTS["err_demo"][lang]
            return

        yield ALERTS["info_unloading"][lang]
        self.engine = None
        torch_gc()
        yield ALERTS["info_unloaded"][lang]

    @staticmethod
    def append(
        chatbot: list[dict[str, str]],
        messages: list[dict[str, str]],
        role: str,
        query: str,
        escape_html: bool,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
        r"""Add the user input to chatbot.

        Inputs: infer.chatbot, infer.messages, infer.role, infer.query, infer.escape_html
        Output: infer.chatbot, infer.messages, infer.query
        """
        return (
            chatbot + [{"role": "user", "content": _escape_html(query) if escape_html else query}],
            messages + [{"role": role, "content": query}],
            "",
        )

    def stream(
        self,
        chatbot: list[dict[str, str]],
        messages: list[dict[str, str]],
        lang: str,
        system: str,
        tools: str,
        image: Any | None,
        video: Any | None,
        audio: Any | None,
        max_new_tokens: int,
        top_p: float,
        temperature: float,
        seed: float | int,
        skip_special_tokens: bool,
        escape_html: bool,
        enable_thinking: bool,
    ) -> Generator[tuple[list[dict[str, str]], list[dict[str, str]]], None, None]:
        r"""Generate output text in stream.

        Inputs: infer.chatbot, infer.messages, infer.system, infer.tools, infer.image, infer.video, ...
        Output: infer.chatbot, infer.messages
        """
        with update_attr(self.engine.template, "enable_thinking", enable_thinking):
            chatbot.append({"role": "assistant", "content": ""})
            response = ""
            for new_text in self.stream_chat(
                messages,
                system,
                tools,
                images=[image] if image else None,
                videos=[video] if video else None,
                audios=[audio] if audio else None,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                temperature=temperature,
                seed=None if seed is None or int(seed) < 0 else int(seed),
                skip_special_tokens=skip_special_tokens,
            ):
                response += new_text
                if tools:
                    result = self.engine.template.extract_tool(response)
                else:
                    result = response

                if isinstance(result, list):
                    tool_calls = [{"name": tool.name, "arguments": json.loads(tool.arguments)} for tool in result]
                    tool_calls = json.dumps(tool_calls, ensure_ascii=False)
                    output_messages = messages + [{"role": Role.FUNCTION.value, "content": tool_calls}]
                    bot_text = "```json\n" + tool_calls + "\n```"
                else:
                    output_messages = messages + [{"role": Role.ASSISTANT.value, "content": result}]
                    bot_text = _format_response(result, lang, escape_html, self.engine.template.thought_words)

                chatbot[-1] = {"role": "assistant", "content": bot_text}
                yield chatbot, output_messages
