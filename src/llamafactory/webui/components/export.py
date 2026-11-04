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
from collections.abc import Generator
from typing import TYPE_CHECKING

from ...extras.constants import PEFT_METHODS
from ...extras.misc import is_env_enabled, torch_gc
from ...extras.packages import is_gradio_available
from ...train.tuner import export_model
from ..common import get_save_dir, load_config
from ..locales import ALERTS


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


GPTQ_BITS = ["8", "4", "3", "2"]


def can_quantize(checkpoint_path: str | list[str]) -> "gr.Dropdown":
    if isinstance(checkpoint_path, list) and len(checkpoint_path) != 0:
        return gr.Dropdown(value="none", interactive=False)
    else:
        return gr.Dropdown(interactive=True)


def save_model(
    lang: str,
    model_name: str,
    model_path: str,
    finetuning_type: str,
    checkpoint_path: str | list[str],
    template: str,
    export_size: int,
    export_quantization_bit: str,
    export_quantization_dataset: str,
    export_device: str,
    export_legacy_format: bool,
    export_dir: str,
    export_hub_model_id: str,
    extra_args: str,
) -> Generator[str, None, None]:
    user_config = load_config()
    error = ""
    if not model_name:
        error = ALERTS["err_no_model"][lang]
    elif not model_path:
        error = ALERTS["err_no_path"][lang]
    elif not export_dir:
        error = ALERTS["err_no_export_dir"][lang]
    elif export_quantization_bit in GPTQ_BITS and not export_quantization_dataset:
        error = ALERTS["err_no_dataset"][lang]
    elif export_quantization_bit not in GPTQ_BITS and not checkpoint_path:
        error = ALERTS["err_no_adapter"][lang]
    elif export_quantization_bit in GPTQ_BITS and checkpoint_path and isinstance(checkpoint_path, list):
        error = ALERTS["err_gptq_lora"][lang]

    try:
        json.loads(extra_args)
    except json.JSONDecodeError:
        error = ALERTS["err_json_schema"][lang]

    if error:
        gr.Warning(error)
        yield error
        return

    args = dict(
        model_name_or_path=model_path,
        cache_dir=user_config.get("cache_dir", None),
        finetuning_type=finetuning_type,
        template=template,
        export_dir=export_dir,
        export_hub_model_id=export_hub_model_id or None,
        export_size=export_size,
        export_quantization_bit=int(export_quantization_bit) if export_quantization_bit in GPTQ_BITS else None,
        export_quantization_dataset=export_quantization_dataset,
        export_device=export_device,
        export_legacy_format=export_legacy_format,
        trust_remote_code=is_env_enabled("LLAMAFACTORY_WEBUI_TRUST_REMOTE_CODE"),
    )
    args.update(json.loads(extra_args))

    if checkpoint_path:
        if finetuning_type in PEFT_METHODS:  # list
            args["adapter_name_or_path"] = ",".join(
                [get_save_dir(model_name, finetuning_type, adapter) for adapter in checkpoint_path]
            )
        else:  # str
            args["model_name_or_path"] = get_save_dir(model_name, finetuning_type, checkpoint_path)

    yield ALERTS["info_exporting"][lang]
    export_model(args)
    torch_gc()
    yield ALERTS["info_exported"][lang]


def create_export_tab(engine: "Engine") -> dict[str, "Component"]:
    with gr.Row():
        export_size = gr.Slider(minimum=1, maximum=100, value=5, step=1)
        export_quantization_bit = gr.Dropdown(choices=["none"] + GPTQ_BITS, value="none")
        export_quantization_dataset = gr.Textbox(value="data/c4_demo.jsonl")
        export_device = gr.Radio(choices=["cpu", "auto"], value="cpu")
        export_legacy_format = gr.Checkbox()

    with gr.Row():
        export_dir = gr.Textbox()
        export_hub_model_id = gr.Textbox()
        extra_args = gr.Textbox(value="{}")

    checkpoint_path: gr.Dropdown = engine.manager.get_elem_by_id("top.checkpoint_path")
    checkpoint_path.change(can_quantize, [checkpoint_path], [export_quantization_bit], queue=False)

    export_btn = gr.Button()
    info_box = gr.Textbox(show_label=False, interactive=False)

    export_btn.click(
        save_model,
        [
            engine.manager.get_elem_by_id("top.lang"),
            engine.manager.get_elem_by_id("top.model_name"),
            engine.manager.get_elem_by_id("top.model_path"),
            engine.manager.get_elem_by_id("top.finetuning_type"),
            engine.manager.get_elem_by_id("top.checkpoint_path"),
            engine.manager.get_elem_by_id("top.template"),
            export_size,
            export_quantization_bit,
            export_quantization_dataset,
            export_device,
            export_legacy_format,
            export_dir,
            export_hub_model_id,
            extra_args,
        ],
        [info_box],
    )

    return dict(
        export_size=export_size,
        export_quantization_bit=export_quantization_bit,
        export_quantization_dataset=export_quantization_dataset,
        export_device=export_device,
        export_legacy_format=export_legacy_format,
        export_dir=export_dir,
        export_hub_model_id=export_hub_model_id,
        extra_args=extra_args,
        export_btn=export_btn,
        info_box=info_box,
    )
