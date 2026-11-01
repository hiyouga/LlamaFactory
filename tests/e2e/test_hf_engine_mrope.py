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

import pytest
import torch
from PIL import Image

from llamafactory.chat.hf_engine import HuggingfaceEngine
from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.extras.packages import is_transformers_version_greater_than
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer


def _build_tiny_qwen3_vl():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    config = Qwen3VLConfig(
        text_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "rope_parameters": {
                "rope_type": "default",
                "mrope_section": [16, 8, 8],
                "mrope_interleaved": True,
                "rope_theta": 5000000,
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 64,
            "deepstack_visual_indexes": [0],
        },
    )
    torch.manual_seed(0)
    return Qwen3VLForConditionalGeneration(config).eval()


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
@pytest.mark.skipif(not is_transformers_version_greater_than("5.3.0"), reason="Requires transformers>=5.3.0")
def test_hf_engine_passes_mm_token_type_ids():
    model_args, data_args, _, generating_args = get_infer_args(
        {"model_name_or_path": "Qwen/Qwen3-VL-30B-A3B-Instruct", "template": "qwen3_vl"}
    )
    tokenizer_module = load_tokenizer(model_args)
    tokenizer, processor = tokenizer_module["tokenizer"], tokenizer_module["processor"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = _build_tiny_qwen3_vl()

    gen_kwargs, _ = HuggingfaceEngine._process_args(
        model,
        tokenizer,
        processor,
        template,
        generating_args.to_dict(),
        [{"role": "user", "content": "<image>What is in this picture?"}],
        images=[Image.new("RGB", (256, 256), (255, 255, 255))],
        input_kwargs={"do_sample": False, "max_new_tokens": 1},
    )
    input_ids = gen_kwargs["inputs"]
    assert "mm_token_type_ids" in gen_kwargs
    assert torch.equal(gen_kwargs["mm_token_type_ids"] == 1, input_ids == model.config.image_token_id)

    captured = []

    def hook(module, args, kwargs):
        if not captured:
            captured.append(kwargs["position_ids"])

    handle = model.model.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.inference_mode():
            model.generate(**gen_kwargs)
    finally:
        handle.remove()

    expected, _ = model.model.get_rope_index(
        input_ids,
        mm_token_type_ids=gen_kwargs["mm_token_type_ids"],
        image_grid_thw=gen_kwargs["image_grid_thw"],
        attention_mask=gen_kwargs["attention_mask"],
    )
    assert torch.equal(captured[0][-3:], expected)  # prefill must use 3D mrope positions, not text positions
