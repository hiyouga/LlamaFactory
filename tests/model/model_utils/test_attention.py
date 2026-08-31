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

import os

import pytest
from transformers.utils import is_flash_attn_2_available


# Compatible with Transformers v4 and Transformers v5
try:
    from transformers.utils import is_torch_sdpa_available
except ImportError:

    def is_torch_sdpa_available():
        return True


from transformers import PretrainedConfig

from llamafactory.extras.constants import AttentionFunction
from llamafactory.extras.packages import is_transformers_version_greater_than
from llamafactory.hparams import ModelArguments
from llamafactory.model.model_utils.attention import configure_attn_implementation
from llamafactory.train.test_utils import load_infer_model


TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

INFER_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "template": "llama3",
}


@pytest.mark.xfail(is_transformers_version_greater_than("4.48"), reason="Attention refactor.")
def test_attention():
    attention_available = ["disabled"]
    if is_torch_sdpa_available():
        attention_available.append("sdpa")

    if is_flash_attn_2_available():
        attention_available.append("fa2")

    llama_attention_classes = {
        "disabled": "LlamaAttention",
        "sdpa": "LlamaSdpaAttention",
        "fa2": "LlamaFlashAttention2",
    }
    for requested_attention in attention_available:
        model = load_infer_model(flash_attn=requested_attention, **INFER_ARGS)
        for module in model.modules():
            if "Attention" in module.__class__.__name__:
                assert module.__class__.__name__ == llama_attention_classes[requested_attention]


def test_gpt_oss_registers_the_flash_attention_3_kernel(monkeypatch: pytest.MonkeyPatch):
    """The gpt-oss branch imports the kernel registrar by name, with no fallback.

    transformers 5.0 renamed `load_and_register_kernel` to `load_and_register_attn_kernel`, so
    reaching for the wrong one ends every gpt-oss run at load time with an ImportError.
    """
    from transformers.integrations import hub_kernels

    registrar = (
        "load_and_register_attn_kernel"
        if is_transformers_version_greater_than("5.0.0")
        else "load_and_register_kernel"
    )
    registered = []
    monkeypatch.setattr(hub_kernels, registrar, registered.append)

    config = PretrainedConfig(model_type="gpt_oss")
    model_args = ModelArguments(model_name_or_path=TINY_LLAMA3)

    configure_attn_implementation(config, model_args)

    assert registered == ["kernels-community/vllm-flash-attn3"]
    assert config._attn_implementation == "kernels-community/vllm-flash-attn3"
    assert model_args.flash_attn == AttentionFunction.FA3
