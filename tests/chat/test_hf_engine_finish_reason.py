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

from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llamafactory.chat.hf_engine import HuggingfaceEngine
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.hparams.data_args import DataArguments
from llamafactory.hparams.generating_args import GeneratingArguments


TINY_MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


class TestFinishReason:
    def setup_method(self):
        self.tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
        self.template = get_template_and_fix_tokenizer(self.tokenizer, DataArguments(template="chatml"))
        self.stop_token_ids = set(self.template.get_stop_token_ids(self.tokenizer))
        self.messages = [{"role": "user", "content": "hi"}]

    def _run_chat(self, generated_ids):
        paired_messages = self.messages + [{"role": "assistant", "content": ""}]
        prompt_ids, _ = self.template.encode_oneturn(self.tokenizer, paired_messages)
        fake_output = torch.tensor([list(prompt_ids) + list(generated_ids)])
        with patch.object(self.model, "generate", return_value=fake_output):
            results = HuggingfaceEngine._chat(
                self.model,
                self.tokenizer,
                None,
                self.template,
                GeneratingArguments().to_dict(),
                self.messages,
            )
        assert len(results) == 1
        return results[0]

    def test_stops_on_template_stop_token(self):
        primary_eos = self.tokenizer.eos_token_id
        secondary = next(token_id for token_id in self.stop_token_ids if token_id != primary_eos)
        result = self._run_chat([secondary])
        assert result.finish_reason == "stop"
        assert result.response_length == 1

    def test_stops_on_tokenizer_eos(self):
        result = self._run_chat([self.tokenizer.eos_token_id])
        assert result.finish_reason == "stop"
        assert result.response_length == 1

    def test_hits_length_limit(self):
        non_stop_id = next(token_id for token_id in range(len(self.tokenizer)) if token_id not in self.stop_token_ids)
        result = self._run_chat([non_stop_id])
        assert result.finish_reason == "length"
        assert result.response_length == 1
