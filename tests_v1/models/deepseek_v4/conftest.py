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

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast


SPECIALS = ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "<think>", "</think>"]
ORIGINAL_TEMPLATE = "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + '\\n' }}{% endfor %}assistant: "


@pytest.fixture
def tokenizer():
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ["user: hi\nassistant: hello\nassistant: ", "system text analysis answer 中文问题你好 second turn"] * 10,
        trainers.BpeTrainer(
            vocab_size=300, special_tokens=SPECIALS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        ),
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=SPECIALS[0],
        eos_token=SPECIALS[1],
        pad_token=SPECIALS[1],
        additional_special_tokens=SPECIALS[2:],
        chat_template=ORIGINAL_TEMPLATE,
    )
