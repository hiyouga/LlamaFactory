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

from llamafactory.hparams import ModelArguments


@pytest.mark.parametrize("field", ["adapter_name_or_path", "add_tokens", "add_special_tokens"])
def test_copyfrom_preserves_normalized_lists(field: str):
    source = ModelArguments(model_name_or_path="local-model", **{field: "first, second"})
    assert getattr(source, field) == ["first", "second"]

    copied = ModelArguments.copyfrom(source)

    assert copied is not source
    assert getattr(copied, field) == ["first", "second"]
    assert copied.to_dict() == source.to_dict()


def test_copyfrom_without_lists():
    source = ModelArguments(model_name_or_path="local-model")

    copied = ModelArguments.copyfrom(source)

    assert copied is not source
    assert copied.to_dict() == source.to_dict()


def test_copyfrom_reference_model_overrides_preserve_tokens():
    source = ModelArguments(
        model_name_or_path="local-model",
        adapter_name_or_path="local-adapter",
        add_tokens="alpha, beta",
        add_special_tokens="<first>, <second>",
    )

    copied = ModelArguments.copyfrom(source, model_name_or_path="local-reference", adapter_name_or_path=None)

    assert copied.model_name_or_path == "local-reference"
    assert copied.adapter_name_or_path is None
    assert copied.add_tokens == ["alpha", "beta"]
    assert copied.add_special_tokens == ["<first>", "<second>"]
    assert source.model_name_or_path == "local-model"
    assert source.adapter_name_or_path == ["local-adapter"]


if __name__ == "__main__":
    pytest.main([__file__])
