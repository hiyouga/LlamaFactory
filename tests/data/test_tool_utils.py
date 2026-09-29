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

from llamafactory.data.tool_utils import MiniMaxM1ToolUtils, MiniMaxM2ToolUtils


def test_minimax_m1_parses_valid_tool_call():
    content = '<tool_calls>\n{"name": "get_weather", "arguments": {"city": "NYC"}}\n</tool_calls>'
    result = MiniMaxM1ToolUtils.tool_extractor(content)
    assert isinstance(result, list) and len(result) == 1
    assert result[0].name == "get_weather"


def test_minimax_m1_returns_content_when_no_valid_tool_call():
    # Wrapper tag is present but its body is not valid JSON, so nothing parses.
    # The extractor should return the original content, not an empty list.
    content = "<tool_calls>\nthis is not json\n</tool_calls>"
    assert MiniMaxM1ToolUtils.tool_extractor(content) == content


def test_minimax_m1_empty_wrapper_returns_content():
    content = "<tool_calls>\n\n</tool_calls>"
    assert MiniMaxM1ToolUtils.tool_extractor(content) == content


def test_minimax_m2_returns_content_when_no_valid_tool_call():
    # Wrapper tag is present but there is no well-formed <invoke> block.
    content = "<minimax:tool_call>\nno invoke block here\n</minimax:tool_call>"
    assert MiniMaxM2ToolUtils.tool_extractor(content) == content


def test_minimax_m2_empty_wrapper_returns_content():
    content = "<minimax:tool_call>\n\n</minimax:tool_call>"
    assert MiniMaxM2ToolUtils.tool_extractor(content) == content
