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

from llamafactory.eval.template import get_eval_template


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
def test_eval_template_en():
    support_set = [
        {
            "question": "Fewshot question",
            "A": "Fewshot1",
            "B": "Fewshot2",
            "C": "Fewshot3",
            "D": "Fewshot4",
            "answer": "B",
        }
    ]
    example = {
        "question": "Target question",
        "A": "Target1",
        "B": "Target2",
        "C": "Target3",
        "D": "Target4",
        "answer": "C",
    }
    template = get_eval_template(name="en")
    messages = template.format_example(example, support_set=support_set, subject_name="SubName")
    assert messages == [
        {
            "role": "user",
            "content": (
                "The following are multiple choice questions (with answers) about SubName.\n\n"
                "Fewshot question\nA. Fewshot1\nB. Fewshot2\nC. Fewshot3\nD. Fewshot4\nAnswer:"
            ),
        },
        {"role": "assistant", "content": "B"},
        {
            "role": "user",
            "content": "Target question\nA. Target1\nB. Target2\nC. Target3\nD. Target4\nAnswer:",
        },
        {"role": "assistant", "content": "C"},
    ]


@pytest.mark.runs_on(["cpu", "mps", "xpu"])
def test_eval_template_zh():
    support_set = [
        {
            "question": "示例问题",
            "A": "示例答案1",
            "B": "示例答案2",
            "C": "示例答案3",
            "D": "示例答案4",
            "answer": "B",
        }
    ]
    example = {
        "question": "目标问题",
        "A": "目标答案1",
        "B": "目标答案2",
        "C": "目标答案3",
        "D": "目标答案4",
        "answer": "C",
    }
    template = get_eval_template(name="zh")
    messages = template.format_example(example, support_set=support_set, subject_name="主题")
    assert messages == [
        {
            "role": "user",
            "content": (
                "以下是中国关于主题考试的单项选择题，请选出其中的正确答案。\n\n"
                "示例问题\nA. 示例答案1\nB. 示例答案2\nC. 示例答案3\nD. 示例答案4\n答案："
            ),
        },
        {"role": "assistant", "content": "B"},
        {
            "role": "user",
            "content": "目标问题\nA. 目标答案1\nB. 目标答案2\nC. 目标答案3\nD. 目标答案4\n答案：",
        },
        {"role": "assistant", "content": "C"},
    ]
