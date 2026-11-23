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

"""Prompt replacements used to build the erased view of CIG-SFT examples."""

from typing import Optional


# The aligned dataset keeps a per-example replacement under this private name so
# that it survives the raw-column removal performed by ``align_dataset``.
ERASED_PROMPT_COLUMN = "_erased_prompt"

ERASED_PROMPT_PRESETS: dict[str, str] = {
    "math": r"[PROBLEM_REMOVED]" "\n" r"Let's think step by step and output the final answer within \boxed{}.",
    "code": "[TASK_REMOVED]",
    "medical": "Question:\n[QUESTION_REMOVED]\n\nOptions:\n[OPTIONS_REMOVED]\n\nAnswer:",
    "default": "[CONTEXT_REMOVED]",
}


def resolve_erased_prompt(value: Optional[str]) -> Optional[str]:
    """Resolve a preset name, or return a non-empty literal prompt unchanged.

    Dataset columns may contain null values. Treating null and blank strings as
    unset lets a row fall back to the dataset-level ``cig_erased_prompt``.
    """
    if not isinstance(value, str) or not value.strip():
        return None

    return ERASED_PROMPT_PRESETS.get(value.strip().lower(), value)
