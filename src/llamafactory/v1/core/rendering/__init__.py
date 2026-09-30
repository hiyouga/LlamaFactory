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

"""Rendering: turn a v1 ``Sample`` into a tokenized ``ModelInput``.

Public entry point is :class:`Renderer`. Internals are split by concern:
``format`` (message<->HF conversion) and ``escape`` (special-token escaping). Assistant supervision
is located by a prompt/full token diff rather than a per-model marker table.
"""

from .rendering import Renderer


__all__ = ["Renderer"]
