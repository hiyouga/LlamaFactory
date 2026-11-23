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

"""FSDPTurbo expert-parallel adapter; all context parallelism stays in LF."""

from .engine import FSDPTurboFSDP2Engine
from .gradients import clip_grad_norm_, synchronize_gradients
from .model_specs import FSDPTurboEPModelSpec
from .parallel_state import FSDPTurboParallelState, get_fsdpturbo_parallel_state


__all__ = [
    "FSDPTurboEPModelSpec",
    "FSDPTurboFSDP2Engine",
    "FSDPTurboParallelState",
    "clip_grad_norm_",
    "get_fsdpturbo_parallel_state",
    "synchronize_gradients",
]
