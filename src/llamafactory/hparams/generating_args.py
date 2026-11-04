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

from dataclasses import asdict, dataclass, field
from typing import Any

from transformers import GenerationConfig


@dataclass
class GeneratingArguments:
    r"""Arguments pertaining to specify the decoding parameters."""

    do_sample: bool = field(
        default=True,
        metadata={"help": "Whether or not to use sampling, use greedy decoding otherwise."},
    )
    temperature: float = field(
        default=0.95,
        metadata={"help": "The value used to modulate the next token probabilities."},
    )
    top_p: float = field(
        default=0.7,
        metadata={
            "help": (
                "The smallest set of most probable tokens with probabilities that add up to top_p or higher are kept."
            )
        },
    )
    top_k: int = field(
        default=50,
        metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k filtering."},
    )
    num_beams: int = field(
        default=1,
        metadata={"help": "Number of beams for beam search. 1 means no beam search."},
    )
    max_length: int = field(
        default=1024,
        metadata={"help": "The maximum length the generated tokens can have. It can be overridden by max_new_tokens."},
    )
    max_new_tokens: int = field(
        default=1024,
        metadata={"help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."},
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "The parameter for repetition penalty. 1.0 means no penalty."},
    )
    length_penalty: float = field(
        default=1.0,
        metadata={"help": "Exponential penalty to the length that is used with beam-based generation."},
    )
    skip_special_tokens: bool = field(
        default=True,
        metadata={"help": "Whether or not to remove special tokens in the decoding."},
    )

    def __post_init__(self) -> None:
        if not 0 < self.top_p <= 1:
            raise ValueError("`top_p` must be in the interval (0, 1], got {0}.".format(self.top_p))
        if self.repetition_penalty <= 0:
            raise ValueError(
                "`repetition_penalty` must be a strictly positive float, got {0}.".format(self.repetition_penalty)
            )

    def to_dict(self, obey_generation_config: bool = False) -> dict[str, Any]:
        args = asdict(self)
        if args.get("max_new_tokens", -1) > 0:
            args.pop("max_length", None)
        else:
            args.pop("max_new_tokens", None)

        if obey_generation_config:
            generation_config = GenerationConfig()
            for key in list(args.keys()):
                if not hasattr(generation_config, key):
                    args.pop(key)

        return args
