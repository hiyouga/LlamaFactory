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

"""Special-token escaping (prompt-injection hardening).

Neutralizes control-token strings (``<|im_start|>``, ``<|image_pad|>`` ...) that appear literally
in user-controlled text, so a crafted dataset cannot inject role markers or media placeholders
into the rendered stream. A no-op for normal data.
"""

import json
from typing import Any

from ...utils.types import Message


def _special_token_strings(tokenizer) -> list[str]:
    """Strings the tokenizer encodes to a reserved/special id.

    Such strings must be neutralized if they appear literally in user text. Derived from
    ``added_tokens_decoder`` so it covers every control token of the model (``<|im_start|>``,
    ``<|image_pad|>``, ``<tts_pad>`` ...), not only ``<|...|>``-shaped ones. Sorted longest-first
    so nested matches escape correctly.
    """
    specials = [str(t) for t in tokenizer.added_tokens_decoder.values() if getattr(t, "special", False)]
    return sorted((s for s in specials if len(s) >= 2), key=len, reverse=True)


def _escape_special(text: str, specials: list[str], special_ids: set[int], tokenizer) -> str:
    """Break any special-token string in user text by inserting U+200B after its first char.

    No-op (no tokenization cost) when the text contains no special-token string. When it does,
    self-validate that the result no longer encodes to a special id -- some normalizers strip
    zero-width chars and would resurrect the collision -- and raise if it does.
    """
    if not any(sp in text for sp in specials):
        return text
    out = text
    for sp in specials:
        if sp in out:
            # Insert a zero-width space (U+200B) after the first char to break the exact
            # special-token string match while keeping the text visually/semantically intact.
            out = out.replace(sp, sp[0] + "\u200b" + sp[1:])
    if special_ids.intersection(tokenizer(out, add_special_tokens=False)["input_ids"]):
        raise ValueError(
            "special-token escape failed: the tokenizer normalized away the break char; "
            "user text contains a literal control token that cannot be safely neutralized."
        )
    return out


def _escape_json_strings(value: Any, specials: list[str], special_ids: set[int], tokenizer) -> Any:
    """Escape every string anywhere inside a decoded JSON value, object keys included.

    A tool call is rendered whole by the chat template, so the function name, an argument
    nested in a list or an object, and an argument's key all reach the prompt exactly the way
    a top-level argument string does. Only strings are touched; other JSON types are returned
    as they are.
    """
    if isinstance(value, str):
        return _escape_special(value, specials, special_ids, tokenizer)
    if isinstance(value, dict):
        return {
            (_escape_special(key, specials, special_ids, tokenizer) if isinstance(key, str) else key): (
                _escape_json_strings(item, specials, special_ids, tokenizer)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_escape_json_strings(item, specials, special_ids, tokenizer) for item in value]
    return value


def _escape_special_in_messages(
    messages: list[Message], specials: list[str], special_ids: set[int], tokenizer
) -> list[Message]:
    """Return messages with special-token strings neutralized in user-controlled literal text.

    Covers ``text``/``reasoning`` block values and every string inside a ``tool_call`` object,
    its function name and nested argument containers included.
    """
    if not specials:
        return messages
    escaped: list[Message] = []
    for message in messages:
        new_content = []
        for content in message["content"]:
            if content["type"] in ("text", "reasoning"):
                new_content.append(
                    {**content, "value": _escape_special(content["value"], specials, special_ids, tokenizer)}
                )
            elif content["type"] == "tool_call":
                try:
                    tc = json.loads(content["value"])
                except (json.JSONDecodeError, TypeError):
                    new_content.append(content)
                    continue
                # A tool_call value that is valid JSON but not an object (list/str/int) is rejected
                # later by `_to_hf_messages` with a clearer message -- pass it through untouched
                # rather than reformat something that never reaches a template.
                if isinstance(tc, dict):
                    new_content.append(
                        {**content, "value": json.dumps(_escape_json_strings(tc, specials, special_ids, tokenizer))}
                    )
                else:
                    new_content.append(content)
            else:
                new_content.append(content)
        escaped.append({**message, "content": new_content})
    return escaped
