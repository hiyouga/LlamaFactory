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

r"""Anchor every selector of the modern stylesheet at ``html``.

Gradio 5 injects custom CSS twice: as written, and again with each selector prefixed by
``.gradio-container.gradio-container-<version> .contain``. That second copy carries extra classes of
specificity, so a plain base rule such as ``.lf-eyebrow {display: flex}`` silently beats a theme
override such as ``html[data-lf-style="soft"] .lf-eyebrow {display: inline-flex}``.

Starting every selector with ``html`` makes the prefixed copy unmatchable (no ``html`` element lives
inside ``.contain``), so only the copy written here applies and the cascade behaves as authored.

Gradio re-emits ``@media`` blocks only in their prefixed form, which would leave every responsive rule
dead. Top-level ``@media`` blocks are therefore rewritten into plain rules scoped by
``html[data-lf-mq~="mqN"]``; the browser script mirrors each media condition into that attribute.
Selectors inside ``@keyframes`` are left alone.
"""

import re


_COMMENT = re.compile(r"/\*.*?\*/", re.S)


def _split_selectors(prelude: str) -> list[str]:
    parts, depth, cur = [], 0, []
    for ch in prelude:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1

        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)

    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def _anchor(selector: str, scope: str = "") -> str:
    match = re.match(r"^(html|:root)(?![\w-])", selector)
    if match:
        return "html" + scope + selector[match.end() :] if scope else selector

    return "html" + scope + " " + selector


def anchor_css(css: str, queries: dict[str, str] | None = None) -> str:
    r"""Anchor selectors at ``html``.

    Args:
        css: the stylesheet.
        queries: if given, each top-level ``@media`` block is rewritten into rules scoped by
            ``html[data-lf-mq~="mqN"]`` and its condition is stored as ``queries["mqN"]``.

    Returns:
        The rewritten stylesheet.
    """
    css = _COMMENT.sub("", css)
    out: list[str] = []
    stack: list[str] = []  # kinds of open blocks: "at-group", "keyframes", "media", "rule"
    scope = ""  # attribute selector of the @media block being rewritten
    buf: list[str] = []
    for ch in css:
        if ch == "{":
            prelude = "".join(buf).strip()
            buf = []
            if prelude.startswith("@media") and queries is not None and not stack:
                key = f"mq{len(queries)}"
                queries[key] = prelude[len("@media") :].strip()
                scope = f'[data-lf-mq~="{key}"]'
                kind = "media"
            elif prelude.startswith("@"):
                kind = "keyframes" if re.match(r"@(-\w+-)?keyframes", prelude) else "at-group"
                out.append(prelude + " {")
            elif stack and stack[-1] == "keyframes":
                kind = "rule"
                out.append(prelude + " {")
            else:
                kind = "rule"
                out.append(", ".join(_anchor(s, scope) for s in _split_selectors(prelude)) + " {")

            stack.append(kind)
        elif ch == "}":
            if (stack and stack[-1] == "rule") or "".join(buf).strip():
                out.append("".join(buf))

            buf = []
            kind = stack.pop() if stack else ""
            if kind == "media":
                scope = ""
            else:
                out.append("}\n")
        elif ch == ";" and not stack:
            out.append("".join(buf) + ";\n")  # top-level @import / @charset
            buf = []
        else:
            buf.append(ch)

    return "".join(out)
