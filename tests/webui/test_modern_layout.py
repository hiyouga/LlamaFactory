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

from llamafactory.extras.packages import is_gradio_available
from llamafactory.webui.modern.cssutil import anchor_css


@pytest.mark.runs_on(["cpu", "mps"])
def test_anchor_css():
    queries = {}
    css = anchor_css(
        ".a, :root[data-x] { color: red; } @media (max-width: 900px) { .b { color: blue; } } "
        "@keyframes spin { from { opacity: 0; } }",
        queries,
    )
    assert queries == {"mq0": "(max-width: 900px)"}
    assert "html .a, :root[data-x] {" in css
    assert 'html[data-lf-mq~="mq0"] .b {' in css
    assert "@media" not in css
    assert "from {" in css


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.skipif(not is_gradio_available(), reason="Gradio is not installed.")
def test_create_modern_ui():
    import gradio as gr

    if int(gr.__version__.split(".")[0]) < 5:
        pytest.skip("The modern layout requires Gradio 5.")

    from llamafactory.webui.modern import create_modern_ui

    demo = create_modern_ui()  # raises if the layout loses or duplicates a component
    assert isinstance(demo, gr.Blocks)
