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

r"""Gradio theme whose variables all point at the ``--lf-*`` design tokens in ``assets/style.css``.

Both visual styles and both modes (light / dark) are expressed purely as token sets in CSS, so every
Gradio variable maps to the same token in light and dark mode and the tokens do the switching.
No web fonts are fetched.
"""

import inspect

from ...extras.packages import is_gradio_available


if is_gradio_available():
    import gradio as gr


# gradio variable -> token expression (used for both light and dark)
_TOKENS = {
    "body_background_fill": "transparent",  # page colour/pattern is painted on html/body by style.css
    "body_text_color": "var(--lf-text)",
    "body_text_color_subdued": "var(--lf-text-3)",
    "background_fill_primary": "var(--lf-popover)",
    "background_fill_secondary": "var(--lf-surface-2)",
    "border_color_primary": "var(--lf-line)",
    "border_color_accent": "var(--lf-accent)",
    "border_color_accent_subdued": "var(--lf-accent-soft)",
    "color_accent": "var(--lf-accent)",
    "color_accent_soft": "var(--lf-accent-soft)",
    "block_background_fill": "transparent",
    "block_border_color": "transparent",
    "block_label_background_fill": "transparent",
    "block_label_border_color": "transparent",
    "block_label_text_color": "var(--lf-text-2)",
    "block_title_text_color": "var(--lf-text-2)",
    "block_info_text_color": "var(--lf-text-3)",
    "panel_background_fill": "var(--lf-surface-2)",
    "panel_border_color": "var(--lf-line)",
    "input_background_fill": "var(--lf-input)",
    "input_background_fill_hover": "var(--lf-input)",
    "input_background_fill_focus": "var(--lf-input-focus)",
    "input_border_color": "var(--lf-input-line)",
    "input_border_color_hover": "var(--lf-input-line-hover)",
    "input_border_color_focus": "var(--lf-accent)",
    "input_placeholder_color": "var(--lf-text-4)",
    "input_shadow_focus": "0 0 0 3px var(--lf-ring)",
    "checkbox_background_color": "var(--lf-input)",
    "checkbox_background_color_hover": "var(--lf-input)",
    "checkbox_background_color_focus": "var(--lf-input)",
    "checkbox_background_color_selected": "var(--lf-accent)",
    "checkbox_border_color": "var(--lf-input-line)",
    "checkbox_border_color_hover": "var(--lf-input-line-hover)",
    "checkbox_border_color_focus": "var(--lf-accent)",
    "checkbox_border_color_selected": "var(--lf-accent)",
    "checkbox_label_background_fill": "var(--lf-input)",
    "checkbox_label_background_fill_hover": "var(--lf-hover)",
    "checkbox_label_background_fill_selected": "var(--lf-accent-soft)",
    "checkbox_label_border_color": "var(--lf-input-line)",
    "checkbox_label_border_color_hover": "var(--lf-input-line-hover)",
    "checkbox_label_border_color_selected": "var(--lf-accent)",
    "checkbox_label_text_color": "var(--lf-text)",
    "checkbox_label_text_color_selected": "var(--lf-accent-text)",
    "button_primary_background_fill": "var(--lf-btn-primary)",
    "button_primary_background_fill_hover": "var(--lf-btn-primary-hover)",
    "button_primary_border_color": "transparent",
    "button_primary_border_color_hover": "transparent",
    "button_primary_text_color": "var(--lf-btn-primary-text)",
    "button_primary_text_color_hover": "var(--lf-btn-primary-text)",
    "button_primary_shadow": "var(--lf-btn-primary-shadow)",
    "button_primary_shadow_hover": "var(--lf-btn-primary-shadow)",
    "button_primary_shadow_active": "none",
    "button_secondary_background_fill": "var(--lf-btn)",
    "button_secondary_background_fill_hover": "var(--lf-btn-hover)",
    "button_secondary_border_color": "var(--lf-btn-line)",
    "button_secondary_border_color_hover": "var(--lf-btn-line)",
    "button_secondary_text_color": "var(--lf-text)",
    "button_secondary_text_color_hover": "var(--lf-text)",
    "button_secondary_shadow": "var(--lf-btn-shadow)",
    "button_secondary_shadow_hover": "var(--lf-btn-shadow)",
    "button_secondary_shadow_active": "none",
    "button_cancel_background_fill": "var(--lf-danger-bg)",
    "button_cancel_background_fill_hover": "var(--lf-danger-bg-hover)",
    "button_cancel_border_color": "var(--lf-danger-line)",
    "button_cancel_border_color_hover": "var(--lf-danger-line)",
    "button_cancel_text_color": "var(--lf-danger-text)",
    "button_cancel_text_color_hover": "var(--lf-danger-text)",
    "button_cancel_shadow": "none",
    "button_cancel_shadow_hover": "none",
    "button_cancel_shadow_active": "none",
    "slider_color": "var(--lf-accent)",
    "loader_color": "var(--lf-accent)",
    "link_text_color": "var(--lf-accent-text)",
    "link_text_color_hover": "var(--lf-accent-text)",
    "link_text_color_active": "var(--lf-accent-text)",
    "link_text_color_visited": "var(--lf-accent-text)",
    "code_background_fill": "var(--lf-code-bg)",
    "table_border_color": "var(--lf-line)",
    "table_even_background_fill": "transparent",
    "table_odd_background_fill": "var(--lf-zebra)",
    "table_row_focus": "var(--lf-accent-soft)",
    "table_text_color": "var(--lf-text)",
    "accordion_text_color": "var(--lf-text)",
    "stat_background_fill": "var(--lf-accent-soft)",
    "error_background_fill": "var(--lf-danger-bg)",
    "error_border_color": "var(--lf-danger-line)",
    "error_text_color": "var(--lf-danger-text)",
    "error_icon_color": "var(--lf-danger-text)",
}

# dimension-like variables without a dark variant
_PLAIN = {
    "block_border_width": "0px",
    "block_shadow": "none",
    "block_radius": "0px",  # blocks are transparent wrappers; rounding them clipped text
    "block_padding": "0px",
    "block_label_text_size": "13px",
    "block_label_text_weight": "550",
    "block_title_text_size": "13px",
    "block_title_text_weight": "550",
    "block_info_text_size": "12px",
    "block_info_text_weight": "400",
    "container_radius": "var(--lf-r-md)",
    "input_radius": "var(--lf-r-sm)",
    "input_border_width": "1px",
    "input_padding": "8px 10px",
    "input_text_size": "13.5px",
    "input_shadow": "none",
    "button_border_width": "1px",
    "button_large_radius": "var(--lf-r-btn)",
    "button_medium_radius": "var(--lf-r-btn)",
    "button_small_radius": "var(--lf-r-btn)",
    "button_large_padding": "9px 18px",
    "button_medium_padding": "8px 16px",
    "button_small_padding": "6px 12px",
    "button_large_text_size": "13.5px",
    "button_medium_text_size": "13.5px",
    "button_small_text_size": "12.5px",
    "button_large_text_weight": "560",
    "button_medium_text_weight": "560",
    "button_small_text_weight": "520",
    "button_transition": "background-color .15s ease, border-color .15s ease, box-shadow .15s ease, color .15s ease",
    "checkbox_border_radius": "5px",
    "checkbox_label_text_size": "13px",
    "checkbox_label_gap": "8px",
    "checkbox_label_padding": "6px 10px",
    "layout_gap": "12px",
    "form_gap_width": "0px",
    "embed_radius": "var(--lf-r-md)",
    "table_radius": "var(--lf-r-sm)",
    "chatbot_text_size": "14px",
    "body_text_size": "14px",
    "prose_text_size": "14.5px",
    "prose_header_text_weight": "650",
    "section_header_text_size": "13px",
    "section_header_text_weight": "600",
}


def build_theme() -> "gr.themes.Base":
    font = gr.themes.Font  # local font stacks only; GoogleFont would fetch from a CDN
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.indigo,
        neutral_hue=gr.themes.colors.slate,
        font=[
            font("Segoe UI Variable Text"),
            font("Segoe UI"),
            font("Noto Sans SC"),
            font("Microsoft YaHei UI"),
            font("PingFang SC"),
            font("Hiragino Sans"),
            font("Malgun Gothic"),
            font("system-ui"),
            font("sans-serif"),
        ],
        font_mono=[
            font("JetBrains Mono"),
            font("Cascadia Mono"),
            font("Consolas"),
            font("SFMono-Regular"),
            font("ui-monospace"),
            font("monospace"),
        ],
    )
    accepted = set(inspect.signature(theme.set).parameters)
    values = dict(_PLAIN)
    for key, token in _TOKENS.items():
        values[key] = token
        if key + "_dark" in accepted:  # a few variables (e.g. color_accent) have no dark twin
            values[key + "_dark"] = token

    return theme.set(**{k: v for k, v in values.items() if k in accepted})
