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

r"""Modern layout for LLaMA Board, enabled with ``LLAMABOARD_UI=modern``.

The classic components are reused unchanged: every control, default value, callback and the task runner
come from ``create_top``, ``create_*_tab`` and ``create_footer``, wired to ``Engine`` exactly as in
``webui/interface.py``. This module only regroups the containers those factories create into pages and
cards before the ``gr.Blocks`` context exits (see ``layout.py``), and adds a sidebar, two visual styles
with light and dark modes, a model overview and relevance hints. Nothing is hidden, disabled or changed
on the user's behalf: groups that do not apply are only dimmed.
"""

import functools
import importlib.util
import json
import os
import platform
from pathlib import Path

from ...data import TEMPLATES
from ...data.template import ReasoningTemplate
from ...extras.constants import MULTIMODAL_SUPPORTED_MODELS, TRAINING_STAGES
from ...extras.env import VERSION
from ...extras.packages import is_gradio_available
from ..common import load_config, save_config
from ..components import (
    create_eval_tab,
    create_export_tab,
    create_footer,
    create_infer_tab,
    create_top,
    create_train_tab,
)
from ..css import CSS
from ..engine import Engine
from . import layout as L
from . import strings as S
from .cssutil import anchor_css
from .model_info import model_card
from .theme import build_theme


if is_gradio_available():
    import gradio as gr


ASSETS = Path(__file__).parent / "assets"
PAGES = ("model", "train", "eval", "infer", "export")
ACCORDIONS = (
    "extra_tab",
    "freeze_tab",
    "lora_tab",
    "rlhf_tab",
    "mm_tab",
    "galore_tab",
    "apollo_tab",
    "badam_tab",
    "swanlab_tab",
)
STYLE_CLASSES = {
    "train.output_box": "lf-log",
    "eval.output_box": "lf-log",
    "train.progress_bar": "lf-prog",
    "eval.progress_bar": "lf-prog",
    "train.loss_viewer": "lf-plot",
    "infer.info_box": "lf-status",
    "export.info_box": "lf-status",
    "train.start_btn": "lf-b-start",
    "eval.start_btn": "lf-b-start",
    "train.stop_btn": "lf-b-stop",
    "eval.stop_btn": "lf-b-stop",
    "train.cmd_preview_btn": "lf-b-aux",
    "eval.cmd_preview_btn": "lf-b-aux",
    "train.arg_save_btn": "lf-b-aux",
    "train.arg_load_btn": "lf-b-aux",
    "footer.device_memory": "lf-vram",
}
# values summarised on the right of each advanced accordion header
SUMMARY_FIELDS = (
    "train.logging_steps",
    "train.save_steps",
    "train.warmup_steps",
    "train.neftune_alpha",
    "train.packing",
    "train.neat_packing",
    "train.train_on_prompt",
    "train.mask_history",
    "train.report_to",
    "train.freeze_trainable_layers",
    "train.freeze_trainable_modules",
    "train.lora_rank",
    "train.lora_alpha",
    "train.lora_dropout",
    "train.loraplus_lr_ratio",
    "train.use_rslora",
    "train.use_dora",
    "train.use_pissa",
    "train.pref_beta",
    "train.pref_ftx",
    "train.image_max_pixels",
    "train.freeze_vision_tower",
    "train.use_galore",
    "train.galore_rank",
    "train.galore_update_interval",
    "train.use_apollo",
    "train.apollo_rank",
    "train.apollo_update_interval",
    "train.use_badam",
    "train.badam_mode",
    "train.badam_switch_mode",
    "train.use_swanlab",
    "train.swanlab_project",
    "train.swanlab_mode",
)
SUMMARY_TEXT = {
    "zh": {
        "log": "日志",
        "save": "保存",
        "warm": "预热",
        "pack": "打包",
        "neat": "无污染打包",
        "prompt": "学提示词",
        "last": "仅末轮",
        "layers": "层",
        "vision": "冻结视觉塔",
        "every": "每 {n} 步",
    },
    "en": {
        "log": "log",
        "save": "save",
        "warm": "warmup",
        "pack": "packing",
        "neat": "neat packing",
        "prompt": "train on prompt",
        "last": "last turn",
        "layers": "layers",
        "vision": "vision frozen",
        "every": "every {n}",
    },
}


def create_modern_ui(demo_mode: bool = False) -> "gr.Blocks":
    _check_gradio()
    engine = Engine(demo_mode=demo_mode, pure_chat=False)
    manager = engine.manager
    pages = [p for p in PAGES if p != "export" or not demo_mode]
    hostname = os.getenv("HOSTNAME", os.getenv("COMPUTERNAME", platform.node())).split(".")[0]
    queries: dict[str, str] = {}
    css = CSS + "\n" + anchor_css((ASSETS / "style.css").read_text(encoding="utf-8"), queries)
    head = (
        '<meta name="color-scheme" content="light dark">\n'
        f"<script>window.LF_MQ = {json.dumps(queries)};</script>\n"
        f"<script>{(ASSETS / 'app.js').read_text(encoding='utf-8')}</script>"
    )
    lang0 = (load_config() or {}).get("lang") or "en"

    heads: dict[str, gr.HTML] = {}  # page key -> page header
    cards: list[tuple[gr.HTML, str, int | None]] = []  # (card header, string key, index)
    stages = []  # holders the classic factories build into; emptied by the moves below

    def card(key: str, index: int | None, *classes: str) -> "gr.Column":
        with gr.Column(elem_classes=["lf-card", *classes]) as col:
            header = gr.HTML(S.card_head_html(lang0, key, index), padding=False, elem_classes="lf-card-top")

        cards.append((header, key, index))
        return col

    def page(key: str) -> "gr.Column":
        return gr.Column(elem_id=f"lf-page-{key}", elem_classes="lf-page")

    def g(elem_id: str):
        return manager.get_elem_by_id(elem_id)

    with gr.Blocks(
        title=f"LLaMA Factory ({hostname})", theme=build_theme(), css=css, head=head, fill_width=True
    ) as demo:
        # sidebar
        with gr.Column(elem_id="lf-nav", scale=0, min_width=0):
            nav = gr.HTML(S.nav_html(lang0, pages, VERSION), padding=False, elem_id="lf-nav-body")
            with gr.Column(elem_id="lf-nav-foot"):
                lang_slot = gr.Column(elem_id="lf-lang-slot")
                appear = gr.HTML(S.appearance_html(lang0), padding=False, elem_id="lf-appear")
                vram_slot = gr.Column(elem_id="lf-vram-slot")

        # main column
        with gr.Column(elem_id="lf-main", scale=1, min_width=0):
            ctx = gr.HTML(elem_id="lf-ctx", padding=False)
            hints = gr.HTML(elem_id="lf-rel", padding=False)
            runstate = gr.HTML(S.run_state_html(lang0, False, None), elem_id="lf-runstate-box", padding=False)

            with page("model"):
                heads["model"] = gr.HTML(S.page_head_html(lang0, "model"), padding=False)
                hero = gr.HTML(
                    S.hero_html(lang0, None, None, None, None, None, 2, 8), padding=False, elem_id="lf-hero-box"
                )
                flow = gr.HTML(S.flow_html(lang0, pages), padding=False, elem_id="lf-flow-box")
                with gr.Row(elem_classes="lf-grid-2", equal_height=False):
                    c_base = card("c_model_base", 1)
                    c_method = card("c_model_method", 2)

                c_runtime = card("c_model_runtime", 3)
                with gr.Column(elem_classes="lf-stage") as stage:
                    manager.add_elems("top", create_top())

                stages.append(stage)

            L.move_component_into(g("top.lang"), lang_slot)
            L.move(L.row_of(g("top.model_name")), c_base)
            L.move(L.row_of(g("top.finetuning_type")), c_method)
            L.move(L.row_of(g("top.quantization_bit")), c_runtime)

            with page("train"):
                heads["train"] = gr.HTML(S.page_head_html(lang0, "train"), padding=False)
                with gr.Row(elem_classes="lf-split", equal_height=False):
                    with gr.Column(elem_classes="lf-split-main", scale=1, min_width=0):
                        c_tdata = card("c_train_data", 1)
                        c_thparams = card("c_train_hparams", 2)
                        c_toutput = card("c_train_output", 3)
                        with gr.Column(elem_classes="lf-adv") as advanced:
                            adv_head = gr.HTML(
                                S.card_head_html(lang0, "c_train_adv", 4), padding=False, elem_classes="lf-adv-top"
                            )

                    with gr.Column(elem_classes="lf-split-side", scale=0, min_width=0):
                        c_trun = card("c_run", None, "lf-run", "lf-run-train")

                with gr.Column(elem_classes="lf-stage") as stage:
                    manager.add_elems("train", create_train_tab(engine))

                stages.append(stage)

            cards.append((adv_head, "c_train_adv", 4))
            L.move(L.row_of(g("train.training_stage")), c_tdata)
            L.move(L.row_of(g("train.learning_rate")), c_thparams)
            L.move(L.row_of(g("train.cutoff_len")), c_thparams)
            L.move(L.row_of(g("train.output_dir")), c_toutput)
            L.move(L.row_of(g("train.device_count")), c_toutput)
            for acc in ACCORDIONS:
                L.move(g("train." + acc), advanced)

            L.move(L.row_of(g("train.start_btn")), c_trun)
            L.move(L.row_of(g("train.progress_bar")), c_trun)
            L.move(L.parent_of(g("train.loss_viewer")), c_trun)
            L.move(L.row_of(g("train.output_box")), c_trun)

            with page("eval"):
                heads["eval"] = gr.HTML(S.page_head_html(lang0, "eval"), padding=False)
                with gr.Row(elem_classes="lf-split", equal_height=False):
                    with gr.Column(elem_classes="lf-split-main", scale=1, min_width=0):
                        c_edata = card("c_eval_data", 1)
                        c_egen = card("c_eval_gen", 2)

                    with gr.Column(elem_classes="lf-split-side", scale=0, min_width=0):
                        c_erun = card("c_run", None, "lf-run", "lf-run-eval")

                with gr.Column(elem_classes="lf-stage") as stage:
                    manager.add_elems("eval", create_eval_tab(engine))

                stages.append(stage)

            L.move(L.row_of(g("eval.dataset_dir")), c_edata)
            L.move(L.row_of(g("eval.cutoff_len")), c_egen)
            L.move(L.row_of(g("eval.max_new_tokens")), c_egen)
            L.move(L.row_of(g("eval.start_btn")), c_erun)
            L.move(L.row_of(g("eval.progress_bar")), c_erun)
            L.move(L.row_of(g("eval.output_box")), c_erun)

            with page("infer") as p_infer:
                heads["infer"] = gr.HTML(S.page_head_html(lang0, "infer"), padding=False)
                c_engine = card("c_infer_engine", 1)
                with gr.Column(elem_classes="lf-stage") as stage:
                    manager.add_elems("infer", create_infer_tab(engine))

                stages.append(stage)

            L.move(L.row_of(g("infer.infer_backend")), c_engine)
            L.move(L.row_of(g("infer.load_btn")), c_engine)
            L.move(L.parent_of(g("infer.info_box")), c_engine)
            L.move(g("infer.chat_box"), p_infer)

            if "export" in pages:
                with page("export"):
                    heads["export"] = gr.HTML(S.page_head_html(lang0, "export"), padding=False)
                    c_xopts = card("c_export_opts", 1)
                    c_xrun = card("c_export_run", 2, "lf-actions-card")
                    with gr.Column(elem_classes="lf-stage") as stage:
                        manager.add_elems("export", create_export_tab(engine))

                    stages.append(stage)

                L.move(L.row_of(g("export.export_size")), c_xopts)
                L.move(L.row_of(g("export.export_dir")), c_xopts)
                L.move(g("export.export_btn"), c_xrun)
                L.move(L.parent_of(g("export.info_box")), c_xrun)

            # shared footer: the VRAM readout lives in the sidebar
            with gr.Column(elem_classes="lf-stage") as stage:
                manager.add_elems("footer", create_footer())

            stages.append(stage)
            L.move(L.row_of(g("footer.device_memory")), vram_slot)

            # title / subtitle are updated by change_lang but not shown in this layout
            with gr.Column(visible=False, elem_id="lf-stash"):
                title = gr.HTML(visible=False)
                subtitle = gr.HTML(visible=False)

            manager.add_elems("head", {"title": title, "subtitle": subtitle})

        # dataset preview dialogs are position:fixed; keep them outside every card and page
        with gr.Column(elem_id="lf-overlays") as overlays:
            pass

        for tab in ("train", "eval"):
            modal = L.parent_of(g(f"{tab}.preview_count"), "Column")
            if modal is not None and "modal-box" in (getattr(modal, "elem_classes", None) or []):
                L.move(modal, overlays)

        # anything the factories add that this layout does not place stays visible, never lost
        for stage in stages:
            L.prune(stage)
            if L.has_component(stage):
                L.add_classes(stage, "lf-stage-left")
            else:
                L.detach(stage)

        problems = L.check_reachable(demo, manager.get_elem_list())
        if problems:
            raise RuntimeError("Modern layout lost or duplicated components:\n" + "\n".join(problems[:20]))

        # styling hooks
        for elem in manager.get_elem_list():
            elem_id = manager.get_id_by_elem(elem)
            if not getattr(elem, "elem_id", None):
                elem.elem_id = "lfp-" + elem_id.replace(".", "-")

        for elem_id, cls in STYLE_CLASSES.items():
            if elem_id.split(".")[0] in pages + ["footer"]:
                L.add_classes(g(elem_id), cls)

        for elem_id in ("train.start_btn", "eval.start_btn"):
            L.add_classes(L.row_of(g(elem_id)), "lf-actions")

        for acc in ACCORDIONS:
            L.add_classes(g("train." + acc), "lf-acc")

        # events of the classic interface
        lang = g("top.lang")
        demo.load(engine.resume, outputs=manager.get_elem_list(), concurrency_limit=None)
        lang.change(engine.change_lang, [lang], manager.get_elem_list(), queue=False)
        lang.input(save_config, inputs=[lang], queue=False)

        # localized chrome
        page_keys = [k for k in pages if k in heads]
        chrome_out = [nav, appear, flow] + [heads[k] for k in page_keys] + [h for h, _, _ in cards]

        def chrome_update(lg):
            out = [
                gr.HTML(S.nav_html(lg, pages, VERSION)),
                gr.HTML(S.appearance_html(lg)),
                gr.HTML(S.flow_html(lg, pages)),
            ]
            out += [gr.HTML(S.page_head_html(lg, k)) for k in page_keys]
            out += [gr.HTML(S.card_head_html(lg, key, ix)) for _, key, ix in cards]
            return out

        demo.load(chrome_update, [lang], chrome_out, queue=False)
        lang.change(chrome_update, [lang], chrome_out, queue=False)

        # context strip
        ctx_in = [
            lang,
            g("top.model_name"),
            g("top.finetuning_type"),
            g("top.template"),
            g("top.quantization_bit"),
            g("top.checkpoint_path"),
        ]
        demo.load(S.ctx_html, ctx_in, ctx, queue=False)
        for src in ctx_in:
            src.change(S.ctx_html, ctx_in, ctx, queue=False)

        # model overview
        hero_in = [
            lang,
            g("top.model_name"),
            g("top.model_path"),
            g("top.finetuning_type"),
            g("top.quantization_bit"),
            g("train.freeze_trainable_layers"),
            g("train.lora_rank"),
        ]

        def hero_update(lg, name, path, ft, qbit, freeze_k, rank):
            info = model_card(name, path, rank) if name else None
            return S.hero_html(lg, info, name, path, ft, qbit, freeze_k, rank)

        demo.load(hero_update, hero_in, hero, queue=False)
        for src in hero_in:
            src.change(hero_update, hero_in, hero, queue=False)

        # live run state (reads Runner.running, changes nothing)
        timer = gr.Timer(2.0)

        def run_state(lg):
            runner = engine.runner
            kind = ("train" if runner.do_train else "eval") if runner.running else None
            return S.run_state_html(lg, bool(runner.running), kind)

        timer.tick(run_state, [lang], runstate, queue=False, show_progress="hidden")

        # relevance hints and accordion summaries
        hint_keys = [
            "top.lang",
            "train.training_stage",
            "top.finetuning_type",
            "top.model_name",
            "train.use_llama_pro",
            "train.ds_stage",
            "top.template",
            "top.booster",
            "infer.infer_backend",
        ]
        hint_keys += ["export.export_quantization_bit"] if "export" in pages else []
        hint_keys += list(SUMMARY_FIELDS)
        hint_in = [g(k) for k in hint_keys]
        hint_fn = functools.partial(_hints_html, hint_keys)
        demo.load(hint_fn, hint_in, hints, queue=False)
        for src in hint_in:
            src.change(hint_fn, hint_in, hints, queue=False)

    return demo


def _check_gradio() -> None:
    from packaging import version

    if version.parse(gr.__version__) < version.parse("5.0.0"):
        raise RuntimeError(
            f"The modern LLaMA Board layout requires gradio>=5.0.0, found {gr.__version__}. "
            "Unset LLAMABOARD_UI to use the classic layout."
        )


@functools.cache
def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


@functools.cache
def _trl_ppo() -> tuple[str, bool]:
    r"""Return the installed TRL version and whether PPO accepts it (see ``train/ppo/trainer.py``)."""
    try:
        import trl
        from packaging import version

        found = version.parse(trl.__version__)
        return trl.__version__, version.parse("0.8.6") <= found <= version.parse("0.9.6")
    except Exception:
        return "", True


def _css_str(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _fmt(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return str(value)


def _summaries(lang: str, v: dict) -> dict[str, str]:
    text = SUMMARY_TEXT["zh" if lang == "zh" else "en"]
    off = S.t(lang, "off")
    parts = [
        f"{text['log']} {_fmt(v['logging_steps'])}",
        f"{text['save']} {_fmt(v['save_steps'])}",
        f"{text['warm']} {_fmt(v['warmup_steps'])}",
    ]
    if v["neftune_alpha"]:
        parts.append(f"NEFTune {_fmt(v['neftune_alpha'])}")

    if v["neat_packing"]:
        parts.append(text["neat"])
    elif v["packing"]:
        parts.append(text["pack"])

    if v["train_on_prompt"]:
        parts.append(text["prompt"])

    if v["mask_history"]:
        parts.append(text["last"])

    if v["report_to"] not in (None, "none", []):
        parts.append(str(v["report_to"]))

    lora = [f"r {_fmt(v['lora_rank'])}", f"α {_fmt(v['lora_alpha'])}"]
    if v["lora_dropout"]:
        lora.append(f"dropout {_fmt(v['lora_dropout'])}")

    if v["loraplus_lr_ratio"]:
        lora.append(f"LoRA+ {_fmt(v['loraplus_lr_ratio'])}")

    lora += [n for n, on in (("rsLoRA", v["use_rslora"]), ("DoRA", v["use_dora"]), ("PiSSA", v["use_pissa"])) if on]
    mm = [f"≤ {v['image_max_pixels']}"] + ([text["vision"]] if v["freeze_vision_tower"] else [])
    every = text["every"]
    return {
        "extra_tab": " · ".join(parts),
        "freeze_tab": f"{_fmt(v['freeze_trainable_layers'])} {text['layers']} · {v['freeze_trainable_modules'] or 'all'}",
        "lora_tab": " · ".join(lora),
        "rlhf_tab": f"β {_fmt(v['pref_beta'])} · ftx {_fmt(v['pref_ftx'])}",
        "mm_tab": " · ".join(mm),
        "galore_tab": (
            f"r {_fmt(v['galore_rank'])} · {every.format(n=_fmt(v['galore_update_interval']))}"
            if v["use_galore"]
            else off
        ),
        "apollo_tab": (
            f"r {_fmt(v['apollo_rank'])} · {every.format(n=_fmt(v['apollo_update_interval']))}"
            if v["use_apollo"]
            else off
        ),
        "badam_tab": f"{v['badam_mode']} · {v['badam_switch_mode']}" if v["use_badam"] else off,
        "swanlab_tab": f"{v['swanlab_project']} · {v['swanlab_mode']}" if v["use_swanlab"] else off,
    }


def _hints_html(keys: list[str], *values) -> str:
    r"""Dim what does not apply, summarise the advanced groups and warn about missing packages."""
    v = {k.split(".", 1)[1]: val for k, val in zip(keys, values)}
    lang = v["lang"]
    stage = TRAINING_STAGES.get(v["training_stage"], "sft")
    ft, llama_pro = v["finetuning_type"], v["use_llama_pro"]
    groups, fields, banners = {}, [], []
    if ft != "lora":
        groups["lora_tab"] = "n_only_lora"

    if not (ft == "freeze" or (ft == "lora" and llama_pro)):
        groups["freeze_tab"] = "n_only_freeze"

    if stage not in ("ppo", "dpo", "kto"):
        groups["rlhf_tab"] = "n_only_pref"
    elif stage == "ppo":
        fields += ["train-pref_beta", "train-pref_ftx", "train-pref_loss"]
    else:
        fields += ["train-reward_model", "train-ppo_score_norm", "train-ppo_whiten_rewards"]
        if stage == "kto":
            fields.append("train-pref_loss")

    if v["model_name"] not in MULTIMODAL_SUPPORTED_MODELS:
        groups["mm_tab"] = "n_only_mm"

    if ft == "lora":
        for acc in ("galore_tab", "apollo_tab", "badam_tab"):
            groups[acc] = "n_no_lora"

    if stage != "sft":
        fields += ["train-neat_packing", "train-train_on_prompt", "train-mask_history"]

    if stage == "ppo":
        fields.append("train-val_size")

    if v["ds_stage"] in (None, "", "none"):
        fields.append("train-ds_offload")

    template = TEMPLATES.get(v["template"])
    if template is not None and not isinstance(template, ReasoningTemplate):
        fields += ["train-enable_thinking", "infer-enable_thinking"]

    trl_version, ppo_ok = _trl_ppo()
    if stage == "ppo" and not ppo_ok:
        banners.append(("train", S.t(lang, "b_ppo").format(v=trl_version)))

    # options whose package is missing: the run would stop with "To fix: pip install ..."
    missing = []
    boosters = {"unsloth": ("unsloth", "unsloth"), "liger_kernel": ("liger_kernel", "liger-kernel")}
    if v.get("booster") in boosters and not _has(boosters[v["booster"]][0]):
        missing.append(("model train eval infer", v["booster"], boosters[v["booster"]][1]))

    if v.get("booster") == "flashattn2" and not _has("flash_attn"):
        banners.append(("model train eval infer", S.t(lang, "b_fa2")))

    for flag, module, pip in (
        ("use_galore", "galore_torch", "galore-torch"),
        ("use_apollo", "apollo_torch", "apollo-torch"),
        ("use_badam", "badam", "badam"),
    ):
        if v.get(flag) and not _has(module):
            missing.append(("train", flag, pip))

    if v.get("ds_stage") not in (None, "", "none") and not _has("deepspeed"):
        missing.append(("train", "DeepSpeed", "deepspeed"))

    if v.get("report_to") in ("tensorboard", "all") and not _has("tensorboard"):
        missing.append(("train", "tensorboard", "tensorboard"))

    if v.get("infer_backend") in ("vllm", "sglang") and not _has(v["infer_backend"]):
        missing.append(("infer", v["infer_backend"], v["infer_backend"]))

    if v.get("export_quantization_bit") not in (None, "", "none") and not _has("gptqmodel"):
        missing.append(("export", "GPTQ", "gptqmodel"))

    metric_pkgs = [
        pip
        for module, pip in (("jieba", "jieba"), ("nltk", "nltk"), ("rouge_chinese", "rouge-chinese"))
        if not _has(module)
    ]
    if metric_pkgs:  # the board always evaluates with predict_with_generate, see runner.py
        missing.append(("eval", "BLEU / ROUGE", " ".join(metric_pkgs)))

    for page_keys, what, pip in missing:
        banners.append((page_keys, S.t(lang, "b_pkg").format(what=what, pip=pip)))

    try:
        summaries = _summaries(lang, v)
    except (KeyError, TypeError, ValueError):
        summaries = {}

    css = []
    for acc in ACCORDIONS:
        selector = f"#lfp-train-{acc}>.label-wrap>span:first-child::after"
        if acc in groups:
            css.append(
                f"#lfp-train-{acc}>.label-wrap{{opacity:.55}}"
                f"{selector}{{content:{_css_str(S.t(lang, groups[acc]))}}}"
                f"#lfp-train-{acc}{{--lf-chip-bg:transparent;--lf-chip-fg:var(--lf-text-3)}}"
            )
        elif summaries.get(acc):
            css.append(f"{selector}{{content:{_css_str(summaries[acc])}}}")

    if fields:
        css.append(",".join(f"#lfp-{x}" for x in sorted(set(fields))) + "{opacity:.42}")

    out = "<style>" + "".join(css) + "</style>"
    for page_keys, text in banners:
        out += f'<div class="lf-banner" role="note" data-page="{page_keys}">{S.esc(text)}</div>'

    return out
