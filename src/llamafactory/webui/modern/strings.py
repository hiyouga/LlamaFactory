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

r"""Text and HTML fragments owned by the modern layout.

Parameter labels always come from ``webui/locales.py``; only navigation, page and card headings, the
appearance controls, relevance notes and the model overview live here. Missing keys fall back to
English, then Chinese.
"""

import html
import json


T = {
    "zh": {
        "setup": "配置",
        "workspace": "工作区",
        "model": "模型",
        "train": "训练",
        "eval": "评估与预测",
        "infer": "对话",
        "export": "导出",
        "model_d": "训练、评估、对话与导出共用这一份模型配置，只需设置一次。",
        "train_d": "预训练、监督微调、奖励建模，以及 DPO / KTO / PPO 偏好对齐。",
        "eval_d": "在数据集上批量生成并计算 BLEU / ROUGE，或保存预测结果。",
        "infer_d": "加载模型后流式对话，支持系统提示词、工具调用与多模态输入。",
        "export_d": "合并适配器、按需量化，导出为可独立加载的完整模型。",
        "search": "搜索",
        "collapse": "折叠 / 展开侧边栏（Ctrl+B）",
        "resize": "拖动调整宽度，双击复原",
        "c_model_base": "基础模型",
        "c_model_base_d": "模型、本地路径与下载源",
        "c_model_method": "微调方法与检查点",
        "c_model_method_d": "决定训练哪些参数；检查点用于续训、评估、对话与导出",
        "c_model_runtime": "量化 · 模板 · 加速",
        "c_model_runtime_d": "加载精度、对话模板、长上下文与算子加速",
        "c_train_data": "数据",
        "c_train_data_d": "训练阶段、数据集与预览",
        "c_train_hparams": "超参数",
        "c_train_hparams_d": "学习率、轮数、批大小与序列长度",
        "c_train_output": "输出与设备",
        "c_train_output_d": "结果目录、参数文件与多卡设置",
        "c_train_adv": "高级设置",
        "c_train_adv_d": "按需展开。标题右侧是当前取值；和当前配置无关的分组会自动变淡",
        "c_run": "运行",
        "c_run_d": "预览命令、启动与实时监控",
        "c_eval_data": "数据",
        "c_eval_data_d": "用于评估或预测的数据集",
        "c_eval_gen": "生成与批量",
        "c_eval_gen_d": "序列长度、批大小、解码参数与输出目录",
        "c_infer_engine": "推理引擎",
        "c_infer_engine_d": "先加载模型，对话区会出现在下方",
        "c_export_opts": "导出设置",
        "c_export_opts_d": "分块大小、导出量化与目标位置",
        "c_export_run": "执行",
        "c_export_run_d": "导出前在「模型」页选好检查点",
        "style": "风格",
        "style_eng": "工程",
        "style_soft": "柔光",
        "mode_dark": "深色",
        "mode_light": "浅色",
        "mode_toggle": "切换深色 / 浅色",
        "none": "未选择模型",
        "ckpt": "检查点",
        "idle": "空闲",
        "running": "运行中",
        "n_only_lora": "当前不生效 · 仅 LoRA",
        "n_only_freeze": "当前不生效 · 仅 Freeze 或 LoRA + LLaMA Pro",
        "n_only_pref": "当前不生效 · 仅 DPO / KTO / PPO",
        "n_only_mm": "当前不生效 · 非多模态模型",
        "n_no_lora": "当前不生效 · 不能与 LoRA 同用",
        "off": "未启用",
        "b_ppo": "当前环境装的是 TRL {v}，而 PPO 要求 0.8.6 ≤ TRL ≤ 0.9.6，直接开始会报版本错误。"
        "需要另建一个环境安装 trl==0.9.6 再跑 PPO。",
        "b_pkg": "当前环境没有安装 {what} 所需的包，开始后会报错。需要先执行：pip install {pip}",
        "b_fa2": "当前环境没有安装 flash-attn，选择 flashattn2 会给出警告并使用默认的注意力实现（通常是 SDPA），不影响运行。",
        "h_cur": "当前模型",
        "h_pick": "在下方选择一个模型，这里会显示它的结构与训练范围。",
        "h_noarch": "没有读到 config.json：填写本地模型目录后可显示结构信息。",
        "h_params": "参数量",
        "h_layers": "层数",
        "h_hidden": "隐藏维度",
        "h_ffn": "FFN 维度",
        "h_heads": "注意力头 Q / KV",
        "h_vocab": "词表",
        "h_ctx": "上下文",
        "h_dtype": "权重精度",
        "h_scope": "训练范围",
        "h_emb": "嵌入",
        "h_head": "输出头",
        "h_tied": "与嵌入共享",
        "lg_train": "可训练",
        "lg_frozen": "冻结",
        "lg_adapter": "适配器",
        "lg_quant": "已量化",
        "sc_full": "全参数微调：全部 {n} 参数都参与训练。",
        "sc_freeze": "冻结微调：只训练{where} {k} 层，约 {n} 参数（{p}）。",
        "sc_freeze_last": "最后",
        "sc_freeze_first": "最前",
        "sc_lora": "LoRA：{L} 层的全部线性层挂接秩 {r} 的低秩适配器，约 {n} 参数（{p}），基座冻结。",
        "sc_qlora": "QLoRA：基座线性层量化为 {b}-bit 并冻结，再挂接秩 {r} 的适配器，约 {n} 参数（{p}）。",
        "sc_oft": "OFT：在线性层上学习正交变换，基座冻结。",
        "flow": "流程",
        "f_model": "选基座与方法",
        "f_train": "数据 · 超参 · 方法",
        "f_eval": "BLEU / ROUGE",
        "f_infer": "加载后交互",
        "f_export": "合并与量化",
        "p_ph": "搜索页面、参数、操作…",
        "p_pages": "页面",
        "p_params": "参数",
        "p_act": "操作",
        "p_empty": "没有匹配项",
        "p_nav": "选择",
        "p_open": "打开",
        "p_close": "关闭",
        "a_style": "切换风格：工程 / 柔光",
        "a_mode": "切换深色 / 浅色",
        "a_nav": "折叠 / 展开侧边栏",
    },
    "en": {
        "setup": "Setup",
        "workspace": "Workspace",
        "model": "Model",
        "train": "Train",
        "eval": "Evaluate & Predict",
        "infer": "Chat",
        "export": "Export",
        "model_d": "One model configuration shared by training, evaluation, chat and export.",
        "train_d": "Pre-training, supervised fine-tuning, reward modelling and DPO / KTO / PPO alignment.",
        "eval_d": "Generate over a dataset and score BLEU / ROUGE, or save the predictions.",
        "infer_d": "Load a model and chat with streaming, system prompts, tool calls and multimodal input.",
        "export_d": "Merge adapters, optionally quantize, and export a standalone model.",
        "search": "Search",
        "collapse": "Collapse / expand sidebar (Ctrl+B)",
        "resize": "Drag to resize, double-click to reset",
        "c_model_base": "Base model",
        "c_model_base_d": "Model, local path and download hub",
        "c_model_method": "Method & checkpoints",
        "c_model_method_d": "What gets trained; checkpoints resume, evaluate, chat and export",
        "c_model_runtime": "Quantization · template · speed",
        "c_model_runtime_d": "Load precision, chat template, long context and kernels",
        "c_train_data": "Data",
        "c_train_data_d": "Stage, datasets and preview",
        "c_train_hparams": "Hyperparameters",
        "c_train_hparams_d": "Learning rate, epochs, batch and sequence length",
        "c_train_output": "Output & devices",
        "c_train_output_d": "Run directory, config file and multi-GPU",
        "c_train_adv": "Advanced",
        "c_train_adv_d": "Current values are shown on the right; groups that do not apply are dimmed",
        "c_run": "Run",
        "c_run_d": "Preview, launch and monitor",
        "c_eval_data": "Data",
        "c_eval_data_d": "Datasets to evaluate or predict on",
        "c_eval_gen": "Generation & batching",
        "c_eval_gen_d": "Length, batch size, decoding and output directory",
        "c_infer_engine": "Inference engine",
        "c_infer_engine_d": "Load a model first; the chat appears below",
        "c_export_opts": "Export options",
        "c_export_opts_d": "Shard size, export quantization and destination",
        "c_export_run": "Run",
        "c_export_run_d": "Pick the checkpoint on the Model page first",
        "style": "Style",
        "style_eng": "Crisp",
        "style_soft": "Soft",
        "mode_dark": "Dark",
        "mode_light": "Light",
        "mode_toggle": "Toggle dark / light",
        "none": "no model selected",
        "ckpt": "checkpoints",
        "idle": "Idle",
        "running": "Running",
        "n_only_lora": "Not in effect · LoRA only",
        "n_only_freeze": "Not in effect · Freeze or LoRA + LLaMA Pro only",
        "n_only_pref": "Not in effect · DPO / KTO / PPO only",
        "n_only_mm": "Not in effect · not a multimodal model",
        "n_no_lora": "Not in effect · incompatible with LoRA",
        "off": "off",
        "b_ppo": "This environment has TRL {v}, but PPO requires 0.8.6 <= TRL <= 0.9.6 and will stop with a "
        "version error. Use a separate environment with trl==0.9.6 for PPO.",
        "b_pkg": "The package required by {what} is not installed; starting will fail. Install it first: pip install {pip}",
        "b_fa2": "flash-attn is not installed; flashattn2 logs a warning and keeps the default attention "
        "implementation (usually SDPA).",
        "h_cur": "Current model",
        "h_pick": "Pick a model below to see its architecture and training scope.",
        "h_noarch": "No config.json found: set a local model directory to see the architecture.",
        "h_params": "Parameters",
        "h_layers": "Layers",
        "h_hidden": "Hidden",
        "h_ffn": "FFN",
        "h_heads": "Heads Q / KV",
        "h_vocab": "Vocab",
        "h_ctx": "Context",
        "h_dtype": "Weights",
        "h_scope": "Training scope",
        "h_emb": "Embed",
        "h_head": "LM head",
        "h_tied": "tied to embeddings",
        "lg_train": "trainable",
        "lg_frozen": "frozen",
        "lg_adapter": "adapter",
        "lg_quant": "quantized",
        "sc_full": "Full fine-tuning: all {n} parameters are trained.",
        "sc_freeze": "Freeze tuning: only the {where} {k} layers are trained, about {n} parameters ({p}).",
        "sc_freeze_last": "last",
        "sc_freeze_first": "first",
        "sc_lora": "LoRA: rank-{r} adapters on every linear layer of {L} blocks, about {n} parameters ({p}); "
        "the base stays frozen.",
        "sc_qlora": "QLoRA: base linear layers quantized to {b}-bit and frozen, plus rank-{r} adapters, "
        "about {n} parameters ({p}).",
        "sc_oft": "OFT: learns orthogonal transforms on linear layers; the base stays frozen.",
        "flow": "Workflow",
        "f_model": "Base & method",
        "f_train": "Data · hparams · method",
        "f_eval": "BLEU / ROUGE",
        "f_infer": "Chat after loading",
        "f_export": "Merge & quantize",
        "p_ph": "Search pages, parameters, actions…",
        "p_pages": "Pages",
        "p_params": "Parameters",
        "p_act": "Actions",
        "p_empty": "No matches",
        "p_nav": "navigate",
        "p_open": "open",
        "p_close": "close",
        "a_style": "Switch style: crisp / soft",
        "a_mode": "Toggle dark / light",
        "a_nav": "Collapse / expand sidebar",
    },
    "ru": {
        "setup": "Настройка",
        "workspace": "Рабочая область",
        "model": "Модель",
        "train": "Обучение",
        "eval": "Оценка и предсказание",
        "infer": "Чат",
        "export": "Экспорт",
        "model_d": "Общие настройки модели для обучения, оценки, чата и экспорта.",
        "train_d": "Предобучение, SFT, обучение модели награды и выравнивание DPO / KTO / PPO.",
        "eval_d": "Генерация по набору данных и расчёт BLEU / ROUGE или сохранение предсказаний.",
        "infer_d": "Загрузите модель и общайтесь с потоковым выводом и вызовом инструментов.",
        "export_d": "Объединение адаптеров, квантизация и экспорт готовой модели.",
        "search": "Поиск",
        "c_model_base": "Базовая модель",
        "c_model_method": "Метод и чекпойнты",
        "c_model_runtime": "Квантизация · шаблон · ускорение",
        "c_train_data": "Данные",
        "c_train_hparams": "Гиперпараметры",
        "c_train_output": "Вывод и устройства",
        "c_train_adv": "Дополнительно",
        "c_run": "Запуск",
        "c_eval_data": "Данные",
        "c_eval_gen": "Генерация",
        "c_infer_engine": "Движок вывода",
        "c_export_opts": "Параметры экспорта",
        "c_export_run": "Запуск",
        "style": "Стиль",
        "style_eng": "Чёткий",
        "style_soft": "Мягкий",
        "mode_dark": "Тёмная",
        "mode_light": "Светлая",
        "none": "модель не выбрана",
    },
    "ko": {
        "setup": "설정",
        "workspace": "작업 공간",
        "model": "모델",
        "train": "학습",
        "eval": "평가 및 예측",
        "infer": "채팅",
        "export": "내보내기",
        "model_d": "학습, 평가, 채팅, 내보내기에서 공용으로 쓰는 모델 설정입니다.",
        "train_d": "사전학습, 지도 미세조정, 보상 모델링, DPO / KTO / PPO 정렬.",
        "eval_d": "데이터셋에서 생성 후 BLEU / ROUGE를 계산하거나 예측을 저장합니다.",
        "infer_d": "모델을 불러와 스트리밍으로 대화합니다.",
        "export_d": "어댑터를 병합하고 필요하면 양자화하여 모델을 내보냅니다.",
        "search": "검색",
        "c_model_base": "기본 모델",
        "c_model_method": "방법과 체크포인트",
        "c_model_runtime": "양자화 · 템플릿 · 가속",
        "c_train_data": "데이터",
        "c_train_hparams": "하이퍼파라미터",
        "c_train_output": "출력과 장치",
        "c_train_adv": "고급 설정",
        "c_run": "실행",
        "c_eval_data": "데이터",
        "c_eval_gen": "생성과 배치",
        "c_infer_engine": "추론 엔진",
        "c_export_opts": "내보내기 설정",
        "c_export_run": "실행",
        "style": "스타일",
        "style_eng": "선명",
        "style_soft": "부드러움",
        "mode_dark": "다크",
        "mode_light": "라이트",
        "none": "모델 선택 안 됨",
    },
    "ja": {
        "setup": "設定",
        "workspace": "ワークスペース",
        "model": "モデル",
        "train": "学習",
        "eval": "評価と予測",
        "infer": "チャット",
        "export": "エクスポート",
        "model_d": "学習・評価・チャット・エクスポートで共用するモデル設定です。",
        "train_d": "事前学習、教師ありファインチューニング、報酬モデル、DPO / KTO / PPO。",
        "eval_d": "データセットで生成し BLEU / ROUGE を計算、または予測を保存します。",
        "infer_d": "モデルを読み込み、ストリーミングで対話します。",
        "export_d": "アダプタをマージし、必要なら量子化してモデルを出力します。",
        "search": "検索",
        "c_model_base": "ベースモデル",
        "c_model_method": "手法とチェックポイント",
        "c_model_runtime": "量子化 · テンプレート · 高速化",
        "c_train_data": "データ",
        "c_train_hparams": "ハイパーパラメータ",
        "c_train_output": "出力とデバイス",
        "c_train_adv": "詳細設定",
        "c_run": "実行",
        "c_eval_data": "データ",
        "c_eval_gen": "生成とバッチ",
        "c_infer_engine": "推論エンジン",
        "c_export_opts": "エクスポート設定",
        "c_export_run": "実行",
        "style": "スタイル",
        "style_eng": "シャープ",
        "style_soft": "ソフト",
        "mode_dark": "ダーク",
        "mode_light": "ライト",
        "none": "モデル未選択",
    },
}


def t(lang: str | None, key: str) -> str:
    for code in (lang or "en", "en", "zh"):
        table = T.get(code, {})
        if key in table:
            return table[key]

    return key


def esc(text) -> str:
    return html.escape("" if text is None else str(text), quote=True)


def _svg(body: str, size: int = 18, stroke_width: float = 1.7) -> str:
    return (
        f'<svg class="lf-ic" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{body}</svg>'
    )


ICONS = {  # inline SVG, currentColor, no external assets
    "model": _svg(
        '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z"/><path d="M12 12l8-4.5"/><path d="M12 12v9"/>'
        '<path d="M12 12L4 7.5"/>'
    ),
    "train": _svg('<path d="M3 20h18"/><path d="M6 16l4-6 4 3 5-8"/><circle cx="19" cy="5" r="1.2"/>'),
    "eval": _svg(
        '<path d="M9 6h11"/><path d="M9 12h11"/><path d="M9 18h11"/><path d="M3.5 6l1.5 1.5L7.5 5"/>'
        '<path d="M3.5 12l1.5 1.5L7.5 11"/><path d="M3.5 18l1.5 1.5L7.5 17"/>'
    ),
    "infer": _svg(
        '<path d="M21 12a8 8 0 0 1-11.6 7.1L4 20.5l1.4-4.9A8 8 0 1 1 21 12z"/>'
        '<path d="M8.5 12h.01M12 12h.01M15.5 12h.01"/>'
    ),
    "export": _svg(
        '<path d="M12 15V3.5"/><path d="M7.5 8L12 3.5 16.5 8"/>'
        '<path d="M4 14v4.5A2.5 2.5 0 0 0 6.5 21h11a2.5 2.5 0 0 0 2.5-2.5V14"/>'
    ),
    "sun": _svg(
        '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4'
        'M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
        16,
    ),
    "moon": _svg('<path d="M20.5 14.5A8.5 8.5 0 0 1 9.5 3.5a8.5 8.5 0 1 0 11 11z"/>', 16),
    "search": _svg('<circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/>', 16),
    "panel": _svg(
        '<rect x="3.5" y="4" width="17" height="16" rx="3"/><path d="M9.5 4v16"/><path d="M14.5 10l-2 2 2 2"/>', 17
    ),
    "style": _svg('<circle cx="12" cy="12" r="8.5"/><path d="M12 3.5a8.5 8.5 0 0 0 0 17z" fill="currentColor"/>', 16),
}

PAGE_ORDER = ("model", "train", "eval", "infer", "export")
NAV_GROUPS = (("setup", ("model",)), ("workspace", ("train", "eval", "infer", "export")))
PAGE_GROUP = {"model": "setup", "train": "workspace", "eval": "workspace", "infer": "workspace", "export": "workspace"}


def _num(page: str) -> str:
    return f"{PAGE_ORDER.index(page) + 1:02d}" if page in PAGE_ORDER else "00"


def nav_html(lang: str, pages, version: str) -> str:
    out = [
        '<div class="lf-nav-top">'
        '<div class="lf-brand"><div class="lf-mark" aria-hidden="true"><span></span><span></span><span></span></div>'
        '<div class="lf-brand-t"><div class="nm">LLaMA Factory</div>'
        f'<div class="sub">LLaMA Board · v{esc(version)}</div></div></div>'
        f'<button type="button" class="lf-icon-btn lf-collapse" data-lf-nav-toggle title="{esc(t(lang, "collapse"))}" '
        f'aria-label="{esc(t(lang, "collapse"))}">{ICONS["panel"]}</button></div>',
        f'<button type="button" class="lf-search" data-lf-palette data-lf-tip="{esc(t(lang, "search"))} · Ctrl K" '
        f'data-lf-tip-nav>{ICONS["search"]}<span class="tx">{esc(t(lang, "search"))}</span><kbd>Ctrl K</kbd></button>',
        '<nav class="lf-nav-list" aria-label="pages">',
    ]
    for group, keys in NAV_GROUPS:
        items = [k for k in keys if k in pages]
        if not items:
            continue

        out.append(f'<div class="lf-nav-cap"><span>{esc(t(lang, group))}</span></div>')
        for key in items:
            out.append(
                f'<a class="lf-nav-item" href="#{key}" data-lf-go="{key}" data-lf-tip="{esc(t(lang, key))}" '
                f'data-lf-tip-nav><span class="ic">{ICONS[key]}</span><span class="tx">{esc(t(lang, key))}</span>'
                '<span class="lf-live" aria-hidden="true"></span></a>'
            )

    out.append("</nav>")
    i18n = {
        k: t(lang, k)
        for k in ("p_ph", "p_pages", "p_params", "p_act", "p_empty", "p_nav", "p_open", "p_close", "a_style", "a_mode")
    }
    i18n.update(a_nav=t(lang, "a_nav"), resize=t(lang, "resize"), pages={k: t(lang, k) for k in PAGE_ORDER})
    out.append(f'<div id="lf-index" hidden data-i18n="{esc(json.dumps(i18n, ensure_ascii=False))}"></div>')
    return "".join(out)


def appearance_html(lang: str) -> str:
    return (
        '<div class="lf-appear">'
        f'<div class="lf-seg" role="group" aria-label="{esc(t(lang, "style"))}">'
        f'<button type="button" data-lf-style-set="eng">{esc(t(lang, "style_eng"))}</button>'
        f'<button type="button" data-lf-style-set="soft">{esc(t(lang, "style_soft"))}</button></div>'
        f'<button type="button" class="lf-icon-btn lf-style-cycle" data-lf-style-cycle data-lf-tip="{esc(t(lang, "style"))}" '
        f'data-lf-tip-nav aria-label="{esc(t(lang, "style"))}">{ICONS["style"]}</button>'
        f'<button type="button" class="lf-icon-btn lf-mode" data-lf-mode-toggle data-lf-tip="{esc(t(lang, "mode_toggle"))}" '
        f'data-lf-tip-nav aria-label="{esc(t(lang, "mode_toggle"))}">'
        f'<span class="i-sun">{ICONS["sun"]}</span><span class="i-moon">{ICONS["moon"]}</span></button>'
        "</div>"
    )


def page_head_html(lang: str, key: str) -> str:
    return (
        '<header class="lf-ph"><div class="lf-ph-main">'
        f'<div class="lf-eyebrow"><span class="n">{_num(key)}</span><span class="g">{esc(t(lang, PAGE_GROUP.get(key, "")))}'
        f"</span></div><h1>{esc(t(lang, key))}</h1><p>{esc(t(lang, key + '_d'))}</p></div></header>"
    )


def card_head_html(lang: str, key: str, index: int | None = None) -> str:
    desc = t(lang, key + "_d")
    desc_html = f'<span class="d">{esc(desc)}</span>' if desc != key + "_d" else ""
    ix = f'<span class="ix">{index:02d}</span>' if index else ""
    live = '<span class="lf-led" aria-hidden="true"></span>' if key == "c_run" else ""
    return (
        f'<div class="lf-card-h">{ix}<div class="tt"><span class="t">{esc(t(lang, key))}{live}</span>'
        f"{desc_html}</div></div>"
    )


def ctx_html(lang: str, name, finetuning, template, quant_bit, checkpoints) -> str:
    crumbs = "".join(
        f'<span class="cr" data-p="{p}"><span class="g">{esc(t(lang, PAGE_GROUP[p]))}</span><i>/</i>'
        f"<b>{esc(t(lang, p))}</b></span>"
        for p in PAGE_ORDER
    )
    if name:
        value = f'<span class="v">{esc(name)}</span>'
    else:
        value = f'<span class="v dim">{esc(t(lang, "none"))}</span>'

    tags = []
    for tag in (finetuning, template, None if quant_bit in (None, "none") else f"{quant_bit}-bit"):
        if tag:
            tags.append(f'<span class="tg">{esc(tag)}</span>')

    if checkpoints:
        n = len(checkpoints) if isinstance(checkpoints, (list, tuple)) else 1
        tags.append(f'<span class="tg tg-ck">{esc(t(lang, "ckpt"))} × {n}</span>')

    return (
        f'<div class="lf-ctx-in"><div class="lf-crumbs">{crumbs}</div>'
        f'<a class="lf-ctx-model" href="#model" data-lf-go="model"><span class="dot"></span>{value}'
        f'<span class="tgs">{"".join(tags)}</span></a>'
        f'<button type="button" class="lf-ctx-search" data-lf-palette>{ICONS["search"]}'
        f"<span>{esc(t(lang, 'search'))}</span><kbd>Ctrl K</kbd></button></div>"
    )


def run_state_html(lang: str, running: bool, kind: str | None) -> str:
    state = "running" if running else "idle"
    return f'<span class="lf-runstate" data-state="{state}" data-kind="{esc(kind or "")}">{esc(t(lang, state))}</span>'


def _count(n: float, lang: str) -> str:
    if lang == "zh":
        if n >= 1e8:
            return f"{n / 1e8:.2f} 亿"

        if n >= 1e4:
            return f"{n / 1e4:,.0f} 万"

        return f"{n:,.0f}"

    if n >= 1e9:
        return f"{n / 1e9:.2f} B"

    if n >= 1e6:
        return f"{n / 1e6:.1f} M"

    return f"{n:,.0f}"


def hero_html(lang: str, card: dict | None, name, path, ft, quant_bit, freeze_k, lora_rank) -> str:
    r"""Model overview: architecture figures and which parts the chosen method trains."""
    if not name:
        body = f'<div class="lf-hero-empty">{esc(t(lang, "h_pick"))}</div>'
        return f'<section class="lf-hero"><div class="lf-hero-k">{esc(t(lang, "h_cur"))}</div>{body}</section>'

    tags = [ft or ""]
    if quant_bit not in (None, "", "none"):
        tags.append(f"{quant_bit}-bit")

    tag_html = "".join(f'<span class="tg">{esc(x)}</span>' for x in tags if x)
    head = (
        f'<div class="lf-hero-top"><div><div class="lf-hero-k">{esc(t(lang, "h_cur"))}</div>'
        f'<div class="lf-hero-name">{esc(name)}</div>'
        f'<div class="lf-hero-path">{esc(path or "")}</div></div><div class="lf-hero-tags">{tag_html}</div></div>'
    )
    if not card:
        return f'<section class="lf-hero">{head}<div class="lf-hero-empty">{esc(t(lang, "h_noarch"))}</div></section>'

    arch, cnt = card["arch"], card["counts"]
    specs = [
        ("h_params", _count(cnt["total"], lang)),
        ("h_layers", str(arch.layers)),
        ("h_hidden", f"{arch.hidden:,}"),
        ("h_ffn", f"{arch.inter:,}"),
        ("h_heads", f"{arch.heads} / {arch.kv_heads}"),
        ("h_vocab", f"{arch.vocab:,}"),
        ("h_ctx", f"{card['max_pos']:,}" if card.get("max_pos") else "—"),
        ("h_dtype", card.get("dtype") or "—"),
    ]
    spec_html = "".join(
        f'<div class="sp"><span class="k">{esc(t(lang, k))}</span><span class="v">{esc(v)}</span></div>'
        for k, v in specs
    )

    n_layers = arch.layers
    try:
        k = int(freeze_k or 0)
    except (TypeError, ValueError):
        k = 2

    rank = int(lora_rank or 8)
    quant = quant_bit not in (None, "", "none") and ft in ("lora", "oft")
    layer_state = ["frozen"] * n_layers
    emb_state = head_state = "frozen"
    if ft == "full":
        layer_state = ["train"] * n_layers
        emb_state = head_state = "train"
        scope = t(lang, "sc_full").format(n=_count(cnt["total"], lang))
    elif ft == "freeze":
        kk = max(-n_layers, min(n_layers, k))
        for i in range(n_layers - kk, n_layers) if kk > 0 else range(0, -kk):
            layer_state[i] = "train"

        n_train = abs(kk) * cnt["layer"]
        scope = t(lang, "sc_freeze").format(
            where=t(lang, "sc_freeze_last" if kk > 0 else "sc_freeze_first"),
            k=abs(kk),
            n=_count(n_train, lang),
            p=f"{n_train / cnt['total'] * 100:.1f}%",
        )
    elif ft in ("lora", "oft"):
        layer_state = ["quant" if quant else "frozen"] * n_layers
        if ft == "oft":
            scope = t(lang, "sc_oft")
        else:
            n_train = cnt["lora"]
            scope = t(lang, "sc_qlora" if quant else "sc_lora").format(
                L=n_layers, r=rank, b=quant_bit, n=_count(n_train, lang), p=f"{n_train / cnt['total'] * 100:.2f}%"
            )
    else:
        scope = ""

    adapters = ft in ("lora", "oft")
    cells = "".join(f'<i class="ly {st}{" ad" if adapters else ""}"></i>' for st in layer_state)
    head_cls = "tied" if arch.tie else head_state
    strip = (
        f'<div class="lf-strip" role="img" aria-label="{esc(scope)}">'
        f'<span class="blk {emb_state}">{esc(t(lang, "h_emb"))}</span>'
        f'<span class="lys" style="--n:{n_layers}">{cells}</span>'
        f'<span class="blk {head_cls}" title="{esc(t(lang, "h_tied")) if arch.tie else ""}">{esc(t(lang, "h_head"))}</span>'
        "</div>"
    )
    legend = "".join(
        f'<span class="lg lg-{k}"><i></i>{esc(t(lang, "lg_" + k))}</span>'
        for k in ("train", "frozen", "adapter", "quant")
        if (k != "adapter" or adapters) and (k != "quant" or quant)
    )
    src = f'<span class="lf-hero-src">{esc(card.get("source", ""))}</span>' if card.get("source") else ""
    return (
        f'<section class="lf-hero">{head}<div class="lf-specs">{spec_html}</div>'
        f'<div class="lf-scope"><div class="lf-scope-h"><span class="k">{esc(t(lang, "h_scope"))}</span>'
        f'<span class="lf-legend">{legend}</span></div>{strip}<div class="lf-scope-t">{esc(scope)}</div></div>'
        f"{src}</section>"
    )


def flow_html(lang: str, pages) -> str:
    items = []
    steps = [p for p in PAGE_ORDER if p in pages]
    for i, page in enumerate(steps, 1):
        items.append(
            f'<a class="lf-step" href="#{page}" data-lf-go="{page}"><span class="n">{i}</span>'
            f'<span class="tx"><b>{esc(t(lang, page))}</b><small>{esc(t(lang, "f_" + page))}</small></span>'
            f"{ICONS[page]}</a>"
        )

    return f'<nav class="lf-flow" aria-label="{esc(t(lang, "flow"))}">{"".join(items)}</nav>'
