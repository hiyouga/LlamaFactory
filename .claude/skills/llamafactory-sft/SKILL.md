---
name: llamafactory-sft
description: One-stop guided LlamaFactory SFT workflow — data prep, model prep, fine-tuning config/method selection, background training, loss visualization, effect validation and model export. TRIGGER when the user wants to fine-tune / SFT a model with LlamaFactory, run a full LlamaFactory training pipeline, or asks to "微调" / "用 llamafactory 做 sft" / "train a LoRA". SKIP for general questions about the codebase or non-training tasks.
---

# LlamaFactory One-Stop SFT Workflow

Guide the user through a complete LlamaFactory SFT run: **data prep → model prep → fine-tuning config → background training → loss visualization → (optional) effect validation → (optional) model export**.

The core of this skill is **interactive guidance**: at every key decision point use `AskUserQuestion` and never make irreversible assumptions on the user's behalf.

The working directory defaults to the current directory (`./`, the repo root). Run all commands via `llamafactory-cli` (or `lmf`).

> **Artifacts directory (keep generated files out of model / checkpoint folders).** All skill-generated **yaml configs** (train / inference / export) and **run logs** (download / train / export) MUST be written to a single dedicated directory, NOT inside `./models/` (downloaded weights) or the training `output_dir` (saved checkpoints). Use `./llamafactory_runs/<run_id>/` where `<run_id>` is `<model>_<method>_YYYYMMDD_HHMM` (e.g. `./llamafactory_runs/qwen3-4b_lora_20260615_0807/`). Create it once at the start of the CLI route and place every config + log file there. The training `output_dir` (the actual model checkpoints) stays separate under `saves/...`, and downloaded base models stay under `./models/...`. This keeps configs/logs, checkpoints, and weights cleanly separated.

> Language: communicate with the user in whatever language they request, or otherwise follow the active language setting/preference. Do not hardcode a fixed conversation language.

---

## Progress board (show only when progress changes)

This workflow has many stages, so the user must always be able to see **where we are**: which steps are done, which is in progress, and which remain.

**Rule:** Render the **progress board** only when the board state changes, or when the user explicitly asks for status/progress. A board state changes when any step marker changes (`[ ]` / `[~]` / `[x]` / `[-]`) or when Stage 0 changes the branch shape (e.g. skipped validation/export for "SFT only"). Do **not** repeat the board on routine status updates, repeated confirmations, log polling, or consecutive messages within the same stage if the markers are unchanged.

Use these markers:
- `[x]` done
- `[~]` in progress (the step you are working on right now)
- `[ ]` not started
- `[-]` skipped (e.g. validation/export when the user chose "SFT only", or download when the model is already local)

Render it as a compact checklist inside a fenced code block. **Always wrap the entire progress board in triple backticks** (use `text` as the info string, or no info string) so Markdown preserves every line break and marker exactly. **The board's language must follow the active conversation language / the user's request — it is NOT required to be Chinese.** The example below is in Chinese only for illustration; render the title and step labels in whatever language the user is using (e.g. English: a "Progress" board with "0. Confirm overall flow", "1. Data preparation", ...):

```
进度看板
[x] 0. 确认整体流程（范围 / 执行方式）
[x] 1. 数据准备
[~] 2. 模型准备
[ ] 3. 微调配置
[ ] 4. SFT 训练
[ ] 5. 效果验证
[ ] 6. 模型导出
```

Guidelines:
- The step list above is the canonical set. Loss visualization is part of **SFT 训练** (it happens automatically once training finishes) — you know to handle it, but do NOT show it as a separate line on the board.
- Mark steps `[-]` instead of dropping them when they don't apply to the chosen flow (e.g. "SFT only" skips 5 & 6; WebUI route collapses 3–6 into the UI).
- After Stage 0, adapt the board to the chosen branch (mark skipped steps `[-]`) and keep that shape for the rest of the run.
- Do **not** render the board as plain paragraph text or as Markdown task-list checkboxes (`- [x] ...`); both can change the intended layout or marker semantics.
- Keep it terse — one line per step. When the board is shown, put it at the **top** of your message, then continue with the actual content / question below it.
- Update markers as soon as a step's status changes; show the updated board once at that transition, and never show a stale board.

---

## Stage 0: Confirm the overall flow

Before doing anything, use a **single `AskUserQuestion` call that asks all three of the most important branches together** (one question object each, in the same call) so the user can settle the whole shape of the run in one step:

1. **Flow scope**:
   - SFT only (data prep + model prep + fine-tuning)
   - Full flow (also includes effect validation + model export)

2. **Execution mode**:
   - Run everything via CLI commands
   - Launch a **WebUI** for the SFT / validation / export parts and let the user operate it

3. **Fine-tuning type** — which SFT fine-tuning type to use, so you can match the corresponding official example yaml directory:
   - **LoRA** → base config from `examples/train_lora/` (recommended default)
   - **Full** (full-parameter) → base config from `examples/train_full/`
   - **QLoRA** (quantized LoRA) → base config from `examples/train_qlora/`

Notes on the fine-tuning-type answer:
- The type only matters for the **all-CLI** route. If the user picked **WebUI** in question 2, **ignore their fine-tuning-type answer** (the type is selected inside the UI) — it does no harm to have asked.
- For the **all-CLI** route, record the chosen type and use it (together with the model family resolved in Stage 2) to pick the base example yaml in Stage 3.

Record the user's choices and branch the later stages accordingly.

---

## Stage 1: Data preparation

Use `AskUserQuestion` to ask about the data source:

- **Use LlamaFactory built-in data**: list selectable datasets from `data/dataset_info.json` (e.g. `identity`, `alpaca_zh_demo`, `alpaca_en_demo`, ...). Allow multi-select and join into `dataset: a,b,c`.
- **Use custom data**: ask for the data file path(s) (multiple allowed).

### Custom data validation & onboarding

For each custom data file:

1. **Read and validate the format.** LlamaFactory supports `alpaca` and `sharegpt` formats; file types may be json/jsonl/csv/parquet/arrow.
   - **Alpaca format** key fields: `instruction` (required), `input` (optional), `output` (required), `system`/`history` (optional).
   - **ShareGPT format** key fields: a `conversations` list whose elements contain `from` (human/gpt) and `value`; multimodal data adds `images`/`videos`/`audios`.
   - Full field definitions are in `data/README.md` / `data/README_zh.md`; read them when needed.
2. **If it does not meet the requirements, fix it**: with the user's consent, convert the data into a compliant format (write a NEW file, never destroy the user's original data in place).
3. **Copy it into the `data/` directory.**
4. **Update `data/dataset_info.json`**: add a dataset description entry. Minimal form:
   ```json
   "my_dataset": { "file_name": "my_dataset.json" }
   ```
   For sharegpt or custom column names, also add `formatting` / `columns` / `tags`.
   When editing this JSON, keep it valid (use Edit for precise insertion; verify braces/commas).

### identity.json and other templated datasets

If the user picks `identity.json` (contains template variables like `{{name}}`, `{{author}}`), use `AskUserQuestion` to ask whether to do a global replacement. If yes:
- **Use `AskUserQuestion` to collect the concrete replacement values for the `{{name}}` and `{{author}}` variables** (one question per variable, e.g. "模型名 {{name}}" and "作者/机构 {{author}}"), plus any other template variables the file contains. Do not assume or hardcode these values — always ask the user.
- **Recommend copying first, then replacing** (e.g. `data/identity_custom.json`) to avoid polluting the repo's built-in file, and register the new name in `dataset_info.json`.
- Use Edit with `replace_all` to perform the replacement.

---

## Stage 2: Model preparation

Use `AskUserQuestion` to ask for the model choice, and **offer a default recommendation** (e.g. `Qwen/Qwen3-4B-Instruct-2507`, matching `examples/train_lora/qwen3_lora_sft.yaml`).

Once the model is decided:
1. **Verify the model type against LlamaFactory's own model list, and use it to pick the matching default yaml.** Before downloading or configuring anything, look up the chosen model in LlamaFactory's built-in registry rather than guessing:
   - `SUPPORTED_MODELS` and `DEFAULT_TEMPLATE` live in `src/llamafactory/extras/constants.py` (registered via `register_model_group(...)`, each group sharing one `template=`). This is the authoritative list of which models LlamaFactory supports and which template each uses.
   - Match the user's model (by HF/ModelScope id or model name) to an entry there to learn its **model family / type and default `template`** (e.g. a Qwen3 model → `qwen3` / `qwen3_nothink`). Cross-check `src/llamafactory/data/template.py` (the `TEMPLATES` dict) if needed.
   - **Use the resolved family/template — together with the fine-tuning type chosen in Stage 0 (LoRA → `examples/train_lora/`, Full → `examples/train_full/`, QLoRA → `examples/train_qlora/`) — to select the closest official example yaml** (and the matching `examples/inference/`, `examples/merge_lora/`) as the base config for Stage 3 — e.g. a Qwen3 model with LoRA → `examples/train_lora/qwen3_lora_sft.yaml`. If there is no exact example for that family/type, pick the nearest supported one and note that you adapted it.
   - **If the model is NOT in the supported list** (no entry / no matching template), clearly tell the user it is unsupported, so they can switch to a supported model or define a custom template — do not silently proceed.
2. **Check whether it already exists locally** (HF cache `~/.cache/huggingface/hub`, or a user-specified local path).
   - **Verify completeness, not just existence — beware empty shells.** A cached directory can exist while being only a few KB (an empty shell where the real download never finished). After finding a local copy, verify it: total size is in the expected GB range, all `*.safetensors` shards referenced by `model.safetensors.index.json` are present, and there are no `*.incomplete` files. Only treat the model as available if it passes; otherwise treat it as NOT downloaded and proceed to download.
3. **If not present locally**: first use `AskUserQuestion` to confirm the **download source**:
   - **Hugging Face Hub**
   - **ModelScope** (often faster in mainland China)
   - **Let the agent decide** (pick automatically based on network reachability / region)

   Then **start a download task concurrently** via `Agent` / Bash with `run_in_background`:
   ```bash
   # Hugging Face (modern CLI; `huggingface-cli` is deprecated — use `hf download`)
   # Enable Xet high-performance transfer for much faster downloads:
   HF_XET_HIGH_PERFORMANCE=1 hf download <model_id> --local-dir <path>
   # ModelScope (set USE_MODELSCOPE_HUB=1 to also let LlamaFactory auto-download at train time)
   modelscope download --model <model_id> --local_dir <path>
   ```
   Notes:
   - `huggingface-cli download` is **deprecated** in recent `huggingface_hub` versions and may just print help text — always use `hf download`.
   - The old `HF_HUB_ENABLE_HF_TRANSFER=1` is **deprecated** too; use `HF_XET_HIGH_PERFORMANCE=1` instead.
   - If a download source is slow (e.g. ModelScope < ~1 MB/s), consider switching to the other source rather than waiting hours. In practice HF + Xet is often dramatically faster.

   You may instead rely on auto-download at train time (set `USE_MODELSCOPE_HUB=1` for ModelScope).
   Keep advancing the config work while it downloads, and report download progress to the user periodically (interval ≤ 100 seconds).

---

## Stage 2.5: GPU selection (detect free GPU(s) once before running)

**Before the first command that uses the GPU** (SFT training, and later validation / export), **detect GPU status once, pick the device(s) to use, and reuse that exact choice** for SFT, validation, and export. Never assume a GPU is free — other jobs may already be using the machine. **Detect only once here** — do NOT re-detect before every run; the index/indices chosen now are reused for the whole workflow.

1. **Detect GPUs and their load.** Try AMD/ROCm first, then NVIDIA. Wrap the call in `timeout` so a live-refreshing monitor can never hang the agent (plain `amd-smi` without a subcommand only prints help and lacks the `VRAM_USAGE` / `GFX%` snapshot, so keep the `monitor` subcommand for the one-shot table):
   ```bash
   timeout 10 amd-smi monitor 2>/dev/null || rocm-smi 2>/dev/null || nvidia-smi
   ```
   Read each GPU's **VRAM usage** and **utilization**. Treat a GPU as **free** when its VRAM usage is near-empty (e.g. ≲ 1 GB) and utilization is low (e.g. ≲ a few %). Also list any running training/inference processes if helpful (e.g. `pgrep -af "llamafactory-cli"`).
2. **Decide which device(s) to use:**
   - **No free GPU:** do NOT silently queue onto a busy card — tell the user which GPUs are busy (and roughly by how much VRAM / what is running) and use `AskUserQuestion` to let them choose: wait for one to free up, share a partially-used GPU anyway, or specify a particular index.
   - **Exactly one free GPU:** use it (single-card).
   - **Multiple free GPUs:** use `AskUserQuestion` to ask the user whether to run **single-card** or **multi-card** (list the free indices). If the user picks multi-card, use the selected free indices together.
3. **Record the chosen index/indices and reuse them for every GPU command** (training, validation, export) by exporting the platform's visibility env var:
   - **AMD/ROCm:** `HIP_VISIBLE_DEVICES=<idx>` (multi-card: comma-separated, e.g. `HIP_VISIBLE_DEVICES=0,1`)
   - **NVIDIA/CUDA:** `CUDA_VISIBLE_DEVICES=<idx>` (multi-card: comma-separated)

   Caveat: the visibility-env index follows the smi enumeration, and the selected device becomes index 0 *inside* the process. The mapping between `HIP_VISIBLE_DEVICES` and the `amd-smi` physical order may not be 1:1 — after launching, confirm the **intended** card's VRAM actually rises in `amd-smi` (and not some other card) before trusting it.

Tell the user **which GPU(s) you selected and why** (free vs busy, single vs multi), and surface it in the relevant status updates.

Notes:
- **Progress board:** GPU selection is part of **模型准备** prep work — handle it before the first run, but do NOT add a separate board line for it.
- **WebUI route:** you don't drive the runs yourself, but still detect and **tell the user which GPU is free** so they can set it in the UI / launch env.

---

## Stage 3A: WebUI route (user chose WebUI)

If the user chose WebUI in Stage 0:

```bash
llamafactory-cli webui
```

- Launch in the background, redirecting logs to a log file.
- Read the actual listening **port** from the log (default 7860) and tell the user.
- If on a remote SSH environment, explain port forwarding:
  ```bash
  ssh -L 7860:localhost:7860 user@remote_host
  ```
  then open `http://localhost:7860` in the local browser.
- Tell the user that subsequent SFT / validation / export are done in the UI.
- **After the UI is up, print a "建议在 WebUI 中填写的配置 / Suggested WebUI settings" table** that maps the choices already made in earlier stages onto the fields the user will fill in the WebUI, so they don't have to remember them. Derive the values from what was resolved earlier — do NOT re-ask. Include at least:

  | WebUI 字段 | 建议值 | 来源 |
  |-----------|--------|------|
  | Model path (模型路径) | `<resolved local model path or HF/MS id>` | Stage 2 模型准备 |
  | Template (对话模板) | `<resolved template>` | Stage 2 注册表解析（与训练一致） |
  | Dataset (数据集) | `<dataset a,b,c>` | Stage 1 数据准备 |
  | Finetuning method (微调方法) | `<lora / full / qlora>` | Stage 0 微调类型 |

  Add any other already-known values that map to UI fields (e.g. dataset dir if custom data was onboarded). Make clear these are **suggestions to enter in the UI** (LlamaFactory's WebUI does not auto-load them), and that the user can still adjust everything in the interface. If something was not resolved yet (e.g. the model path because the user deferred to train-time auto-download), say so instead of inventing a value.

---

## Stage 3B: CLI route (user chose all-CLI)

If the user chose all-CLI:

1. Using the official example that matches **both the chosen model family AND the fine-tuning type selected in Stage 0** (LoRA → `examples/train_lora/`, Full → `examples/train_full/`, QLoRA → `examples/train_qlora/`; e.g. a Qwen3 + LoRA run → `examples/train_lora/qwen3_lora_sft.yaml`) as a template, **match the default yaml to the chosen model**. **Prefer the official example's default parameters** — treat the example yaml as the source of truth and change as little as possible. Any field may still be changed when the user's needs or data selection call for it (including `template`), but **every change must be tracked and justified** (see step 3).
   - If the repo has no ready-made config for that model, check whether it is supported (template list). If unsupported, report back to the user.
2. **Print the key parameters to the user for confirmation** as a table: `model_name_or_path`, `stage: sft`, `finetuning_type` (lora/full/qlora), `lora_rank`/`lora_target`, `dataset`, `template`, `cutoff_len`, `output_dir`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `learning_rate`, `num_train_epochs`, `bf16`, etc.
3. **If you changed ANY value away from the official example's default**, then *after* the full parameter table, also show a **separate "差异 / Diff vs. default" table** listing only the changed fields, in the form `| parameter | default value | new value | reason |`. The reason must state *why* it changed — e.g. user-specified, required adaptation to the chosen model/dataset, or another concrete cause. This makes every deviation explicit and reviewable. If nothing was changed from the defaults, say so explicitly and omit the diff table.
4. Use `AskUserQuestion` to ask whether the fine-tuning **method/parameters** need adjusting (finetuning_type, lora_rank, learning rate, number of epochs, etc.), and modify as needed. When you do adjust, update the diff table accordingly.

> **Small-dataset hyperparameter reminder.** For very small datasets (e.g. `identity` with ~91 rows), the example defaults can produce too few steps: total_steps ≈ ceil(num_rows / (batch_size × grad_accum)) × epochs. If total_steps is tiny (single digits) or `logging_steps` ≥ total_steps, you will get too few loss points to plot a curve, and the model may underfit. Conversely, cranking epochs very high on a single tiny dataset causes **overfitting / catastrophic forgetting** (the model memorizes identity but its general language ability degrades — garbled or wrong-language output). Recommended mitigation: mix in a general dataset (e.g. `identity_custom,alpaca_zh_demo,alpaca_en_demo`), keep epochs moderate, and set `logging_steps` small enough to capture several points. Surface this trade-off to the user when relevant.

---

## Stage 4: Generate yaml and run training

1. **Generate the final yaml** into the **artifacts directory** (`./llamafactory_runs/<run_id>/`, see the top-of-file convention), NOT into the checkpoint or model folders. Use a **distinguishing suffix** so multiple runs do not collide — combine the **model name + fine-tuning method + date/time stamp**, e.g. `llamafactory_runs/<run_id>/sft.yaml` where `<run_id>` = `<model>_<method>_YYYYMMDD_HHMM` (build the stamp with `date +%Y%m%d_%H%M`). The yaml's `output_dir` (the actual checkpoints) is a separate location under `saves/<model>/<method>/sft_YYYYMMDD_HHMM`. Show the yaml to the user for **final confirmation**. Make sure `plot_loss: true` so loss can be plotted later.
2. After confirmation, **run SFT in the background on the GPU(s) chosen in Stage 2.5**, writing the log into the same artifacts directory:
   ```bash
   TS=$(date +%Y%m%d_%H%M)
   RUN_ID="<model>_<method>_${TS}"
   RUN_DIR="llamafactory_runs/${RUN_ID}"
   mkdir -p "${RUN_DIR}"
   # Prefix with the selected GPU's visibility env var (HIP_VISIBLE_DEVICES for AMD, CUDA_VISIBLE_DEVICES for NVIDIA):
   HIP_VISIBLE_DEVICES=<idx> nohup llamafactory-cli train "${RUN_DIR}/sft.yaml" > "${RUN_DIR}/train.log" 2>&1 &
   ```
   Use Bash with `run_in_background`, or `nohup ... &`.
   - **Beware the "fake completion" notification.** When you launch training with `nohup ... &` (or a backgrounded wrapper), the wrapper shell exits immediately and the harness may emit a `<task-notification> ... completed (exit code 0)` event — but the **real training process is still running** (it was detached). Do NOT treat that notification as "training finished". Training is only truly done when you confirm it via the actual process/log: `pgrep -f "llamafactory-cli train"` shows no process AND the log contains `Training completed` / a `train_runtime` line / the final `train metrics`. Until then, keep polling.
3. **Periodically check task status and report to the user** (interval ≤ 100 seconds): tail the log file, check the process is alive (and optionally `amd-smi` / `nvidia-smi`). **Each status report must include BOTH the current run state AND the current loss.**
   - Progress (`current step / total steps`) comes from the tqdm progress-bar lines, which use `\r`; convert `\r`→`\n` first, e.g. `tr '\r' '\n' < "${RUN_DIR}/train.log" | grep -oE "[0-9]+/[0-9]+ \[[^]]*\]" | tail -1` (reference the log path directly — a literal `<log>` placeholder would be parsed by bash as a redirection operator and error out).
   - Loss is logged as dict lines like `{'loss': '5.227', 'grad_norm': '4.634', 'learning_rate': '9.924e-05', 'epoch': '5'}` (emitted every `logging_steps`). Extract the latest with e.g. `grep -aoE "\{'loss':[^}]*\}" "${RUN_DIR}/train.log" | tail -1`, and report the latest `loss` value (optionally with `epoch`) to the user alongside the step progress.

---

## Stage 5: Loss visualization

After training finishes:
- `output_dir` will contain `training_loss.png` (because `plot_loss: true`); **just tell the user the image path** — do NOT render the loss curve on the command line.

---

## Stage 6: Effect validation (optional, if user chose full flow)

Load the fine-tuned checkpoint for inference validation.

**Up front — before asking the user for test questions — tell them about interactive self-testing.** Explain that interactive `chat` can't be driven in this agent environment, so validation here uses a non-interactive script, but if they want to test interactively themselves they can run the following in their own terminal (mention this once at the very start of this stage, before step 2's question, and again in the closing summary after validation finishes):
```bash
llamafactory-cli chat llamafactory_runs/<run_id>/infer.yaml
```

1. Generate an inference config into the **artifacts directory** (`./llamafactory_runs/<run_id>/infer.yaml`). **The config differs by the `finetuning_type` chosen in Stage 0 — branch accordingly:**

   - **LoRA / QLoRA** (the training `output_dir` is a LoRA *adapter*, not a full model) — base it on `examples/inference/qwen3_lora_sft.yaml` and use `adapter_name_or_path`:
     ```yaml
     model_name_or_path: <base_model>
     adapter_name_or_path: <output_dir>   # LoRA adapter path (the training output_dir)
     template: <template>                 # MUST match the template used at training time
     infer_backend: huggingface
     trust_remote_code: true
     ```
   - **Full** (the training `output_dir` is already a complete set of model weights — there is NO adapter) — base it on `examples/inference/qwen3_full_sft.yaml` and point `model_name_or_path` directly at the `output_dir`, with **NO `adapter_name_or_path` field at all**:
     ```yaml
     model_name_or_path: <output_dir>     # the full fine-tuned model (training output_dir)
     template: <template>                 # MUST match the template used at training time
     infer_backend: huggingface
     trust_remote_code: true
     ```
     Adding an `adapter_name_or_path` for a Full run is wrong (there is no adapter to load) and will fail or silently load nothing.

   - **Keep `template` identical to the training config.** A mismatch (e.g. trained with `qwen3_nothink` but inferred with `qwen3`) activates think / tool-call special tokens the model never saw in training and produces garbled output.
   - **Show the inference config as a table and get user confirmation before running.** Just like the training yaml, after generating `infer.yaml` print its key parameters as a table (for LoRA/QLoRA: `model_name_or_path`, `adapter_name_or_path`, `template`, `infer_backend`, `trust_remote_code`; for Full: `model_name_or_path`, `template`, `infer_backend`, `trust_remote_code`) and use `AskUserQuestion` to let the user confirm (or request changes) before you run any inference. Do not start validation until the user confirms.
2. Ask the user to provide **test text** (and test image path(s) for multimodal models).
3. **Use a non-interactive batch inference script** (preferred). `llamafactory-cli chat` is an **interactive** REPL and cannot be driven in this agent / background environment, so validation is done with a short script (write it into the artifacts dir) that loads `ChatModel` with the **same args as `infer.yaml`** (so it must follow the same LoRA/QLoRA-vs-Full branching — include `adapter_name_or_path` only for LoRA/QLoRA, never for Full) and feeds the test prompts, e.g. for a LoRA/QLoRA run:
   ```python
   from llamafactory.chat import ChatModel
   chat = ChatModel({
       "model_name_or_path": "<base_model>",
       "adapter_name_or_path": "<output_dir>",  # LoRA/QLoRA only — OMIT this key for Full
       "template": "<template>",
       "infer_backend": "huggingface",
       "trust_remote_code": True,
   })
   for q in ["你是谁？", "Who are you?"]:
       print(q, "->", chat.chat([{"role": "user", "content": q}])[0].response_text)
   ```
   For a **Full** run, drop the `adapter_name_or_path` key and set `"model_name_or_path": "<output_dir>"`.
   Running this in the foreground is fine (only training really needs background); **run it on the GPU(s) chosen in Stage 2.5** by prefixing the command with the visibility env var, e.g. `HIP_VISIBLE_DEVICES=<idx> python <script>` (or `CUDA_VISIBLE_DEVICES=<idx>` on NVIDIA); just **report the outputs to the user** when it finishes.
   - **After validation finishes, state the interactive self-test command again** (the same `llamafactory-cli chat llamafactory_runs/<run_id>/infer.yaml` shown at the start of this stage), so the user has it both before and after validation.
4. Multimodal: pass images together with the prompt and show the model's answer.

---

## Stage 7: Model export (optional)

If the user wants to export, **the meaning of "export" depends on the `finetuning_type` chosen in Stage 0 — branch accordingly:**

- **LoRA / QLoRA** — export *merges* the LoRA adapter into the base model and writes a standalone full model. This is the classic `merge_lora` flow.
- **Full** — there is **no adapter and nothing to merge**; the training `output_dir` is already a complete model. "Export" here just re-saves / tidies those full weights (plus tokenizer, generation config, etc.) into a clean user-specified directory. Do NOT describe this as "merging".

1. Ask for the **export name/directory** (user-specified). This is where the exported model weights go (e.g. under `./models/...`) — it is the model output, separate from the artifacts directory.
   - For **LoRA / QLoRA**, **prefer a `merged` suffix** in the recommended name (e.g. `./models/<base_model>-merged`, or `<base_model>-<identity>-merged`) so the merged output is clearly distinguishable from the base weights.
   - For **Full**, `merged` is misleading (nothing was merged) — recommend a plain descriptive suffix instead (e.g. `./models/<base_model>-sft`, or `<base_model>-<identity>`).
   - The user can always override.
2. Generate an export config into the **artifacts directory** (`./llamafactory_runs/<run_id>/export.yaml`), based on `examples/merge_lora/*.yaml`. **The config differs by finetuning_type:**

   - **LoRA / QLoRA** — point `model_name_or_path` at the base model and `adapter_name_or_path` at the adapter (`output_dir`):
     ```yaml
     model_name_or_path: <base_model>
     adapter_name_or_path: <output_dir>
     template: <template>                 # MUST match the training template
     trust_remote_code: true
     export_dir: <user-specified name>
     export_size: 5
     export_device: auto                  # auto = use the GPU chosen in Stage 2.5; cpu is much slower
     export_legacy_format: false
     ```
     Note: when merging LoRA/QLoRA, do **not** load a quantized model or set `quantization_bit` — merging into a quantized base produces a broken model. Merge into the full-precision base, then quantize separately if needed.
   - **Full** — point `model_name_or_path` directly at the `output_dir` and use **NO `adapter_name_or_path`**:
     ```yaml
     model_name_or_path: <output_dir>     # the full fine-tuned model (training output_dir)
     template: <template>                 # MUST match the training template
     trust_remote_code: true
     export_dir: <user-specified name>
     export_size: 5
     export_device: auto                  # auto = use the GPU chosen in Stage 2.5; cpu is much slower
     export_legacy_format: false
     ```
   - **`export_device` controls whether the merge/save runs on GPU or CPU.** `export_device: cpu` does the whole merge on CPU — it does **not** use the GPU even if you set a visibility env var, and for some models it can be very slow or appear to stall. To actually use the GPU chosen in Stage 2.5, set **`export_device: auto`** (it will place the model on the visible GPU). Prefer `auto` when a GPU is free; only fall back to `cpu` when no GPU is available (and warn the user it will be slow).
   - **Show the export config as a table and get user confirmation before running.** Just like the training yaml, after generating `export.yaml` print its key parameters as a table (for LoRA/QLoRA include `adapter_name_or_path`; for Full omit it — list `model_name_or_path`, `template`, `export_dir`, `export_size`, `export_device`, `export_legacy_format`, etc.) and use `AskUserQuestion` to let the user confirm (or request changes) before you run the export. Do not start the export until the user confirms.
3. Run (export is usually quick — foreground is fine; only training really needs background). **Run it on the GPU(s) chosen in Stage 2.5** by prefixing with the visibility env var, and tee the output into the artifacts dir for the record:
   ```bash
   # HIP_VISIBLE_DEVICES for AMD, CUDA_VISIBLE_DEVICES for NVIDIA; pair with export_device: auto to use the GPU.
   HIP_VISIBLE_DEVICES=<idx> llamafactory-cli export llamafactory_runs/<run_id>/export.yaml 2>&1 | tee llamafactory_runs/<run_id>/export.log
   ```
4. When export completes, tell the user the final model path.

---

## Final summary (end of the whole workflow)

When all chosen stages are finished, give a concise closing summary and then **stop**. The summary may include ONLY:

- the produced artifacts (config/log dir, LoRA adapter, loss curve image path, merged-model export dir);
- the training result (steps / epochs / final loss / runtime);
- the validation result (if validation ran);
- the export directory (if export ran);
- the ready-to-run command for interactive self-testing (e.g. `llamafactory-cli chat ...`).

**Do NOT propose or offer any follow-up work** — no suggestions about quantization, GGUF conversion, deployment, serving as an API, further training, or anything similar. Do not end with questions like "需要我帮你……吗？". Simply describe what was produced and stop.

---

## General requirements

- Ask first with `AskUserQuestion` at every irreversible or ambiguous decision; do not assume.
- **Confirm before every yaml write — no exceptions.** Before writing ANY yaml config to disk (train / inference / export), you MUST first print its key parameters as a table AND call `AskUserQuestion` to get the user's explicit confirmation. Only write the file after the user confirms. This applies to every config in every stage, not just training — do not "save time" by writing infer.yaml or export.yaml directly. If the user requests changes, update the table and re-confirm before writing.
- Run **long-running** commands (model download, training) **in the background + report periodically** (interval ≤ 100 seconds). Inference validation and export are usually quick and do NOT need background — run them in the foreground.
- **Detect GPU status once before running and pin the free device(s)** (see Stage 2.5). Detect a single time at the start, pick the device(s) — if multiple are free, ask single-vs-multi — then reuse the same index/indices for SFT / validation / export via the platform's visibility env var (`HIP_VISIBLE_DEVICES` for AMD, `CUDA_VISIBLE_DEVICES` for NVIDIA). Do NOT re-detect before each run. Never silently launch onto a busy card — if none are free, ask the user. For export to actually use the GPU, also set `export_device: auto` (not `cpu`).
- **Keep generated yaml configs and logs in the artifacts directory** (`./llamafactory_runs/<run_id>/`); do not write them into `./models/` (downloaded weights) or the training `output_dir` (saved checkpoints).
- **Prefer official-example defaults; track every deviation.** When generating the training yaml, start from the official example and change as little as possible. List any changed field in a "diff vs. default" table with a concrete reason (user-specified, model/dataset adaptation, etc.).
- **Keep `template` consistent across train / inference / export.** Use the template from the official example unless there's a tracked reason to change it; a train/infer mismatch produces garbled output.
- Never destroy the user's original data files in place; do not pollute the repo's built-in `data/*.json` — prefer copies.
- Keep `dataset_info.json` valid JSON after edits.
