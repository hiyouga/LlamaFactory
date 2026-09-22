# LightOnOCR-2 Integration for LLaMA-Factory

## Overview

[LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B) is a compact 1B-parameter
end-to-end multilingual vision-language model for state-of-the-art OCR. It converts document
images (PDFs, scans, photos) into clean, naturally ordered text without brittle multi-stage
OCR pipelines.

- **Paper**: [arXiv:2601.14251](https://arxiv.org/abs/2601.14251)
- **Blog**: <https://huggingface.co/blog/lightonai/lightonocr-2>
- **License**: Apache 2.0

### Architecture

| Component              | Detail                                                       |
|------------------------|--------------------------------------------------------------|
| Vision Encoder         | Pixtral ViT (from Mistral-Small-3.1), native resolution, patch size 14 |
| Multimodal Projector   | 2-layer MLP with GELU, spatial merge factor 2 (4x token reduction) |
| Language Model Decoder | Qwen3 (28 layers, 1024 hidden, 16 heads, 8 KV heads)        |
| Parameters             | ~1B total                                                    |
| Max Resolution         | 1540 px longest edge                                         |
| model_type             | `lighton_ocr` (auto-patched from `mistral3`)                 |
| Chat Format            | ChatML (`<\|im_start\|>` / `<\|im_end\|>`)                  |
| Image Token            | `<\|image_pad\|>`                                            |

### Key Difference from GLM-OCR

LightOnOCR-2 performs OCR **without explicit task prompts**. The extraction behavior is
embedded in the model weights. The user message contains only the image — no
"Text Recognition:" text is needed. The model natively outputs Markdown with LaTeX math spans.

---

## Available Checkpoints

All checkpoints are registered in LLaMA-Factory with `template=lighton_ocr`:

| Model Name                     | HuggingFace ID                              | Description                                |
|--------------------------------|---------------------------------------------|--------------------------------------------|
| `LightOnOCR-2-1B`             | `lightonai/LightOnOCR-2-1B`                | Best OCR model (base + RLVR)              |
| `LightOnOCR-2-1B-base`        | `lightonai/LightOnOCR-2-1B-base`           | Supervised pretraining baseline            |
| `LightOnOCR-2-1B-bbox`        | `lightonai/LightOnOCR-2-1B-bbox`           | OCR + image bounding box prediction        |
| `LightOnOCR-2-1B-bbox-base`   | `lightonai/LightOnOCR-2-1B-bbox-base`      | Bbox variant supervised baseline           |
| `LightOnOCR-2-1B-bbox-soup`   | `lightonai/LightOnOCR-2-1B-bbox-soup`      | Task-arithmetic merge (OCR + bbox)         |
| `LightOnOCR-2-1B-ocr-soup`    | `lightonai/LightOnOCR-2-1B-ocr-soup`       | Checkpoint averaged OCR variant            |

**Recommended for fine-tuning**: `LightOnOCR-2-1B-base` (clean supervised checkpoint,
no RLVR artifacts that could interfere with domain-specific fine-tuning).

---

## Integration Details

### Files Modified

**`src/llamafactory/data/template.py`** — Registered the `lighton_ocr` template:

```python
register_template(
    name="lighton_ocr",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin(name="pixtral", image_token="<|image_pad|>"),
)
```

Design decisions:

- **ChatML format** matches the model's Jinja chat template from `tokenizer_config.json`.
- **`pixtral` mm_plugin** because the vision encoder is Pixtral-based (from Mistral-Small-3.1).
  The `PixtralPlugin` correctly handles `image_break_token` (`<|vision_pad|>`) and
  `image_end_token` (`<|vision_end|>`) from the processor at runtime.
- **`<|image_pad|>` image token** matches the model's tokenizer configuration.
- **`replace_eos=True`** so `<|im_end|>` becomes the EOS token (matching the model config
  where `eos_token_id=151645` = `<|im_end|>`).
- **No `default_system`** because LightOnOCR-2 is trained without explicit system prompts.

**`src/llamafactory/extras/constants.py`** — Registered all 6 LightOnOCR-2 checkpoints
with `template="lighton_ocr"` and `multimodal=True`.

**`src/llamafactory/model/model_utils/visual.py`** — Registered `lighton_ocr` composite model
with correct weight names (different from Mistral3):

```python
_register_composite_model(
    model_type="lighton_ocr",
    projector_key="model.vision_projection",
    vision_model_keys=["vision_encoder"],
)
```

**`src/llamafactory/model/model_utils/lightonocr.py`** — Auto-patcher module that
transparently fixes LightOnOCR-2 configs at load time (see below).

**`src/llamafactory/model/loader.py`** — Calls the auto-patcher before any
`AutoConfig` / `AutoProcessor` loading.

### Config Auto-Patching (Important)

LightOnOCR-2 models on HuggingFace ship with `model_type: "mistral3"` in their
`config.json`.  However, transformers >= 5.1 has a **native** `lighton_ocr` model
type that uses the correct weight naming (`vision_encoder` / `vision_projection`
instead of Mistral3's `vision_tower` / `multi_modal_projector`).  Without patching,
**the vision encoder loads with random weights** and training is useless.

Additionally, the `processor_config.json` stores `patch_size` as a bare integer,
causing thousands of noisy log messages during training.

**This is handled automatically.**  When LlamaFactory loads any LightOnOCR-2 model,
the auto-patcher (`model_utils/lightonocr.py`) detects and fixes both issues
in-place.  The patch is idempotent and only runs when needed.

You can also run the standalone script manually:

```bash
# Patch a specific model
python scripts/patch_lightonocr.py lightonai/LightOnOCR-2-1B-base

# Patch all cached LightOnOCR models
python scripts/patch_lightonocr.py --all
```

### Files NOT Modified (no changes needed)

- **`src/llamafactory/data/collator.py`**: LightOnOCR-2 does NOT use mRoPE (unlike
  GLM-OCR / Qwen2-VL), so no 3D position ID handling is needed.

---

## Dataset Preparation

### 1. Convert PAGE-XML / ALTO-XML to ShareGPT Format

Use the provided conversion script:

```bash
python convert_pagexml_to_lightonocr_sharegpt.py \
    --input_dir /path/to/your/xml-dataset \
    --output_dir ./data \
    --dataset_name my_ocr_dataset \
    --format auto \
    --unicode_form NFC
```

This produces:
- `data/my_ocr_dataset.json` — ShareGPT-format JSON
- `data/my_ocr_dataset/` — Cropped image files

The key difference from the GLM-OCR conversion script is the **user prompt format**:

| Model         | User Content                      |
|---------------|-----------------------------------|
| GLM-OCR       | `<image>Text Recognition:`        |
| LightOnOCR-2  | `<image>`                         |

### 2. Register the Dataset

Add to `data/dataset_info.json`:

```json
{
  "my_ocr_dataset": {
    "file_name": "my_ocr_dataset.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  }
}
```

### 3. Sample JSON Entry

```json
{
  "messages": [
    {"role": "user", "content": "<image>"},
    {"role": "assistant", "content": "transcribed text here"}
  ],
  "images": ["my_ocr_dataset/abc123def456.png"]
}
```

---

## Training

### LoRA SFT (recommended)

Use the provided example config:

```bash
llamafactory-cli train lightonocr_lora_sft.yaml
```

Or with a custom config:

```yaml
### model
model_name_or_path: lightonai/LightOnOCR-2-1B-base
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: my_ocr_dataset
template: lighton_ocr
cutoff_len: 4096

### output
output_dir: saves/lightonocr/my-dataset-lora/sft
logging_steps: 100
save_steps: 2000
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 5
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true

### eval
do_eval: true
val_size: 0.1
per_device_eval_batch_size: 4
eval_strategy: steps
eval_steps: 2000

### early stopping
load_best_model_at_end: true
metric_for_best_model: eval_loss
greater_is_better: false
early_stopping_steps: 3  # stop after 3 evals without improvement
```

### GPU Memory Requirements

| GPU             | Batch Size | Grad Accum | Effective Batch | Quantization |
|-----------------|------------|------------|-----------------|--------------|
| RTX 3060 12GB   | 2          | 8          | 16              | 4-bit (QLoRA)|
| RTX 3090 24GB   | 4          | 4          | 16              | Optional     |
| A100 40GB       | 8          | 2          | 16              | None needed  |

### Training Tips

- **`cutoff_len: 4096`** — LightOnOCR-2 supports up to 6144 tokens during pretraining.
  For line-level OCR crops 1024–2048 is enough; for full pages use 4096.
- **`learning_rate: 1e-4`** — LoRA benefits from a higher learning rate than full fine-tuning
  because updates are inherently smaller (scaled by `lora_alpha / lora_rank`). The original
  paper used `6e-5` for full-weight training; for LoRA, `1e-4` to `2e-4` is standard.
- **`lora_target: all`** — LoRA is applied to all linear modules. The `lighton_ocr` composite
  model registration ensures the vision encoder is excluded by default when
  `freeze_vision_tower: true` (the default).
- **Avoid truncation** for multimodal inputs — image token counts depend on resolution,
  and truncating can cause token mismatches with the vision encoder.

---

## Comparison with GLM-OCR

| Feature                | GLM-OCR                          | LightOnOCR-2                     |
|------------------------|----------------------------------|----------------------------------|
| Template name          | `glm_ocr`                        | `lighton_ocr`                    |
| model_type             | `glm_ocr`                        | `lighton_ocr` (auto-patched)     |
| Vision encoder         | GLM4V (Qwen2-VL style)          | Pixtral ViT (Mistral-Small-3.1) |
| mm_plugin              | `glm4v`                          | `pixtral`                        |
| Image token            | `<\|image\|>`                    | `<\|image_pad\|>`               |
| User prompt            | `<image>Text Recognition:`       | `<image>` (image only)           |
| Position IDs           | 3D mRoPE required                | Standard (no mRoPE)              |
| Language decoder       | GLM4 / ChatGLM                   | Qwen3                            |
| Output format          | Plain text                       | Markdown with LaTeX              |
| Parameters             | ~1.5B                            | ~1B                              |
| Conversion script      | `convert_pagexml_to_glmocr_sharegpt.py` | `convert_pagexml_to_lightonocr_sharegpt.py` |

---

## Conversion Script Options

```
usage: convert_pagexml_to_lightonocr_sharegpt.py [-h] --input_dir INPUT_DIR
    [--output_dir OUTPUT_DIR] [--dataset_name DATASET_NAME]
    [--format {pagexml,alto,auto}] [--unicode_form {NFC,NFD,NFKC,NFKD}]
    [--min_text_length N] [--min_crop_size N]
    [--include_full_pages] [--no_full_pages]
    [--include_paragraphs] [--no_paragraphs]
    [--paragraph_min_lines N] [--paragraph_max_lines N]
    [--line_separator SEP] [--batch_size N]
    [--image_format {png,jpg,jpeg}] [--symlink_images]
    [--image_dir DIR] [--verbose] [--max-files N]
```

The script creates three levels of training samples:
1. **Line-level**: Individual text lines cropped from polygon coordinates
2. **Paragraph-level**: Groups of 5–10 consecutive lines merged into one crop
3. **Full-page**: The entire page image with all transcriptions joined

All samples use `<image>` as the user content (no text prompt).
