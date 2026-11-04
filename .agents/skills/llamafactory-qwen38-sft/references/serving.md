# Artifact evaluation and serving

Use this reference only after a training artifact exists and passes the reload requirement in [training.md](training.md).

## Identify the artifact

- A QLoRA run produces an adapter. Keep the original base model and load it with `adapter_name_or_path`.
- A HyperParallel full-SFT run produces a directly loadable Hugging Face model. Do not confuse a resumable distributed checkpoint with the final Hugging Face artifact.

Confirm the expected model or adapter files, trainer state, metrics, and logs before evaluation or export.

## Evaluate

Use an independent dataset or a validation split fixed before training. Register it as a separate LlamaFactory dataset, retain stable sample IDs, and check that neither the prompts nor their reference answers occur in the training set. A smoke dataset or examples seen during training can validate plumbing but cannot establish a quality gain.

Install the repository's generation-metric dependencies with `python -m pip install -r requirements/metrics.txt` before using `predict_with_generate`; a training-only environment may not contain `jieba`, `nltk`, or `rouge-chinese`. Re-run `python -m pip check` after installation.

For instruction-following tasks, run generation on the held-out set with `do_predict: true` and `predict_with_generate: true`. Use the configuration for the artifact type instead of adapting QLoRA fields implicitly.

### QLoRA base and tuned runs

Use this configuration for both runs. Keep `model_name_or_path` on the same base model and add `adapter_name_or_path` only to the tuned run. Preserve the training run's quantization loading fields in both evaluation configurations; in particular, Ascend QLoRA requires `double_quantization: false`:

```yaml
model_name_or_path: <base-model-path>
# adapter_name_or_path: <adapter-path>  # Tuned run only.
quantization_bit: 4
quantization_method: bnb
double_quantization: <same-as-training>  # false for the repository's Ascend QLoRA path.
trust_remote_code: true

stage: sft
do_predict: true
finetuning_type: lora

eval_dataset: <held-out-dataset-name>
template: qwen3_8
# enable_thinking: false  # Optional; set deliberately and keep identical in both runs.
cutoff_len: <evaluation-cutoff>
overwrite_cache: true
preprocessing_num_workers: <worker-count>
dataloader_num_workers: <worker-count>

output_dir: <unique-evaluation-output>
overwrite_output_dir: true
report_to: none

per_device_eval_batch_size: 1
predict_with_generate: true
do_sample: false
max_new_tokens: <generation-limit>
seed: <fixed-seed>
```

### HyperParallel full-SFT base and tuned runs

Use a full-tuning evaluation configuration with no adapter fields. Point the base run at the original model and the tuned run at the final Hugging Face artifact:

```yaml
model_name_or_path: <base-model-path-or-tuned-full-model-path>
trust_remote_code: true

stage: sft
do_predict: true
finetuning_type: full

eval_dataset: <held-out-dataset-name>
template: qwen3_8
# enable_thinking: false  # Optional; set deliberately and keep identical in both runs.
cutoff_len: <evaluation-cutoff>
overwrite_cache: true
preprocessing_num_workers: <worker-count>
dataloader_num_workers: <worker-count>

output_dir: <unique-evaluation-output>
overwrite_output_dir: true
report_to: none

per_device_eval_batch_size: 1
predict_with_generate: true
do_sample: false
max_new_tokens: <generation-limit>
seed: <fixed-seed>
```

Do not carry `adapter_name_or_path`, `finetuning_type: lora`, or QLoRA quantization fields into the full-SFT configuration by accident. Prefer the deployment precision when it fits the evaluation hardware. If both full-model runs require on-the-fly 4-bit loading, add identical quantization fields to both configurations and disclose that their absolute scores may differ from unquantized evaluation.

Run each configuration with `llamafactory-cli train <eval.yaml>`. Despite the command name, `do_predict: true` performs prediction without training and writes `generated_predictions.jsonl` under `output_dir`.

Create one run for the base model and one for the tuned artifact:

- Give every run a separate output directory. Keep the held-out data, template, thinking settings, cutoff, decoding parameters, seed, and hardware policy identical.

Choose metrics before inspecting the outputs. Use deterministic checks for structured tasks, such as exact match, schema validity, required-field recall, tool-call validity, or task-specific unit tests. For open-ended answers, use a fixed rubric and blinded human or judge scoring, and retain representative wins, ties, and regressions. Token-level loss or perplexity can supplement generation metrics but should not replace task-level evaluation.

Report paired base and tuned results, the number of evaluated samples, the evaluation-data version, metric definitions, and decoding settings. Claim improvement only when the predefined acceptance criteria pass without unacceptable regressions. A decreasing training loss alone is not evidence that tuning improved behavior.

## Export a QLoRA adapter

Create the export configuration outside the model and checkpoint directories:

```yaml
model_name_or_path: <base-model-path>
adapter_name_or_path: <adapter-checkpoint-path>
template: qwen3_8
finetuning_type: lora
export_dir: <merged-model-output-path>
export_size: 5
export_device: cpu
export_legacy_format: false
```

Run `llamafactory-cli export <export.yaml>`, then reload the tokenizer, processor, and model from the merged directory. Retain the original adapter until the merged artifact passes this check. Full-SFT artifacts do not require adapter merging; reload their final Hugging Face directory directly.

## Verify direct inference and the local API

Perform direct inference first with an artifact-appropriate YAML. Then start the service locally:

```bash
API_HOST=127.0.0.1 API_PORT=8000 API_MODEL_NAME=<display-name> \
  llamafactory-cli api <inference.yaml>
```

From another terminal, verify discovery and generation:

```bash
curl -sS http://127.0.0.1:8000/v1/models
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<display-name>","messages":[{"role":"user","content":"<held-out-prompt>"}],"max_tokens":128}'
```

Inspect the response content, not only the HTTP status. If `API_KEY` is set, send a bearer token without printing or storing it. Keep the bind address local unless external exposure has been explicitly authorized and secured.

## Completion record

Report the code and package revisions, hardware, model and data versions, exact configs and commands, training and evaluation results, artifact paths, and API results. Include failures, fixes, and untested limitations. Remove credentials, internal addresses, usernames, absolute private paths, private model or dataset names, and sensitive samples before publishing.
