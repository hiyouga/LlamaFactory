# CIG-SFT

CIG-SFT is available on LLaMA-Factory's standard (`llamafactory-cli train`) Transformers SFT path. The first integration supports text-only full-parameter and freeze fine-tuning. Its frozen reference model uses the existing DeepSpeed and FSDP model-placement helpers. It keeps the normal SFT fallback when `use_cig_sft_loss` is false.

## Quick start

Start from `examples/extras/cig_sft/qwen25_full_sft.yaml`:

```bash
llamafactory-cli train examples/extras/cig_sft/qwen25_full_sft.yaml
```

The recipe sets `cig_erased_prompt` to `default`. The reference model runs the teacher-forced original and erased-input views to produce next-token probability distributions. For each gold response token, the implementation extracts its conditional probability and compares the corresponding log probabilities; the resulting detached per-token credits are stored in `cig_credit_cache` and reused by later runs with the same reference model and tokenized examples. `cig_lambda`, `cig_eps`, and `cig_max_beta` control the projection and numerical safeguards.

## Dataset columns

The ordinary Alpaca and ShareGPT formats continue to work. A dataset can provide a replacement prompt for every row by registering an additional column in `data/dataset_info.json`:

```json
{
  "my_cig_dataset": {
    "file_name": "my_cig_dataset.jsonl",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "erased_prompt": "erased_instruction"
    }
  }
}
```

A non-empty row value overrides the dataset-level `cig_erased_prompt`; blank or null values fall back to the dataset-level value. The built-in presets are `default`, `math`, `code`, and `medical`; any other string is treated as a literal replacement prompt.

## Supported scope and limitations

The current path requires one user message and one assistant response per example. It rejects packing, streaming datasets, multimodal erased views, Megatron Bridge, MCA, HyperParallel, LoRA/OFT, and other non-standard SFT backends. The frozen reference model remains available while training; once a credit is cached, the training step reuses it without another reference-model forward. Evaluation and generation use the ordinary model inputs and loss.

Set `use_cig_sft_loss: false` (or remove it) to return to standard SFT. CIG-SFT cannot be combined with DFT, ASFT, or EAFT.
