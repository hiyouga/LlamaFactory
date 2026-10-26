# Training workflows

Use this reference after the applicable gates in [setup.md](setup.md) pass. Resolve dataset, sequence length, sample limit, gradient accumulation, duration, and output paths for the actual hardware instead of copying them from an example.

## Path A: single-device 4-bit QLoRA

On Ascend, start from `examples/train_qlora/qwen3_lora_sft_bnb_npu.yaml`. On CUDA, start from `examples/train_qlora/qwen3_lora_sft_otfq.yaml`. Copy the selected file into the run directory, change the model to `Qwen/Qwen3.8-27B` and the template to `qwen3_8`, then resolve the dataset, output path, and run-size controls for the requested workflow.

Launch with one previously selected device:

```bash
ASCEND_RT_VISIBLE_DEVICES=<id> llamafactory-cli train <single-device-smoke.yaml>
```

Use `CUDA_VISIBLE_DEVICES` on CUDA. Accept the path only after measuring its actual loading and training peak. The output is a LoRA adapter and must later be loaded with the same base model.

## Path B: multi-device HyperParallel FSDP2 full SFT

### Ascend smoke-tested path

On Ascend, use `examples/ascend/qwen3_8_full_sft_hyper_fsdp2.yaml` directly as the repository training configuration. It already selects full SFT, HyperParallel, `qwen3_8`, batch size 1, and bf16; do not rebuild those settings from the Qwen3.5 example.

Copy `examples/accelerate/fsdp2_config_qwen38.yaml` into the run directory as well and set `num_processes` to the selected visible-device count. The example defaults to 16 processes; a smaller world size may require additional memory controls for full tuning. Its wrap classes use Qwen3.5 names because Qwen3.8 uses the `Qwen3_5` Transformers implementation.

Launch on the selected devices with the platform visibility variable:

```bash
ASCEND_RT_VISIBLE_DEVICES=<id0,id1,...> accelerate launch \
  --config_file <fsdp2-smoke.yaml> \
  src/train.py <multi-device-smoke.yaml>

```

This path is full tuning and does not use quantization. Begin with the repository example unchanged except for the model, data, output, and hardware-dependent run controls. Tune parallelism or memory settings only after that baseline is understood.

Retain the repository example's `bf16: true` because compute precision materially changes memory and training behavior. Do not repeat stable LlamaFactory defaults merely for visibility; write fields when overriding a default or when the resolved configuration must be portable across framework revisions.

### CUDA validation path

The repository does not currently ship a CUDA-specific Qwen3.8 HyperParallel example, and this workflow has not server-validated that combination. Do not present the Ascend result as CUDA support merely by replacing `ASCEND_RT_VISIBLE_DEVICES` with `CUDA_VISIBLE_DEVICES`.

When CUDA HyperParallel validation is explicitly requested, copy the checked-in training and Accelerate YAMLs into a run-specific directory, change only the model, data, output, device-count, and other demonstrated hardware-dependent controls, and launch with `CUDA_VISIBLE_DEVICES`. Apply the same real-model acceptance gate below and record the CUDA, PyTorch, Transformers, HyperParallel, and LlamaFactory revisions. Mark the result pending until the run completes; only a passing run may be reported as server-validated.

## Performance disclosure

Most Qwen3.8 layers use linear attention. Record whether `flash-linear-attention` and `causal-conv1d` enable the optimized path or Transformers falls back to its torch implementation. Do not compare or promise optimized throughput for a fallback run.

## Training acceptance

Before a formal run, execute a bounded smoke test with the selected training path and the real model. Confirm the intended backend and ranks, model loading, data preprocessing, forward, backward, and at least two consecutive optimizer updates with finite losses. A single step can finish before optimizer-state memory reaches its steady level.

For a requested training artifact, require a clean exit, complete expected files, and successful reload before evaluation, export, or serving. Record the environment, device count, exact command and configuration, duration, peak memory, loss or evaluation metrics, artifact paths, and reload result. Choose checkpoint retention deliberately, verify output space before training, and use a fresh output directory after the smoke test.
