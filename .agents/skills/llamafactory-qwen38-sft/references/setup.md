# Setup and compatibility

Run commands from the LlamaFactory repository root. Treat the current checkout's `README`, `pyproject.toml`, `requirements/`, examples, and source as authoritative.

## Record the starting state

Before installing or launching anything, record:

- the LlamaFactory commit and selected Python environment;
- accelerator type, driver and CUDA or CANN version, and selected device IDs;
- device utilization, process ownership, and available device memory;
- free space in the model-cache and training-output filesystems.

On Ascend, source the matching toolkit `set_env.sh` in every shell that invokes LlamaFactory. Confirm that the environment, `llamafactory-cli`, `torchrun`, and `accelerate` resolve from the intended installation.

## Install the selected stack

Use an isolated environment and install in this order:

1. Install the hardware-compatible PyTorch stack. On Ascend, use the torch and torch-npu pair pinned by `requirements/npu.txt` and ensure `pyyaml` is installed before importing torch-npu.
2. Install the LlamaFactory checkout with `python -m pip install -e .`.
3. Install only the optional dependencies required by the selected training path.
4. Install `transformers==5.8.0`, the version validated for this workflow and accepted by this checkout.
5. For HyperParallel, use the version declared compatible by the current LlamaFactory checkout. Until LlamaFactory declares a fixed version, use the latest `master`, resolve it to an immutable full commit SHA, and record that SHA with the run configuration. Confirm that `HyperParallelArguments` and `parallelize_model` import, then use the smallest representative training smoke test as the compatibility gate. Install a source checkout with `python -m pip install --no-deps -e <hyperparallel-checkout>`. Do not select accelerator extras, because they can replace the torch stack selected in step 1.

For Ascend QLoRA, continue with [ascend-qlora.md](ascend-qlora.md). Do not load those build instructions for CUDA QLoRA or HyperParallel runs.

Do not disable LlamaFactory's version check. If `pyproject.toml` and `check_dependencies()` declare different bounds, obey the stricter bound. Record declared constraints and installed versions separately from the smoke-test result.

## Dependency gates

Run the gates required by the selected path:

```bash
python -m pip check
llamafactory-cli version
python -c 'import torch, transformers, accelerate, peft; assert transformers.__version__ == "5.8.0"; print(torch.__version__, transformers.__version__, accelerate.__version__, peft.__version__)'
python -c 'import torch_npu; print(torch_npu.__version__)'  # Ascend only
python -c 'import bitsandbytes'                            # QLoRA only
python -c 'import hyper_parallel; from hyper_parallel.integration.llamafactory import HyperParallelArguments, parallelize_model; print(hyper_parallel.__file__, HyperParallelArguments())'  # HyperParallel only
```

For Ascend QLoRA, complete the backend gate in [ascend-qlora.md](ascend-qlora.md); importing bitsandbytes alone does not prove that its NPU backend works. For HyperParallel, verify the printed checkout and both public integration symbols. When using `master`, an unrecorded moving branch or the development version string alone is not reproducible evidence. An import gate does not replace the training smoke test.

For this workflow, HyperParallel's declared `transformers>=5.11.0` requirement is a known metadata-only conflict with the validated `transformers==5.8.0` stack. Install HyperParallel from the recorded source revision with `--no-deps`, report this exact `pip check` finding, and do not treat this finding alone as a failed gate. Do not ignore any other dependency conflict. The HyperParallel integration imports and the real multi-device smoke test remain mandatory compatibility gates.

## Model compatibility and cache gates

Before a full download or training launch, confirm that the current checkout maps `Qwen3.8-27B` to `Qwen/Qwen3.8-27B` and template `qwen3_8`. Confirm with Transformers 5.8.0 that:

- `AutoConfig.model_type` is `qwen3_5`;
- the model class is `Qwen3_5ForConditionalGeneration`;
- the tokenizer, processor, and LlamaFactory template load.

Before declaring a local model complete, require:

- every shard named by `model.safetensors.index.json`, validated against the resolved upstream revision;
- `config.json`, tokenizer files, `preprocessor_config.json`, and `video_preprocessor_config.json`;
- no relevant incomplete download files;
- read access for the user that will run training.

Stop at a failed gate, preserve its log, and report the failure without substituting another model, backend, or training method.
