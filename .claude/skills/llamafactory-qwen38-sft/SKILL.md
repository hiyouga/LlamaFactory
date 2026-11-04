---
name: llamafactory-qwen38-sft
description: Run end-to-end LlamaFactory SFT for Qwen/Qwen3.8-27B using single-device 4-bit QLoRA or multi-device HyperParallel FSDP2 full tuning, including setup, training validation, artifact evaluation, export, and local serving. Use only for Qwen3.8-27B.
---

# Qwen3.8-27B SFT with LlamaFactory

Drive LlamaFactory directly from environment setup through a verified local artifact. Do not build a separate agent runtime or training harness.

## Model contract

- Use the exact model `Qwen/Qwen3.8-27B`; do not substitute Qwen3-8B or Qwen3.5-27B.
- Use the LlamaFactory template `qwen3_8`. Transformers correctly exposes the architecture as `qwen3_5` / `Qwen3_5ForConditionalGeneration`.
- Treat the model as multimodal and choose text-only or multimodal data deliberately. Preserve the repository example's model-component defaults unless the requested tuning scope requires an override.
- Keep the `qwen3_8` template when changing thinking behavior. Record explicit `enable_thinking` or `reasoning_effort` overrides with the run configuration.
- Use single-device 4-bit QLoRA or multi-device HyperParallel with Accelerate FSDP2 full tuning. The checked-in HyperParallel recipe is smoke-tested on Ascend; treat CUDA HyperParallel and artifact-level completion as pending until they independently pass the applicable acceptance gates. Do not silently replace either path when a dependency or capacity gate fails.

## References

Load only the material needed for the requested stage:

- For environment creation, dependency gates, hardware selection, or model-cache validation, read [setup.md](references/setup.md).
- For the Ascend 4-bit QLoRA dependency path, additionally read [ascend-qlora.md](references/ascend-qlora.md).
- For training configuration, QLoRA, HyperParallel, or the training smoke test, read [training.md](references/training.md).
- For evaluation, artifact validation, export, inference, or local API verification, read [serving.md](references/serving.md).
- For the complete workflow, read them in that order.

## Workflow

1. Establish the requested scope, dataset, accelerator, selected devices, and time and disk limits.
2. Record the LlamaFactory revision and validate the environment and local model before training.
3. Validate and register custom data using the current repository documentation; preserve source data and keep held-out samples separate.
4. Resolve the training parameters and keep generated configuration, logs, and artifacts in explicit run-specific directories.
5. Run the smallest representative smoke test at the requested validation scope, then use a fresh output directory for a formal run.
6. When an artifact is requested, reload it before export or serving. Compare the base and tuned models with the same held-out prompts and decoding settings.
7. Report versions, hardware, configs, commands, metrics, artifact paths, failures, fixes, and untested limitations. Mark conclusions as code-confirmed, server-validated, or pending.

## Execution boundaries

- Pin the selected devices with the platform visibility variable and do not share occupied devices without explicit agreement.
- Run long downloads and training jobs with durable logs. Verify the real worker process and output artifacts; a detached launcher exiting successfully is not completion.
- Keep base weights, generated configs and logs, and training artifacts in separate paths. Record their resolved absolute paths.
- Bind services to `127.0.0.1` by default. External exposure requires explicit authorization, authentication, and a network plan.
- Never print or persist credentials.
- Do not alter model code merely to make the workflow pass. Preserve reproducible framework failures and treat their fixes as separately testable changes.

## Package maintenance

Treat `.agents/skills/llamafactory-qwen38-sft` as the canonical package. When changing it, update the `.claude/skills/llamafactory-qwen38-sft` compatibility mirror byte-for-byte and verify both directories with `diff -qr` before committing. This rule is for skill maintenance and does not add a step to an SFT run.
