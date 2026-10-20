# Ascend 4-bit QLoRA dependency path

Read this reference only for single-device 4-bit QLoRA on Ascend. The LlamaFactory checkout's current README and `examples/train_qlora/qwen3_lora_sft_bnb_npu.yaml` remain authoritative.

Build the NPU backend from the `multi-backend-refactor` branch of `bitsandbytes-foundation/bitsandbytes` with CMake `COMPUTE_BACKEND=npu`; a regular CUDA-oriented bitsandbytes wheel is not an Ascend acceptance result.

Prefer building a wheel in an isolated build environment and installing that wheel into the training environment with `--no-deps`. Record the bitsandbytes commit and wheel checksum so the environment can be reproduced without letting the build toolchain replace the selected Python or PyTorch stack.

Before training:

1. Import bitsandbytes from the selected training environment and confirm that the NPU shared library, rather than a CUDA fallback, is loaded.
2. Run a small NPU tensor-quantization operation. Import success alone is insufficient.
3. Preserve `double_quantization: false` from the repository's NPU example unless a separately validated backend revision supports otherwise.

If import or the backend operation fails, preserve the build and runtime logs before troubleshooting. Inspect `libbitsandbytes_npu.so` with `readelf` and `ldd`, confirm that its dependencies resolve, and check whether its RPATH or RUNPATH refers to a removed build environment. Stop without replacing the requested QLoRA path with another tuning method when the backend gate remains unresolved.
