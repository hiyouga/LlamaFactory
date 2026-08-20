# FSDPTurbo EP/EFSDP and LlamaFactory FSDP2/CP Design

Chinese version: [FSDPTurbo EP/EFSDP 与 LlamaFactory FSDP2/CP 设计说明](../../../zh/advanced/distributed/fsdpturbo-ep-efsdp.md)

This document describes the current implementation of the `fsdpturbo` distributed plugin. Its core principle is a clear separation of responsibilities:

- FSDPTurbo owns expert parallelism (EP), expert parameter sharding (EFSDP), and device operator registration.
- LlamaFactory owns process initialization, the base DeviceMesh, outer FSDP2, CP, model initialization, and weight loading.
- The LlamaFactory integration layer combines the two parameter layouts and handles gradient norms across meshes.

## 1. Configuration Boundaries

Common parallel topology belongs to `TrainingArguments`, while FSDPTurbo-only settings remain in `dist_config`:

```yaml
cp_size: 1

dist_config:
  name: fsdpturbo
  ep_size: 16
  ep_dispatcher: eager
```

The fields used by the minimal example have the following responsibilities:

- `ep_size`: expert-parallel group size.
- `ep_dispatcher`: FSDPTurbo EP dispatcher, which defaults to `eager`.

`dp_size`, `cp_size`, `cp_mode`, `mp_replicate_size`, `mp_shard_size`, and `dist_timeout` are common topology fields and therefore remain at the top level. `dist_config` is parsed strictly as `FSDPTurboParams`; putting a common topology field inside it is rejected instead of being silently ignored.

The top-level training option `bf16` controls FSDPTurbo parameter storage and compute dtype. The backend casts the model before FSDP materialization, so `ModelEngine` does not need to read distributed-backend configuration.

The following advanced fields are optional and are therefore omitted from the minimal YAML example above:

- `fsdp_ignored_modules`: additional modules excluded from the outer LlamaFactory FSDP2 path. Expert parameters selected by the model spec are automatically added to the ignored set by the integration layer, so normal configurations do not need to repeat them here.
- `hook_modules`: optional module patterns for FSDPTurbo EFSDP hooks. The default is an empty list.
- `fsdp_implementation`: the FSDPTurbo EFSDP implementation, either `native` or `custom`. The default is `native`.

The model spec determines the EFSDP targets. Non-expert parameters such as attention, embeddings, and the LM head do not enter the FSDPTurbo EFSDP plan. They remain managed by the outer LlamaFactory FSDP2 layer.

Model-specific module paths and preparation logic are managed exclusively by the `FSDPTurboEPModelSpec` registry. Built-in specs currently cover `qwen3_moe` and `qwen3_5_moe`; unregistered models fail with an explicit error. `ep_modules` and `ep_fsdp_modules` are not YAML options, and strict parameter parsing rejects them to prevent configuration from drifting away from the actual model structure.

## 2. Mesh Initialization

LlamaFactory's `DistributedInterface` initializes only its existing model and data meshes. It is unaware of EP and EFSDP and does not expose an extra mesh registration interface for distributed plugins. The FSDPTurbo expert topology is independently created and owned by `FSDPTurboParallelState` in the plugin module:

```text
run_sft / run_dpo / run_rm
  -> DistributedInterface(training_args)
     -> initialize LlamaFactory model/data meshes
  -> DistributedPlugin("fsdpturbo").shard_model(...)
     -> FSDPTurboFSDP2Engine.__init__()
        -> FSDPTurboParallelState.initialize()
           -> initialize and retain the expert parent mesh and submeshes
```

`FSDPTurboParallelState` creates a four-dimensional expert parent mesh:

```text
(edp, efsdp, ep, expert_cp)
```

Its current dimensions are calculated as follows:

```text
dp_size       = world_size / cp_size
ep_fsdp_size  = dp_size / ep_size
edp_size      = dp_size / (ep_size * ep_fsdp_size)
mesh_shape    = (edp_size, ep_fsdp_size, ep_size, cp_size)
```

The state object retains `edp_mesh`, `efsdp_mesh`, `ep_mesh`, and `expert_cp_mesh`. Model sharding and gradient norm logic inside the plugin read expert communication domains from this state, while other LlamaFactory backends do not need to implement or know about these interfaces. Initialization validates that `ep_size` is positive and divides `dp_size`; repeated initialization also rejects topology changes.

## 3. Model Sharding Order

The wrapping order must remain "expert side first, outer FSDP2 second":

```text
DistributedPlugin("fsdpturbo")
  -> FSDPTurboFSDP2Engine.shard_model(model)
     -> prepare_model_ep(model)
        -> expert_parallelize_modules(model, ep_mesh, ep_plan)
        -> expert_fully_shard_modules(model, efsdp_mesh, ep_plan, fsdp_plan)
        -> collect expert parameters as ignored_params
     -> FSDP2Engine.prepare_model(model, ignored_params=...)
        -> apply outer fully_shard to the remaining Transformer Layers and root module
```

This prevents the same expert parameter from being managed by both EFSDP and outer FSDP2. Outer FSDP2 continues to reuse LlamaFactory's model initialization, checkpoint, and save flows.

The LlamaFactory integration layer accepts `eager`, `fused`, `mc2`, and `domino` and forwards the selected value unchanged to FSDPTurbo. Their implementation boundaries and current validation status differ:

| Dispatcher | Main path | Additional requirements | Validation in this PR |
| --- | --- | --- | --- |
| `eager` | Uses PyTorch implementations of permute, unpermute, and grouped matmul while tensors remain on the current accelerator, with standard AllToAll for token dispatch and combine | Minimal dependencies; serves as the reference implementation | End-to-end numerical and performance validation completed on Ascend A3 |
| `fused` | Keeps the same AllToAll topology while replacing permute, unpermute, and grouped matmul with device-fused operators | Requires matching device operators, dtypes, and layouts; local operators may fall back to eager when an expert receives no tokens | End-to-end numerical and performance validation completed on Ascend A3 |
| `mc2` | Uses dedicated operators that fuse AllToAllV with grouped matmul to reduce intermediate communication-computation overhead | Requires the MC2 NPU operators, an HCCL communicator, and their shape and dtype constraints | Implemented by FSDPTurbo but not validated end to end in this PR |
| `domino` | Splits the first dimension of the expert-module input into two slices and uses a separate communication stream and events to overlap AllToAll with expert computation | Requires asynchronous stream/event support and enough token work in both slices to amortize scheduling overhead | Implemented by FSDPTurbo but not validated end to end in this PR |

Only `eager` and `fused` are validated here because they cover the reference path and the commonly used A3 device-fused path, respectively, and therefore isolate and establish the correctness of the EP/EFSDP integration between LlamaFactory and FSDPTurbo. The current experiment matrix was not extended to `mc2` and `domino`: they add operator, communication-scheduling, and input-shape constraints that require separate numerical comparisons, long-run stability tests, and profiler analysis. They are accepted configuration choices, but the results in this PR should not be interpreted as evidence that they have reached the same stability, numerical, or performance level.

## 4. FSDPTurbo Dependency Entry Points

LlamaFactory imports each required object directly from the module that defines it:

```python
from fsdp_turbo.distributed.expert_parallel.expert_fully_shard_parallel import (
    expert_fully_shard_modules,
)
from fsdp_turbo.distributed.expert_parallel.expert_parallel import expert_parallelize_modules
from fsdp_turbo.fsdp_turbo_config import EPPlanConfig, FSDPPlanConfig
from fsdp_turbo.utils.str_match import module_name_match
```

The imports occur inside `prepare_model_ep()`, so other distributed backends remain importable when FSDPTurbo is not installed. They intentionally bypass aggregate exports from `fsdp_turbo.distributed.__init__` to avoid extra dependencies and potential import cycles during package initialization.

## 5. Gradient Norms

Outer and expert parameters can belong to different DTensor meshes and therefore cannot be passed together to a single standard `clip_grad_norm_()` call. The `fsdpturbo` plugin groups parameters by their owning mesh and computes local p-power sums:

- Non-expert parameters are reduced over the DP and CP groups.
- Expert parameters are reduced over the EFSDP, EP, and expert-CP groups retained by `FSDPTurboParallelState`.
- After the global norm is assembled, the same clipping coefficient is applied to every local gradient.

A zero-gradient warmup runs during startup so that the required collectives are initialized before training begins. This is currently a backend-specific implementation for `fsdpturbo`; other backends retain their existing gradient norm paths until the upstream distributed plugin interface is decoupled.

## 6. Weight Loading

LlamaFactory retains the `init_on_meta` and safetensors loading flow. The parent `FSDP2Engine` loader dynamically invokes the FSDPTurbo engine override through `self._copy_weights(...)`, so the method is not dead code. It supports DTensors with multiple `Shard` placements by calculating the local slice for the current rank along each mesh dimension in sequence. Model save and checkpoint interfaces continue to reuse the LlamaFactory FSDP2 implementation.

## 7. Kernel Plugin

FLA operators do not belong in the distributed configuration. Operator selection is handled through an independent `kernel_config`:

```yaml
kernel_config:
  name: auto, flash-linear-attention
  include_kernels: chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
  chunk_size: 32
```

The call path is:

```text
ModelEngine
  -> apply_kernels("auto, flash-linear-attention")
     -> accelerator-specific LlamaFactory auto kernels
     -> KernelPlugin("flash-linear-attention").apply(...)
        -> fsdp_turbo.ops.get_op()
           -> FSDPTurbo device operator registry
        -> fsdp_turbo.utils.patch.patch_model_members()
           -> FLA backend implementation
```

`chunk_size` accepts `16`, `32`, and `64`, with a default of `64`. The kernel plugin and distributed plugin are independent. `name: flash-linear-attention` installs only the selected FLA operators. The comma-separated `name: auto, flash-linear-attention` form composes LlamaFactory's accelerator-specific automatic kernels with the FLA plugin before distributed sharding. LlamaFactory owns the operator-to-model-attribute mapping and `chunk_size` binding; FSDPTurbo owns device operator registration, selection, and generic callable patching. FLA stays explicit because it has optional external dependencies and is not part of the built-in `auto` set. FSDPTurbo subsequently replaces the target expert module's `forward`, so the final expert execution path is selected by `ep_dispatcher`; an MoE kernel applied during the auto stage is not retained as a separate second expert execution path.

## 8. CP Runtime Constraints and Validation Scope

When `init_on_meta` constructs the model, it must propagate `attn_implementation` in the same way as the `from_pretrained` path. Otherwise, the model falls back to a non-FlashAttention implementation and Ulysses CP cannot start. Before calling Hugging Face FlashAttention, Ulysses reconstructs the global attention mask. Only two-dimensional position IDs participate in packed-sequence detection. Multi-axis position IDs such as Qwen3.5 mRoPE have already been consumed by rotary embedding and must not be passed to the FlashAttention packed-sequence detection logic.

The current implementation has completed the following BF16 AdamW full SFT validations with Qwen3.5-35B-A3B on Atlas 900 A3 SuperPoD and Atlas 950 SuperPoD systems. This revalidation used FSDPTurbo `0e96fbc`. The A3 environment used CANN 9.0.0, PyTorch 2.7.1, and torch-npu 2.7.1.post4; the A5 environment used CANN 9.1.0-beta.3, PyTorch 2.10.0, and torch-npu 2.10.0.post2. Performance is calculated from the step 1 and step 100 log timestamps and excludes initialization and compilation before the first step as well as model saving after training:

| Machine | CP | EP | EFSDP | Checkpoint | Kernel / Dispatcher | Steps | Loss (first -> last) | Performance | Result |
| --- | ---: | ---: | ---: | --- | --- | ---: | --- | ---: | --- |
| Atlas 900 A3 SuperPoD | 1 | 16 | 1 | Off | FLA (chunk size 16) / eager | 100 | 1.3361 -> 0.0793 | 2.51 s/it | Passed and saved |
| Atlas 900 A3 SuperPoD | 1 | 16 | 1 | Off | FLA (chunk size 16) / fused | 100 | 1.3354 -> 0.1179 | 2.17 s/it | Passed and saved |
| Atlas 900 A3 SuperPoD | 2 | 4 | 2 | Off | auto + FLA (chunk size 64) / fused | 100 | 1.8114 -> 0.5260 | 7.65 s/it | Passed and saved |
| Atlas 900 A3 SuperPoD | 2 | 4 | 2 | Off | auto + FLA (chunk size 64) / eager | 100 | 1.8095 -> 0.5596 | 5.88 s/it | Passed and saved |
| Atlas 950 SuperPoD | 1 | 8 | 1 | Off | no kernel plugin configured / eager | 100 | 1.3575 -> 0.4439 | 2.68 s/it | Passed and saved |

Loss and gradient norm remained finite in all five runs, and every run completed 100 steps and model saving. With the same partition, the per-step loss correlation between eager and fused was 0.997 for EP16 and 0.977 for CP2/EP4/EFSDP2, which indicates consistent optimization trajectories. The performance effect depends on the partition: fused was about 13% faster than eager with EP16, but about 30% slower after adding CP and EFSDP. Fused therefore should not be treated as the default optimum for every mesh.

The EP16 runs used global batch 16 and cutoff length 256. The CP2 runs used global batch 8 and cutoff length 128. The A5 run used global batch 8 and cutoff length 256. The first-to-last loss validates convergence within each run; absolute loss values across different partition groups should not be used directly as a numerical-equivalence conclusion.
