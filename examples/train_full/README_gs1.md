# Native GS1 decision training

The default (v0) training CLI supports GS1 checkpoints through `stage: decision`.
This stage uses GS1's native compiler and decision head, not chat conversion or
next-token prediction. No changes to or installation of the GS1 repository are
required; model code is loaded from the checkpoint with explicit remote-code trust.

## Run

From the repository root, in a virtual environment with the appropriate PyTorch build:

```bash
pip install -e .
pip install -r requirements/gs1.txt
# Preserve the official warmup/joint memberships and native seeded joint order.
python examples/train_full/prepare_gs1_decision.py \
  /path/to/sampled-3000.jsonl /path/to/distribution-v4.5 data/gs1_stages
# Stage 1: NLL; head, seven structural rows and AttnRes routers.
FORCE_TORCHRUN=1 USE_V1=0 llamafactory-cli train \
  examples/train_full/gs1_nll_decision.yaml dataset_dir=data/gs1_stages dataset=gs1_nll
# Stage 2: NLL+Brier; load weights with a fresh optimizer (not resume_from_checkpoint).
FORCE_TORCHRUN=1 USE_V1=0 llamafactory-cli train \
  examples/train_full/gs1_full_decision.yaml dataset_dir=data/gs1_stages dataset=gs1_joint \
  model_name_or_path=saves/gs1-0.8b/full/nll
```

The example pins the published GS1-0.8B revision. All training settings use normal
LLaMA Factory arguments. `decision_brier_weight` is the only additional loss
parameter (default 1.0; set 0 for distribution NLL alone).

The demo is a schema/launch example, not a dataset for measuring decision quality.
Use a separate held-out dataset for evaluation:

```bash
llamafactory-cli train examples/train_full/gs1_full_decision.yaml \
  do_eval=true eval_dataset=your_gs1_eval
```

## Register native data

Keep the original `request` and `targets` objects in a local JSONL file. Register it
in `data/dataset_info.json` (or the registry in your chosen `dataset_dir`):

```json
{
  "my_gs1_train": {
    "file_name": "my_gs1_train.jsonl",
    "formatting": "gs1"
  }
}
```

See `data/gs1_demo.jsonl` for Choice, Noul and Score examples, including a soft
target and multiple questions in one record. Target `candidate_ids` are matched
by identity, not by array position. Distributions must be finite, nonnegative and
sum to one. As in the native 4B recipe, every logical question has unit weight;
`provenance.sample_weight` and `question_loss_weights` do not scale the objective.

```bash
CUDA_VISIBLE_DEVICES=0 FORCE_TORCHRUN=1 USE_V1=0 \
  llamafactory-cli train examples/train_full/gs1_full_decision.yaml \
  dataset=my_gs1_train max_samples=3000 output_dir=saves/gs1/my-run
```

`max_samples` limits original records per dataset **before** flattening questions;
3000 records can contain more than 3000 decision questions. Use an already sampled
JSONL file for reproducible random sampling. Neither labels nor input text are
rewritten. A branch exceeding `cutoff_len` raises an error; it is never truncated.
Keep train and evaluation records in separate registered datasets.

## Objective and scope

Each logical question is one training example. Warmup minimizes distribution NLL;
the joint stage minimizes `distribution_NLL + expected_Brier` (1:1), averaged over
the actual global question count across all accumulated microbatches and ranks.
Zero-loss tail rows prevent distributed samplers from repeating real questions.
Each stage visits its selected questions once; the stage datasets are disjoint.
Expected multiclass Brier is `sum(p**2 - 2*p*target + target)`; probabilities and
losses use FP32 and zero target mass never multiplies padded `-inf` logits.

Supported: text-only native GS1, full text-backbone/head training, CPU/single-device
and DDP, DeepSpeed ZeRO-2/3, FSDP1/2, gradient accumulation/checkpointing, loss-only evaluation,
standard Trainer checkpoint/resume and native `save_pretrained` export.
`freeze_vision_tower` defaults to true. Set `freeze_language_model: true` to train
only the decision head. `decision_warmup: true` instead trains the head, seven
structural token embedding rows and AttnRes routers, with `decision_brier_weight: 0`.
Other embedding rows receive zero gradient and zero decay; this uses native
embedding parameters so sharded checkpoints and exports need no special format.
It allocates optimizer state for the whole embedding matrix, unlike the local
trainer's compact seven-row delta. The calibrator remains frozen.

`learning_rate` is the head peak rate. Warmup embedding/router rates are 0.2/0.5
of that rate; joint backbone/router rates are 0.5. Router and warmup embedding
decay are zero; other parameters use `weight_decay: 0.01`. The examples use seed
41, norm cap 1.0 and the native cosine schedule: 10% initial/final rates, 20/100
warmup steps, clamped for shorter stages. On 16 GPUs the example accumulation of
256 gives global batch 4096; adjust accumulation for another device count.

This reproduces the supervised two-stage adaptation recipe from an existing GS1
checkpoint. It does not initialize a new decision model from raw Qwen weights or
reproduce the original 4B model's complete training history. The pinned example
is 0.8B; supply the intended GS1 checkpoint for another model size.

ZeRO-3 gathers the native text model and decision head as leaf modules so variable
length cohorts and frozen-weight checkpoint replay have identical collectives.
Optimizer states remain sharded; peak parameter memory includes a full text model.

The implementation excludes LoRA/quantization, packing, streaming, token-level
metrics/losses, generation, and v1/WebUI integration.
`template` is omitted because GS1 owns its prompt format. Do not enable causal-LM
kernels or change the vocabulary. Unsupported settings fail explicitly.

Resume with the normal `resume_from_checkpoint=...` argument. Processor assets
are retained in both checkpoints and final model exports; the exported model
continues to work with `AutoModel.from_pretrained(..., trust_remote_code=True)`
and `model.eval().decide(...)`.

## Multiple GPUs and machines

Use the same decision configuration and dataset with LLaMA Factory's standard
launcher and backend arguments. No GS1-specific launcher or distributed trainer
is needed. For example, on one node with eight allocated GPUs:

```bash
FORCE_TORCHRUN=1 USE_V1=0 NPROC_PER_NODE=8 \
  llamafactory-cli train examples/train_full/gs1_full_decision.yaml
```

Select a sharded backend by appending one of the following argument groups to
that command (do not combine DeepSpeed and FSDP):

| Backend | Arguments |
| --- | --- |
| DDP | None |
| DeepSpeed ZeRO-2 | `deepspeed=examples/deepspeed/ds_z2_config.json` |
| DeepSpeed ZeRO-3 | `deepspeed=examples/deepspeed/ds_z3_config.json` |
| FSDP1 | `'fsdp=[full_shard,auto_wrap]' fsdp_config=examples/train_full/gs1_fsdp1.json` |
| FSDP2 | `'fsdp=[full_shard,auto_wrap]' fsdp_config=examples/train_full/gs1_fsdp2.json` |

Install `requirements/deepspeed.txt` for DeepSpeed. FSDP uses the PyTorch and
Accelerate dependencies already required by the training environment. Existing
DeepSpeed CPU-offload configurations can also be supplied; they additionally
require a compatible local CPUAdam build/toolchain. Offload and vendor-specific
parallelism are separate from the basic backend tests.

For two machines, launch the same command on **both** nodes with a shared,
reachable rendezvous address and port. Set `NODE_RANK=0` on the first machine and
`NODE_RANK=1` on the second. Replace `192.168.1.10` below with the first node's
reachable address. The model, data, output directory and configuration
must be available at the same paths on every node:

```bash
# First machine; run concurrently with the second-machine command.
FORCE_TORCHRUN=1 USE_V1=0 NNODES=2 NODE_RANK=0 NPROC_PER_NODE=8 \
MASTER_ADDR=192.168.1.10 MASTER_PORT=29500 \
  llamafactory-cli train examples/train_full/gs1_full_decision.yaml \
  deepspeed=examples/deepspeed/ds_z3_config.json

# Second machine.
FORCE_TORCHRUN=1 USE_V1=0 NNODES=2 NODE_RANK=1 NPROC_PER_NODE=8 \
MASTER_ADDR=192.168.1.10 MASTER_PORT=29500 \
  llamafactory-cli train examples/train_full/gs1_full_decision.yaml \
  deepspeed=examples/deepspeed/ds_z3_config.json
```

The same `NNODES` / `NODE_RANK` settings apply to DDP and FSDP. With existing
Slurm allocations, execute each node's command using
`srun --jobid=<allocation-id> --overlap -N1 -n1 ...`; start one launcher per node,
not one launcher per GPU. A new Slurm allocation can instead start one launcher
per node with `NODE_RANK=$SLURM_PROCID`.

The effective batch size is `per_device_train_batch_size *
gradient_accumulation_steps * world_size`, where each example is one logical
question. Adjust accumulation when changing the number of GPUs. Distributed
sampling, gradient synchronization, evaluation gathering, optimizer sharding and
checkpoint/resume are handled by the existing Trainer/Accelerate implementation.
Resume using the same backend and topology with `resume_from_checkpoint=...`.
FSDP2 keeps distributed training shards in intermediate checkpoints. After
training, the stage switches to `FULL_STATE_DICT` for a final native pretrained
export; use that final output directory for `AutoModel.from_pretrained`.

GS1's published strict loader first materializes the checkpoint on each rank's
CPU; ZeRO-3 partitioning is deferred until Trainer prepares the model. Initial
loading therefore needs CPU memory for one complete model per process, even
though training parameters and optimizer states are subsequently sharded. FSDP
parameters are normalized to FP32 before wrapping to accommodate the native
checkpoint's mixed BF16/FP32 weights; BF16/FP16 forward computation still follows
the standard mixed-precision configuration. Trainable weights outside ZeRO-3
are kept in FP32, matching the standard full-tuning precision policy; ZeRO-3
manages its own master weights. FSDP1 needs `use_orig_params: true`
(as in the supplied configuration) to mix frozen vision/calibrator parameters
with trainable parameters.

For a shared NFS cache, isolate Triton compilation caches by node, for example
`export TRITON_CACHE_DIR="$PWD/.cache/triton-$(hostname)"`, on both machines.
Choose NCCL interfaces according to the cluster network. If RDMA is unavailable,
a TCP fallback uses `NCCL_IB_DISABLE=1` and `NCCL_SOCKET_IFNAME=<network-interface>`;
these are environment settings, not model adaptations. Avoid a shared cache
between machines with different GPU architectures. Tensor, pipeline, expert and
context parallelism (including the v1-only distributed plugins) are outside this
v0 decision-stage integration.

The standard Accelerate entry point is also available. For example, FSDP1 on
one node (adjust the existing configuration's process count as needed):

```bash
USE_V1=0 accelerate launch --config_file examples/accelerate/fsdp_config.yaml \
  src/train.py examples/train_full/gs1_full_decision.yaml
```
