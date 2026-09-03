# 分布式训练

训练命令检测到多设备后会自动通过 `torchrun` 启动。拓扑字段属于 `TrainingArguments`，后端专属字段放在 `dist_config`。

## FSDP2

```yaml
dist_config:
  name: fsdp2
  reshard_after_forward: true
  offload_params: false
  pin_memory: true
  dcp_path: null
```

## FSDPTurbo

FSDPTurbo 在 FSDP2 基础上提供 MoE 专家并行和专家参数分片。先安装 FSDPTurbo 依赖：

```bash
python -m pip install -r requirements/fsdpturbo.txt
```

然后在 `dist_config` 中选择 `fsdpturbo`：

```yaml
dist_config:
  name: fsdpturbo
  ep_size: 16
  ep_dispatcher: eager
```

`ep_size` 必须能够整除 data parallel size。完整示例见 `examples/v1/train_full/train_full_qwen3_moe_fsdpturbo_ep_fsdp.yaml`。

## DeepSpeed

```yaml
dist_config:
  name: deepspeed
  config_file: examples/deepspeed/ds_z3_config.json
```

`config_file` 是必填字段。

## Ulysses Context Parallel

```yaml
flash_attn: flash_attention_2
cp_mode: ulysses
cp_size: 2

dist_config:
  name: fsdp2
```

设置 `cp_size > 1` 后，`BaseTrainer` 根据 `cp_mode` 调用 `SequenceParallelModelPlugin("ulysses")`，并使用 Sequence Parallel loss 完成跨 CP 进程的损失聚合。Ulysses 需要 `flash_attention_2` 和 FSDP2，但不要求特定的 `batching_strategy`；可以根据数据类型和序列长度选择 `normal` 或支持的 padding-free 策略。

`cp_size` 需要能够整除 world size。模型的 attention head 数需要能够整除 `cp_size`，KV head 数需要能够整除 `cp_size`，或能够被 `cp_size` 整除。当前只有 SFT 支持 `cp_size > 1`；DPO 和 RM 要求 `cp_size: 1`。

## 配置并行拓扑

```yaml
dp_size: 4
cp_size: 2
cp_mode: ulysses
mp_replicate_size: 2
mp_shard_size: 4
dist_timeout: 18000
```

上例使用 8 个进程，并同时构造两套 DeviceMesh：

- Data Mesh 的形状为 `dp_size × cp_size = 4 × 2`，分别用于 Data Parallel 和 Context Parallel。
- Model Mesh 的形状为 `mp_replicate_size × mp_shard_size = 2 × 4`。FSDP 在 4 个进程间分片参数，并在 2 个分片组间复制参数。

`mp_replicate_size` 和 `mp_shard_size` 描述 FSDP 的二维参数 Mesh，不是额外的 Tensor Parallel 配置。未显式指定时，`dp_size` 默认为 `world_size / cp_size`，`mp_shard_size` 默认为 `world_size / mp_replicate_size`。后端完整配置见[训练参数](../configuration/training.md#dist_config)。

## 配置多机启动

CLI 读取 `NNODES`、`NODE_RANK`、`NPROC_PER_NODE`、`MASTER_ADDR` 和 `MASTER_PORT`。例如使用 4 台机器、每台机器 8 个设备时，在每台机器上执行：

```bash
NNODES=4 \
NODE_RANK=<0到3，各节点不同> \
NPROC_PER_NODE=8 \
MASTER_ADDR=<rank 0 节点的 IP> \
MASTER_PORT=29500 \
llamafactory-cli sft config.yaml
```

4 个节点需要使用相同的 `NNODES`、`NPROC_PER_NODE`、`MASTER_ADDR` 和 `MASTER_PORT`，并分别设置 `NODE_RANK=0`、`1`、`2`、`3`。
