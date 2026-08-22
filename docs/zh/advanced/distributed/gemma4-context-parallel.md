# Gemma 4 上下文并行

LLaMA Factory 支持使用 DeepSpeed 对纯文本 Gemma 4 SFT 启用 Ulysses 上下文并行。注意力全交换通信和
Gemma 4 Triton 内核由 `gemma-triton-flash-attn>=0.2.0` 提供。

从维护中的 fork 安装不可变版本：

```bash
pip install "gemma-triton-flash-attn[hf] @ git+https://github.com/StevenShi-23/gemma-triton-flash-attn.git@v0.2.0"
```

将 `flash_attn` 设为 `triton_gqa`，并将 `context_parallel_size` 设为分布式 world size 的因数。在 8 卡
训练中，`context_parallel_size: 4` 会建立两个数据并行组，每组包含 4 个上下文并行 rank。完整配置参见
`examples/train_lora/gemma4_lora_sft_cp.yaml`。

首个公开版本有意将支持范围限制为：

- 使用 DeepSpeed 和 BF16 的 SFT；
- `per_device_train_batch_size: 1` 且 `dataloader_drop_last: true`；
- 使用 `average_tokens_across_devices: true` 对全局移位 token loss 进行归一化；
- 不使用 packing 的纯文本数据，并冻结视觉塔和多模态投影层；
- 同一训练任务中不启用评估、生成、标签平滑、自定义 SFT loss 或 Transformers 原生并行。

Accelerate 切分 batch 流之前，每个样本会在连续的 CP rank 上重复。同一 CP 组因此接收相同样本，而不同
组接收不同的数据并行样本。标签在切分序列前基于完整序列完成移位，所以每个 CP 边界上的因果预测目标均
准确无误。Trainer 会显式求和每个 shard 上移位后的 token loss，再按所有 DP 组和梯度累积 microbatch
中的全局不重复 token 数归一化，并补偿 DeepSpeed 在整个 world 上的梯度平均，避免 CP rank 稀释梯度。

有效的全局不重复 batch size 为
`per_device_train_batch_size * gradient_accumulation_steps * (world_size / context_parallel_size)`。Trainer 和
DeepSpeed 仍可能显示包含 CP 重复副本的物理 world-size batch；设置学习率或解读样本吞吐量时应使用不重复
batch size。

查询头数量必须能被 `context_parallel_size` 整除。KV 头数量可整除时会被切分；当 Gemma 4 全局注意力层
的 KV 头少于 CP rank 时，注意力包会先扩展 KV 头，再执行全交换。
