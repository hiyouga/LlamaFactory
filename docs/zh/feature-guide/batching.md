# 批处理策略

`batching_strategy` 决定每个 micro-batch 包含多少条样本，以及这些样本是通过 padding 组成矩形张量还是拼接为一条连续序列。

| 策略 | 样本数 | 序列组织方式 | 适用场景 |
|------|--------|--------------|----------|
| `normal` | 固定 | 按 batch 内最长序列 padding | 样本长度接近，或训练多模态模型 |
| `padding_free` | 固定 | 将多条样本拼接为一条连续序列 | 样本长度差异较大，希望减少 padding |
| `dynamic_batching` | 动态 | 按最长序列 padding | 希望每个 batch 使用接近固定数量的 padded token |
| `dynamic_padding_free` | 动态 | 按 token 预算选择样本并拼接 | 样本长度差异较大，希望同时动态调整样本数并移除 padding |

例如设置 `cutoff_len: 2048`、`micro_batch_size: 4` 时，动态策略的 token 预算为 `2048 × 4 = 8192`。假设依次读到的样本长度为 2048、512、512、512：

- `normal` 固定选择 4 条样本，并将每条样本 padding 到 2048，最终处理 8192 个 token 位置。
- `padding_free` 仍选择 4 条样本，但将它们拼接为长度 3584 的序列，从而移除 padding。
- `dynamic_batching` 在 `最长样本长度 × 样本数` 不超过 8192 的范围内决定样本数，然后按最长样本进行 padding。
- `dynamic_padding_free` 在样本总长度不超过 8192 的范围内决定样本数，并将所选样本拼接起来。

因此，动态策略中的 `micro_batch_size` 用于计算 token 预算，并不表示最终 batch 一定包含相同数量的样本。

## 选择批处理策略

```yaml
batching_strategy: dynamic_padding_free
micro_batch_size: 4
cutoff_len: 2048
max_steps: 100
flash_attn: flash_attention_2
```

仓库在 `examples/v1/train_batching_strategy/` 下为四种策略提供了完整示例。

## 动态批处理约束

- `dynamic_batching` 必须设置正数 `max_steps`。
- `dynamic_batching` 不支持 `save_epochs`，应使用 `save_steps`。
- `padding_free` 和 `dynamic_padding_free` 需要设置 `flash_attn: flash_attention_2`。
- `normal` 以外的策略仅支持纯文本数据；使用其他策略处理多模态数据时，BatchGenerator 会在生成 batch 时抛出 `NotImplementedError`。

内部 collate 和状态恢复流程见[BatchGenerator](../developer-guide/core/batch_generator.md)。
