# 奖励模型训练（RM）

`rm` 入口训练回答评分模型，使 chosen 回答的评分高于 rejected 回答。

RM 与 [DPO](dpo.md) 使用相同的偏好对数据结构，输出模型用于回答评分，不作为普通聊天模型使用。

## 训练配置

`peft_config.name: lora` 启用 LoRA，未配置 `peft_config` 时进行全参训练。分布式后端由 `dist_config` 配置，支持条件见[分布式训练](distributed_training.md)。以下为完整的 LoRA RM 配置（`config.yaml`）：

```yaml
model: Qwen/Qwen3-0.6B
train_dataset: data/v1_dpo_demo.yaml

peft_config:
  name: lora
  r: 16
  target_modules: all

dist_config:
  name: fsdp2

output_dir: outputs/qwen3_rm
micro_batch_size: 1
cutoff_len: 2048
learning_rate: 1.0e-5
max_steps: 10
```

```bash
llamafactory-cli rm config.yaml
```

入口会将 `model_class` 设置为 `cls`，初始化 score head，并在训练开始前检查首个样本是否包含 `chosen_messages` 和 `rejected_messages`。

## 训练约束

RM 当前要求 `cp_size` 为 `1`。`cutoff_len` 需要保留 chosen 和 rejected 的有效 token；否则当前 micro-batch 无法组成偏好对。
