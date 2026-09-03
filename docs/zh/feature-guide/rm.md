# 奖励模型训练（RM）

`rm` 入口使用序列分类模型对 chosen/rejected 回答进行成对排序训练。

## 训练配置

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
