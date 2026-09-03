# 偏好优化（DPO）

v1 通过统一的 `dpo` 入口运行偏好优化。`pref_loss` 可以选择标准 DPO 使用的 `sigmoid`，也可以选择 `orpo` 或 `simpo`。数据必须是 chosen/rejected 偏好对。

## 运行 DPO

```bash
llamafactory-cli dpo examples/v1/train_lora/train_lora_dpo.yaml
```

## 训练配置

```yaml
model: Qwen/Qwen3-4B
model_class: llm
train_dataset: data/v1_dpo_demo.yaml

peft_config:
  name: lora
  r: 16
  lora_alpha: 32
  target_modules: all

pref_loss: sigmoid
pref_beta: 0.1
pref_ftx: 0.0
dpo_label_smoothing: 0.0

dist_config:
  name: fsdp2

output_dir: outputs/qwen3_dpo
micro_batch_size: 1
cutoff_len: 2048
learning_rate: 1.0e-5
max_steps: 10
```

## 选择偏好损失

| `pref_loss` | 说明 |
|-------------|------|
| `sigmoid` | 标准 DPO；使用 `pref_beta` |
| `orpo` | ORPO 目标 |
| `simpo` | SimPO；额外使用 `simpo_gamma` |

`pref_ftx` 加入 SFT 损失，`dpo_label_smoothing` 用于 cDPO。设置 `ld_alpha` 后，LD-DPO 会将 chosen 和 rejected 中超出较短响应长度的尾部 token log-prob 乘以该系数。参数定义见[训练参数](../configuration/training.md#trainingarguments)。

## 参考模型

全参训练会建立独立的 reference model。LoRA 训练复用 policy model 的基座权重，并在计算 reference log-prob 时禁用 adapter。
