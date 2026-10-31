# 偏好优化（DPO）

v1 通过统一的 `dpo` 入口运行偏好优化，`pref_loss` 支持 `sigmoid`、`orpo` 和 `simpo`。数据必须是 chosen/rejected 偏好对。

## 运行 DPO

```bash
llamafactory-cli dpo examples/v1/train_lora/train_lora_dpo.yaml
```

## 训练配置

`peft_config.name: lora` 启用 LoRA，未配置 `peft_config` 时进行全参训练。以下是完整的 LoRA DPO 配置，保存为 `config.yaml` 后运行 `llamafactory-cli dpo config.yaml`：

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

## 偏好损失

`pref_loss` 指定偏好损失：

- `sigmoid`：标准 DPO，相对参考策略进行偏好优化，`pref_beta` 控制偏好项缩放。
- `orpo`：无参考策略的 odds-ratio 偏好目标，基于回答的平均 log-prob 计算。
- `simpo`：无参考策略的平均 log-prob 差值目标，通过 `simpo_gamma` 设置 margin。

`pref_ftx` 加入 SFT 损失，`dpo_label_smoothing` 用于 cDPO。设置 `ld_alpha` 后，LD-DPO 会将 chosen 和 rejected 中超出较短响应长度的尾部 token log-prob 乘以该系数。参数定义见[训练参数](../configuration/training.md#trainingarguments)。

## 参考模型

标准 DPO 需要 reference log-prob：全参训练会建立独立的 reference model；LoRA 训练复用 policy model 的基座权重，并在计算 reference log-prob 时禁用 adapter。ORPO 和 SimPO 的目标计算不使用 reference log-prob。
