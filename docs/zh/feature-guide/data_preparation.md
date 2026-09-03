# 数据准备

v1 将训练样本统一为 Messages 结构。`DataEngine` 根据 `train_dataset` 指向的路径加载数据，并在存在 `converter` 时转换原始字段。

## 配置训练数据集

`train_dataset` 接受以下形式：

- 本地数据集 YAML，例如 `data/v1_sft_demo.yaml`
- 本地数据文件或目录
- Hugging Face Hub 数据集 ID
- Hub 数据集仓库中的 YAML

`eval_dataset` 字段已定义，评估流程尚未实现。完整字段见[数据参数](../configuration/data.md#dataarguments)。

## SFT 数据格式

```json
{
  "messages": [
    {
      "role": "user",
      "content": [{"type": "text", "value": "介绍一下你自己。"}],
      "loss_weight": 0.0
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "value": "我是一个 AI 助手。"}],
      "loss_weight": 1.0
    }
  ]
}
```

`content` 是内容块列表；文本使用 `text`，多模态内容可以使用 `image_url`、`audio_url` 或 `video_url`。`loss_weight` 是该 assistant turn 的监督权重，并应用到该回复的每个监督 token。`0.0` 不参与损失计算，`1.0` 使用完整权重，也可以设置 `0.5` 等中间值调整不同回复的相对权重。

多轮对话会按每个受监督的 assistant turn 展开为多条训练样本，每条样本只监督最后一个 assistant turn。

多模态 SFT 示例位于 `data/v1_multimodal_demo.yaml`，对应训练配置为 `examples/v1/train_full/train_multimodal.yaml`。

## DPO/RM 数据格式

DPO 和 RM 使用 `chosen_messages` 与 `rejected_messages`：

```json
{
  "chosen_messages": [
    {"role": "user", "content": [{"type": "text", "value": "问题"}], "loss_weight": 0.0},
    {"role": "assistant", "content": [{"type": "text", "value": "更优回答"}], "loss_weight": 1.0}
  ],
  "rejected_messages": [
    {"role": "user", "content": [{"type": "text", "value": "问题"}], "loss_weight": 0.0},
    {"role": "assistant", "content": [{"type": "text", "value": "较差回答"}], "loss_weight": 1.0}
  ]
}
```

## 组合多个数据集

```yaml
identity:
  path: data/identity.json
  source: local
  converter: alpaca

demo:
  path: organization/dataset
  source: hf_hub
  split: train
  size: 1000
  weight: 0.5
  streaming: false
```

同一个 YAML 中的 streaming 配置必须一致；当前训练路径不支持 streaming 数据集。多个条目会组成一个全局数据索引；`size` 与 `weight` 用于控制每个数据集的采样规模。

## 转换现有数据格式

| 名称 | 原始数据 |
|------|----------|
| `alpaca` | `instruction`、`input`、`output` |
| `sharegpt` | `conversations` |
| `pair` | chosen/rejected 偏好对 |

扩展 converter 的接口见[数据插件](../developer-guide/plugins/data_plugins.md)。
