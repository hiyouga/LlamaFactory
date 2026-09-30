# 数据参数

## DataArguments

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `train_dataset` | `str \| None` | `None` | 训练数据集路径、YAML 或 Hub ID |
| `eval_dataset` | `str \| None` | `None` | 字段已定义；评估流程尚未实现 |

## DatasetInfo

数据集 YAML 的每个顶层条目使用以下字段：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `path` | `str` | 必填 | 本地路径或 Hub ID |
| `source` | `local \| hf_hub` | `hf_hub` | 数据来源 |
| `split` | `str` | `train` | 数据集 split |
| `converter` | `str \| None` | `None` | `alpaca`、`sharegpt`、`pair` 或已注册名称 |
| `size` | `int \| None` | 全部 | 重采样后的目标索引数量 |
| `weight` | `float` | 不设置 | 索引数量的缩放倍率 |
| `streaming` | `bool` | `false` | 字段已定义；当前训练路径不支持 streaming 数据集 |

### 采样规模的计算

`size` 和 `weight` 作用于数据集展开后的训练索引。多轮 SFT 中，每个受监督的 assistant turn 对应一个索引条目，因此索引数量可能大于原始数据的行数；偏好对数据则以一组 chosen/rejected 为一个条目。

使用正数 `size`、`weight` 时，按以下顺序计算：

1. 设置 `size` 后，先从展开后的索引中有放回抽取 `size` 个条目；省略时保留完整索引。
2. 设置 `weight` 后，再从上一步结果中有放回抽取 `int(当前条目数 × weight)` 个条目，小数部分向下取整；省略时保留上一步结果。

例如，`size: 1000` 与 `weight: 0.5` 同时设置时，先抽取 1000 个条目，再抽取其中的 500 个条目。最终规模为 500，但可能包含重复条目，也可能没有覆盖某些原始记录。

`weight` 控制每个数据集贡献的索引数量，不是各数据集之间归一化后的抽样概率，也不改变样本的损失权重。`weight > 1` 可以增加索引数量。显式设置 `weight: 1.0` 仍会执行有放回重采样；省略 `size` 和 `weight` 才会直接保留全部展开后的索引。

数据格式和多数据集组合方式见[数据准备](../feature-guide/data_preparation.md)。
