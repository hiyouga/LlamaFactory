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
| `size` | `int \| None` | 全部 | 样本数 |
| `weight` | `float` | `1.0` | 采样权重 |
| `streaming` | `bool` | `false` | 字段已定义；当前训练路径不支持 streaming 数据集 |

数据格式和多数据集组合方式见[数据准备](../feature-guide/data_preparation.md)。
