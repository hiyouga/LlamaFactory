# DataEngine

`DataEngine(dataset_path: str)` 同时实现 PyTorch Dataset 接口和数据集配置分派。

## 加载数据集

```text
dataset_path
  → 识别 YAML、目录、数据文件或 Hub ID
  → 生成 dataset name → DatasetInfo 映射
  → DataLoaderPlugin 加载
  → 可选 DataConverterPlugin 转换
  → 计算每个数据集的长度、offset 和全局索引
```

`DataEngine` 在样本中注入 `_dataset_name`，并根据全局 index 定位具体数据集和局部 index。多轮 SFT 对话按受监督的 assistant turn 展开，每个 turn 成为一条独立索引条目；非 SFT 样本保持完整。

## 解析 DatasetInfo

字段结构定义在 `utils/types.py`。`path` 是唯一必填字段，`source` 默认 `hf_hub`，`split` 默认 `train`，`streaming` 默认 `false`。

## 处理 Streaming Dataset

所有数据集必须同时为 map-style 或同时为 streaming。混合模式无法共享同一种索引和 sampler 语义，因此 DataEngine 初始化阶段会拒绝。当前 BatchGenerator 只支持 map-style dataset；streaming dataset 进入批处理初始化时会抛出 `NotImplementedError`。

## 扩展数据源与格式

新增 source 时注册 `DataLoaderPlugin`；新增原始字段格式时注册 `DataConverterPlugin`。用户用法见[数据准备](../../feature-guide/data_preparation.md)。
