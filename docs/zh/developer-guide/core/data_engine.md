# DataEngine

`DataEngine(dataset_path: str)` 位于 `core/data_engine.py`，把一个或多个原始数据集暴露为可按整数索引访问的统一 Dataset。它处理数据来源、字段转换和样本索引，返回的仍是消息结构；分词和监督标签由后续 [Renderer](renderer.md)生成。

## 加载数据集

```text
dataset_path
  → 识别 YAML、目录、数据文件或 Hub ID
  → 生成 dataset name → DatasetInfo 映射
  → hf_hub 直接加载，其他 source 调用 DataLoaderPlugin.load
  → 逐条转换样本，确定 SFT 的受监督轮次
  → 建立 (dataset_name, sample_index, cut) 全局索引
  → 根据 size / weight 调整索引
```

初始化时，`datasets` 保存各来源加载出的 Dataset，`dataset_infos` 保存来源配置，`data_index` 保存训练样本到原始行的映射。索引项中的 `cut` 表示取消息列表的哪个前缀；它不是 token 截断长度。

多轮 SFT 的每个受监督 assistant turn 对应一个索引项，多个索引项可以指向同一条原始对话。DPO 等非 SFT 样本保持完整，`cut` 为 `None`。这一设计让 sampler 能按训练样本分配索引，同时保留原始数据集。

## 读取一个样本

`__getitem__` 从全局索引取出数据集名称、原始行号和 `cut`，再由 `_get` 读取原始行、调用 converter，并按 `cut` 截取消息前缀。返回样本带有 `_dataset_name`，供后续处理识别来源。

Converter 既会在初始化建立索引时调用，也会在实际取样时调用；DataEngine 不把索引阶段的转换结果缓存为新的数据集。因此转换函数应对相同输入产生一致的消息结构，避免依赖调用次数或在转换中改变原始行，否则索引记录的轮次可能与训练读取的内容不一致。

## 解析 DatasetInfo

字段结构定义在 `utils/types.py`。`_get_dataset_info` 将路径、Hub ID 或 YAML 统一成名称到 DatasetInfo 的映射，后续加载逻辑只处理这个映射。字段与默认值见[数据参数](../../configuration/data.md#datasetinfo)。

## 处理 Streaming Dataset

所有数据集必须同时为 map-style 或同时为 streaming。混合模式无法共享同一种索引和 sampler 语义，因此 DataEngine 初始化阶段会拒绝。当前 BatchGenerator 只支持 map-style dataset；streaming dataset 进入批处理初始化时会抛出 `NotImplementedError`。

## 扩展数据源与格式

加载新来源时，扩展 `DataLoaderPlugin`，保持返回 Dataset 的约定；只改变原始字段的含义时，扩展 `DataConverterPlugin`，保持返回标准 Sample 的约定。两种接口及其调用参数见[数据插件](../plugins/data_plugins.md)。

多轮展开和全局索引属于 DataEngine，共用索引行为应在此处理。模型模板、tokenization 和 padding 分别属于 Renderer 与 BatchGenerator，不放入 loader 或 converter。用户用法见[数据准备](../../feature-guide/data_preparation.md)。
