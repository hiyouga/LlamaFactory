# 数据插件

数据插件把数据来源和原始字段格式从 DataEngine 的索引逻辑中分离。Loader 负责获得可读取的数据集，Converter 负责解释其中的一条记录；两者由不同配置字段选择，可以组合使用。

## 与 DataEngine 的交接

```text
DatasetInfo.source
  → hf_hub：DataEngine 直接调用 datasets.load_dataset
  → 其他来源：DataLoaderPlugin(source).load(dataset_info)
      → Dataset
Dataset 中的一条原始记录 + DatasetInfo.converter
  → DataConverterPlugin(converter)(raw_sample)
      → 标准 SFTSample / DPOSample
```

相同的本地 loader 可以配合 `alpaca` 或 `sharegpt` converter；数据已经是标准消息结构时省略 converter。分词与标签生成发生在后续 Renderer 中。

## DataLoaderPlugin

接口定义在 `plugins/data_plugins/loader.py`。DataEngine 调用 `DataLoaderPlugin(source).load(dataset_info)`，而 `load` 会从 DatasetInfo 中提取 `path`、`split` 和 `streaming`，再将这三个位置参数传给注册函数。

因此，注册函数接收的是以下参数，不是整个 DatasetInfo：

```python
@DataLoaderPlugin("example").register()
def load_example(path, split, streaming):
    ...
```

当前注册的 `local` 根据文件扩展名选择 Hugging Face dataset builder，再加载文件或目录。Hub 数据由 DataEngine 直接分派，当前没有通过一个名为 `hf_hub` 的 loader 注册项加载。

返回的数据集由 DataEngine 持有并用于建立索引。新增 loader 必须保持调用参数与返回数据集的约定；当前训练链路要求可按索引读取的数据集，具体 streaming 边界见 [DataEngine](../core/data_engine.md#处理-streaming-dataset)。

## DataConverterPlugin

接口定义在 `plugins/data_plugins/converter.py`。Converter 接收单条原始样本字典，返回一个 v1 `SFTSample` 或 `DPOSample`，不是输入或返回整个 batch。当前注册：

- `alpaca`
- `sharegpt`
- `pair`

```python
@DataConverterPlugin("example").register()
def convert_example(raw_sample):
    return {"messages": ...}
```

返回字段必须符合 `utils/types.py` 中的 Messages 类型。SFT 返回 `messages`，偏好数据返回 `chosen_messages` 和 `rejected_messages`；图片等媒体也在这一阶段转为标准内容块，实际媒体特征由 Renderer 的 processor 生成。

DataEngine 在建立多轮索引和实际取样时都会调用 converter。它应稳定地转换一条记录，避免在函数中随机选择对话轮次或维护读取进度。轮次展开、采样规模和读取顺序分别由 DataEngine 与 sampler 管理。

## 调整数据索引

`adjust_data_index` 根据 `size`、`weight` 调整某个数据集的索引，`select_data_sample` 处理索引选择。两者是 `loader.py` 中的普通函数，不通过插件名称注册或路由。改变来源、格式时扩展 loader/converter；改变共用索引语义时，应检查这些函数及 DataEngine 的调用位置。

注册在模块导入时生效，机制见[插件注册机制](../baseplugin_mechanism.md)。使用配置见[数据准备](../../feature-guide/data_preparation.md)。
