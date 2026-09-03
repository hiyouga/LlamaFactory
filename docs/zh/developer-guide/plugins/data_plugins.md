# 数据插件

## DataLoaderPlugin

Loader 将一个 `DatasetInfo` 条目变成 Hugging Face Dataset。当前注册 `local`；Hub 数据通过 DataEngine 的来源分派处理。

```python
@DataLoaderPlugin("example").register()
def load_example(dataset_info):
    ...
```

## DataConverterPlugin

Converter 将原始样本批次转成 v1 `SFTSample` 或 `DPOSample`。当前注册：

- `alpaca`
- `sharegpt`
- `pair`

```python
@DataConverterPlugin("example").register()
def convert_example(examples):
    return {"messages": ...}
```

返回字段必须符合 `utils/types.py` 中的 Messages 类型。偏好 converter 必须产生 `chosen_messages` 和 `rejected_messages`。

## 调整数据索引

数据索引调整和样本选择是普通函数，不存在旧版文档描述的 `DataIndexPlugin` 或 `DataSelectorPlugin`。
