# LLaMA-Factory v1 数据预处理

## 总览

LLaMA-Factory `v1` 采用了全新的数据处理架构，主要包含以下核心组件：

- **DataEngine**：数据引擎，负责数据集的加载、索引和转换等各种插件的接入和调用，并提供数据访问接口
- **DataConverterPlugin**：数据转换器，将非标准格式转换为统一的标准格式
- **DataLoaderPlugin**：数据加载插件，支持多种文件格式的加载
- **DataIndexPlugin**：数据索引插件，支持数据集的采样和权重调整
- **DataSelectorPlugin**：数据选择插件，支持灵活的数据访问方式

与 LLaMA-Factory `v0` 版本相比，`v1` 版本采用了统一的数据格式（Messages Format），所有数据都会被转换为标准的对话消息列表；此外，`v1` 版本通过 DataEngine 与 Plugin 机制，提供了自定义数据处理流的接口，具有更好的可扩展性和一致性。

---

## 目录

- [基本用法](#基本用法)
- [标准数据格式](#标准数据格式)
- [数据集配置文件](#数据集配置文件)
- [完整示例](#完整示例)

---

## 基本用法

### 在训练配置文件，可以通过如下方式配置数据集：

<details open>
<summary>方式 1：使用 HF Hub Repo ID</summary>

直接指定 HF Hub 上的数据集 Repo ID，DataEngine 会自动从 HF Hub 下载并加载数据集。

注：使用 Repo ID 直接加载的数据集需要为标准格式

**训练配置文件示例：**

```yaml
# example_sft.yaml

...

dataset: llamafactory/v1-sft-demo  # HF Hub Repo ID

...
```

</details>

<details>
<summary>方式 2：使用 HF Hub 上的 YAML 配置文件</summary>

`dataset`字段指定 HF Hub 上的 `dataset_info.yaml` 的 URI，DataEngine 会自动下载该配置文件并根据其中的配置加载数据集。

**训练配置文件示例：**

```yaml
# example_sft.yaml

...

dataset: llamafactory/v1-sft-demo/dataset_info.yaml  # 远程 dataset_info.yaml 路径

...

```

</details>

<details>
<summary>方式 3：使用本地 HF 数据集文件路径</summary>

`dataset`字段指定本地的数据集文件路径（`.json`、`.jsonl` 等）

注：直接指定数据集文件路径，要求该数据文件的格式已为标准格式

**训练配置文件示例：**

```yaml
# example_sft.yaml

...

dataset: ~/data/v1_sft_demo.jsonl   # 本地数据集文件绝对路径

...
```

</details>

<details>
<summary>方式 4：使用本地 YAML 配置文件路径</summary>

`dataset`字段指定本地的 `dataset_info.yaml` 配置文件路径，DataEngine 会根据该配置加载其中的数据集。

**训练配置文件示例：**

```yaml
# example_sft.yaml

...

dataset: ~/data/dataset_info.yaml    # 本地 dataset_info.yaml 文件路径

...
```

</details>


---



## 标准数据格式

v1 使用统一的 **Messages 格式**作为标准数据格式。每个样本都是一个包含 `messages` 字段的 JSON 对象。

针对alpaca、sharegpt、以及dpo等格式的数据，可以通过内置的`DataConverterPlugin`插件，自动将其转化为标准格式，对于其他自定义格式的数据，用户也可通过自定义`DataConverterPlugin`来实现数据格式标准化，这部分内容参见[`DataConverterPlugin`](../dev-guide/plugins/data-plugins.md#dataconverterplugin)

### 1. SFT（监督微调）样本格式


```json
{
  "messages": [
    {
      "role": "system",
      "content": [{"type": "text", "value": "You are a helpful assistant."}],
      "loss_weight": 0.0
    },
    {
      "role": "user",
      "content": [{"type": "text", "value": "Hello, who are you?"}],
      "loss_weight": 0.0
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "value": "I am an AI assistant."}],
      "loss_weight": 1.0
    }
  ]
}
```

#### 字段说明：

- **messages**: 消息列表，包含一轮或多轮对话
  - **role**: 消息角色，可选值：
    - `"system"`: 系统提示
    - `"user"`: 用户输入
    - `"assistant"`: 模型回复
  - **content**: 内容列表，每个元素包含：
    - **type**: 内容类型，可选值：
      - `"text"`: 文本内容
      - `"image_url"`: 图像 URL（多模态）
      - `"audio_url"`: 音频 URL（多模态）
      - `"video_url"`: 视频 URL（多模态）
      - `"tools"`: 工具描述
      - `"tool_calls"`: 工具调用
      - `"reasoning"`: 推理过程
    - **value**: 具体内容（字符串）
  - **loss_weight**: 损失权重（浮点数）
    - `0.0`: 不计算损失（用于提示词部分）
    - `1.0`: 完全计算损失（用于回复部分）
    - 可设置为其他值以调整不同部分的学习权重

- **_dataset_name** (可选): 数据集名称，由 DataEngine 自动添加
- **extra_info** (可选): 额外信息字段

### 2. DPO（偏好对齐）样本格式

```json
{
  "chosen_messages": [
    {
      "role": "user",
      "content": [{"type": "text", "value": "用户提问"}],
      "loss_weight": 0.0
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "value": "更优的回答"}],
      "loss_weight": 1.0
    }
  ],
  "rejected_messages": [
    {
      "role": "user",
      "content": [{"type": "text", "value": "用户提问"}],
      "loss_weight": 0.0
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "value": "较差的回答"}],
      "loss_weight": 1.0
    }
  ]
}
```

### 3. 多模态支持

对于多模态数据，可以在 `content` 列表中添加非文本类型的内容：

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "value": "这张图片里有什么？"},
        {"type": "image_url", "value": "path/to/image.jpg"}
      ],
      "loss_weight": 0.0
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "value": "图片中有一只猫。"}],
      "loss_weight": 1.0
    }
  ]
}
```

**说明**：`image_url`、`audio_url`、`video_url` 的路径可以是相对路径或绝对路径，具体加载方式由 `DataLoaderPlugin` 决定。

---

## 数据集配置文件

### 1. dataset_info.yaml 配置文件格式

`dataset_info.yaml` 支持同时配置多个数据集，支持分别从 HF Hub 和本地获取数据集，数据集默认会混合并打乱顺序。

**示例配置文件：`data/dataset_info.yaml`**

```yaml
# 数据集 1：使用本地文件 + Alpaca 转换器
identity:
  file_name: ~/data/identity.json            #本地数据集文件绝对路径
  converter: alpaca                           # 使用 alpaca 转换器

# 数据集 2：指定自定义数据集目录
alpaca_en_demo:
  file_name:  ~/data/alpaca_en_demo.json     # 数据集文件名
  converter: alpaca                           # 转换器插件
  size: 500                                   # 只使用 500 个样本
  weight: 0.5                                 # 数据集权重，用于控制该数据集的采样频率
  split: train                                # 数据集划分，默认为 train
  streaming: false                            # 是否流式加载，默认为 false

# 数据集 3：从 Hugging Face Hub 加载
hf_dataset:
  hf_hub_url: llamafactory/v1-sft-demo  # HF repo ID
  streaming: false  

# 数据集 4：已经是标准格式，无需转换器
standard:
  file_name: ~/data/v1_sft_demo.jsonl   # 本地标准数据集文件路径

# 数据集 5：自定义数据集和 converter 插件
custom_dataset:
  file_name: custom_data.json
  converter: custom_converter
  weight: 1.0  
```

### 2. 配置字段说明

#### 数据源配置（二者必选其一）：

- **hf_hub_url** (str): Hugging Face Hub 数据集仓库 ID
  - 示例：`"llamafactory/v1-sft-demo"`
  - 如果指定，则从 HF Hub 加载数据集

- **file_name** (str): 本地文件路径
  - 支持格式：`.json`、`.jsonl`、`.csv`、`.parquet`、`.arrow`、`.txt`

#### 可选配置：

- **split** (str): 数据集划分，默认为 `"train"`
- **converter** (str): 数据转换器名称
  - 可选值：`"alpaca"`（更多转换器持续添加中，也可在 data_plugin 中添加自定义 converter）
  - 如果不指定，则假定数据已是标准格式
- **size** (int): 使用的样本数量，默认使用全部
- **weight** (float): 数据集权重，用于混合数据集时的采样频率，默认为 1.0
- **streaming** (bool): 是否流式加载，默认为 `False`

---


## 完整示例

### 1. 基础使用示例

```python
from llamafactory.v1.config.data_args import DataArguments
from llamafactory.v1.core.data_engine import DataEngine

# 使用本地 YAML 配置
data_args = DataArguments(
    dataset="~/data/v1_sft_demo.jsonl",
    cutoff_len=2048
)

# 初始化 DataEngine
engine = DataEngine(data_args=data_args)

# 查看数据集信息
print(f"数据集总样本数: {len(engine)}")
print(f"数据集列表: {list(engine.datasets.keys())}")

# 访问数据样本
sample = engine[0]
print(f"样本格式: {sample.keys()}")
print(f"消息列表: {sample['messages']}")

# 批量访问
batch = engine[0:10]
print(f"批量样本数: {len(batch)}")
```

### 2. 输出示例

**查看数据集信息输出：**

```
数据集总样本数: 500
数据集列表: ['default']
样本格式: dict_keys(['_dataset_name', 'messages'])
消息列表: [{'role': 'user', 'content': [{'type': 'text', 'value': 'hi'}], 'loss_weight': 0.0}, {'role': 'assistant', 'content': [{'type': 'text', 'value': 'Hello! I am {{name}}, an AI assistant developed by {{author}}. How can I assist you today?'}], 'loss_weight': 1.0}]
批量样本数: 10
```

**访问单个样本输出：**

```python
{
  '_dataset_name': 'alpaca_en_demo',
  'messages': [
    {
      'role': 'user',
      'content': [{'type': 'text', 'value': 'What is the capital of France?'}],
      'loss_weight': 0.0
    },
    {
      'role': 'assistant',
      'content': [{'type': 'text', 'value': 'The capital of France is Paris.'}],
      'loss_weight': 1.0
    }
  ]
}
```

### 3. 混合多数据集配置文件示例

**配置文件：`data/mixed_datasets.yaml`**

```yaml
dataset_1:
  file_name: alpaca_en_demo.json
  converter: alpaca
  weight: 1.0

dataset_2:
  file_name: identity.json
  converter: alpaca
  weight: 2.0

dataset_3:
  hf_hub_url: llamafactory/v1-sft-demo
  weight: 1.5
```


### 4. 多模态数据示例

**数据文件：`data/multimodal_demo.jsonl`**

标准化后数据示例：

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "value": "Who are they?"},
          {"type": "image_url", "value": "mllm_demo_data/1.jpg"}
        ],
        "loss_weight": 0.0
      },
      {
        "role": "assistant",
        "content": [
          {"type": "text", "value": "They're Kane and Gretzka from Bayern Munich."}
        ],
        "loss_weight": 1.0
      },
      {
        "role": "user",
        "content": [
          {"type": "text", "value": "What are they doing?"},
          {"type": "image_url", "value": "mllm_demo_data/1.jpg"}
        ],
        "loss_weight": 0.0
      },
      {
        "role": "assistant",
        "content": [
          {"type": "text", "value": "They are celebrating on the soccer field."}
        ],
        "loss_weight": 1.0
      }
    ]
  },
  {
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "value": "Who is he?"},
          {"type": "image_url", "value": "mllm_demo_data/2.jpg"}
        ],
        "loss_weight": 0.0
      },
      {
        "role": "assistant",
        "content": [
          {"type": "text", "value": "He's Thomas Muller from Bayern Munich."}
        ],
        "loss_weight": 1.0
      },
      {
        "role": "user",
        "content": [
          {"type": "text", "value": "Why is he on the ground?"}
        ],
        "loss_weight": 0.0
      },
      {
        "role": "assistant",
        "content": [
          {"type": "text", "value": "Because he's sliding on his knees to celebrate."}
        ],
        "loss_weight": 1.0
      }
    ]
  }
]
```

```python
from llamafactory.v1.config.data_args import DataArguments
from llamafactory.v1.core.data_engine import DataEngine

data_args = DataArguments(dataset="data/multimodal_demo.jsonl")
engine = DataEngine(data_args=data_args)

# 访问多模态样本
sample = engine[0]
print("用户消息内容：")
for content_item in sample['messages'][0]['content']:
    print(f"  类型: {content_item['type']}, 值: {content_item['value']}")
```

---

**注意事项**：

1. 所有数据最终都会转换为标准的 Messages 格式
2. 通过 `converter` 插件可以支持多种数据格式
3. 通过 `weight` 和 `size` 参数可以灵活控制数据分布
4. 支持同时使用本地数据集和 HuggingFace Hub 数据集
5. 多模态数据通过在 `content` 中添加不同类型的元素来支持
6. 更多细节信息请参考我们的 [API REFERENCE](../dev-guide/core/data-engine.md#dataengine)
