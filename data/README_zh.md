[dataset_info.json](dataset_info.json) 包含了所有可用的数据集。如果您希望使用自定义数据集，请**务必**在 `dataset_info.json` 文件中添加*数据集描述*，并通过修改 `dataset: 数据集名称` 配置来使用数据集。

其中 `dataset_info.json` 文件应放置在 `dataset_dir` 目录下。您可以通过修改 `dataset_dir` 参数来使用其他目录。默认值为 `./data`。

目前我们支持 **alpaca** 格式和 **sharegpt** 格式的数据集。允许的文件类型包括 json、jsonl、csv、parquet 和 arrow。

```json
"数据集名称": {
  "hf_hub_url": "Hugging Face 的数据集仓库地址（若指定，则忽略 script_url 和 file_name）",
  "ms_hub_url": "ModelScope 的数据集仓库地址（若指定，则忽略 script_url 和 file_name）",
  "script_url": "包含数据加载脚本的本地文件夹名称（若指定，则忽略 file_name）",
  "file_name": "该目录下数据集文件夹或文件的名称（若上述参数未指定，则此项必需）",
  "formatting": "数据集格式（可选，默认：alpaca，可以为 alpaca 或 sharegpt）",
  "ranking": "是否为偏好数据集（可选，默认：False）",
  "subset": "数据集子集的名称（可选，默认：None）",
  "revision": "Hugging Face 数据集仓库的分支、标签或提交 SHA（可选，默认：main）",
  "split": "所使用的数据集切分（可选，默认：train）",
  "folder": "Hugging Face 仓库的文件夹名称（可选，默认：None）",
  "num_samples": "该数据集所使用的样本数量。（可选，默认：None）",
  "columns（可选）": {
    "prompt": "数据集代表提示词的表头名称（默认：instruction）",
    "query": "数据集代表请求的表头名称（默认：input）",
    "response": "数据集代表回答的表头名称（默认：output）",
    "history": "数据集代表历史对话的表头名称（默认：None）",
    "messages": "数据集代表消息列表的表头名称（默认：conversations）",
    "system": "数据集代表系统提示的表头名称（默认：None）",
    "tools": "数据集代表工具描述的表头名称（默认：None）",
    "images": "数据集代表图像输入的表头名称（默认：None）",
    "videos": "数据集代表视频输入的表头名称（默认：None）",
    "audios": "数据集代表音频输入的表头名称（默认：None）",
    "chosen": "数据集代表更优回答的表头名称（默认：None）",
    "rejected": "数据集代表更差回答的表头名称（默认：None）",
    "kto_tag": "数据集代表 KTO 标签的表头名称（默认：None）"
  },
  "tags（可选，用于 sharegpt 格式）": {
    "role_tag": "消息中代表发送者身份的键名（默认：from）",
    "content_tag": "消息中代表文本内容的键名（默认：value）",
    "user_tag": "消息中代表用户的 role_tag（默认：human）",
    "assistant_tag": "消息中代表助手的 role_tag（默认：gpt）",
    "observation_tag": "消息中代表工具返回结果的 role_tag（默认：observation）",
    "function_tag": "消息中代表工具调用的 role_tag（默认：function_call）",
    "system_tag": "消息中代表系统提示的 role_tag（默认：system，会覆盖 system column）"
  }
}
```

## Alpaca 格式

### 指令监督微调数据集

- [样例数据集](alpaca_zh_demo.json)

在指令监督微调时，`instruction` 列对应的内容会与 `input` 列对应的内容拼接后作为提示词，即提示词为 `instruction\ninput`。而 `output` 列对应的内容为模型回答。

对于推理类模型的微调，如果数据集包含思维链，则需要把思维链放在模型回答中，例如 `<think>cot</think>output`。

如果指定，`system` 列对应的内容将被作为系统提示词。

`history` 列是由多个字符串二元组构成的列表，分别代表历史消息中每轮对话的指令和回答。注意在指令监督微调时，历史消息中的回答内容**也会被用于模型学习**。

```json
[
  {
    "instruction": "用户指令（必填）",
    "input": "用户输入（选填）",
    "output": "模型回答（必填）",
    "system": "系统提示词（选填）",
    "history": [
      ["第一轮指令（选填）", "第一轮回答（选填）"],
      ["第二轮指令（选填）", "第二轮回答（选填）"]
    ]
  }
]
```

对于上述格式的数据，`dataset_info.json` 中的*数据集描述*应为：

```json
"数据集名称": {
  "file_name": "data.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "system": "system",
    "history": "history"
  }
}
```

> [!TIP]
> 如果模型本身具备推理能力（如 Qwen3）而数据集不包含思维链，LLaMA-Factory 会自动为数据添加空思维链。当 `enable_thinking` 为 `True` 时（慢思考，默认），空思维链会添加到模型回答中并且计算损失，否则会添加到用户指令中并且不计算损失（快思考）。请在训练和推理时保持 `enable_thinking` 参数一致。
>
> 如果您希望训练包含思维链的数据时使用慢思考，训练不包含思维链的数据时使用快思考，可以设置 `enable_thinking` 为 `None`。但该功能较为复杂，请谨慎使用。

### 预训练数据集

- [样例数据集](c4_demo.jsonl)

在预训练时，只有 `text` 列中的内容会用于模型学习。

```json
[
  {"text": "document"},
  {"text": "document"}
]
```

对于上述格式的数据，`dataset_info.json` 中的*数据集描述*应为：

```json
"数据集名称": {
  "file_name": "data.json",
  "columns": {
    "prompt": "text"
  }
}
```

### 偏好数据集

偏好数据集用于奖励模型训练、DPO 训练、ORPO 训练和 SimPO 训练。

它需要在 `chosen` 列中提供更优的回答，并在 `rejected` 列中提供更差的回答。

```json
[
  {
    "instruction": "用户指令（必填）",
    "input": "用户输入（选填）",
    "chosen": "优质回答（必填）",
    "rejected": "劣质回答（必填）"
  }
]
```

对于上述格式的数据，`dataset_info.json` 中的*数据集描述*应为：

```json
"数据集名称": {
  "file_name": "data.json",
  "ranking": true,
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "chosen": "chosen",
    "rejected": "rejected"
  }
}
```

### KTO 数据集

KTO 数据集需要提供额外的 `kto_tag` 列。详情请参阅 [sharegpt](#sharegpt-格式)。

### 多模态图像数据集

多模态图像数据集需要提供额外的 `images` 列。详情请参阅 [sharegpt](#sharegpt-格式)。

### 多模态视频数据集

多模态视频数据集需要提供额外的 `videos` 列。详情请参阅 [sharegpt](#sharegpt-格式)。

### 多模态音频数据集

多模态音频数据集需要提供额外的 `audios` 列。详情请参阅 [sharegpt](#sharegpt-格式)。

## Sharegpt 格式

### 指令监督微调数据集

- [样例数据集](glaive_toolcall_zh_demo.json)

相比 alpaca 格式的数据集，sharegpt 格式支持**更多的角色种类**，例如 human、gpt、observation、function 等等。它们构成一个对象列表呈现在 `conversations` 列中。

注意其中 human 和 observation 必须出现在奇数位置，gpt 和 function 必须出现在偶数位置。默认所有的 gpt 和 function 会被用于学习。

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "用户指令"
      },
      {
        "from": "function_call",
        "value": "工具参数"
      },
      {
        "from": "observation",
        "value": "工具结果"
      },
      {
        "from": "gpt",
        "value": "模型回答"
      }
    ],
    "system": "系统提示词（选填）",
    "tools": "工具描述（选填）"
  }
]
```

对于上述格式的数据，`dataset_info.json` 中的*数据集描述*应为：

```json
"数据集名称": {
  "file_name": "data.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "system": "system",
    "tools": "tools"
  }
}
```

### 预训练数据集

尚不支持，请使用 [alpaca](#alpaca-格式) 格式。

### 偏好数据集

- [样例数据集](dpo_zh_demo.json)

Sharegpt 格式的偏好数据集同样需要在 `chosen` 列中提供更优的消息，并在 `rejected` 列中提供更差的消息。

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "用户指令"
      },
      {
        "from": "gpt",
        "value": "模型回答"
      },
      {
        "from": "human",
        "value": "用户指令"
      }
    ],
    "chosen": {
      "from": "gpt",
      "value": "优质回答"
    },
    "rejected": {
      "from": "gpt",
      "value": "劣质回答"
    }
  }
]
```

对于上述格式的数据，`dataset_info.json` 中的*数据集描述*应为：

```json
"数据集名称": {
  "file_name": "data.json",
  "formatting": "sharegpt",
  "ranking": true,
  "columns": {
    "messages": "conversations",
    "chosen": "chosen",
    "rejected": "rejected"
  }
}
```

### KTO 数据集

- [样例数据集](kto_en_demo.json)

KTO 数据集需要额外添加一个 `kto_tag` 列，包含 bool 类型的人类反馈。

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "用户指令"
      },
      {
        "from": "gpt",
        "value": "模型回答"
      }
    ],
    "kto_tag": "人类反馈 [true/false]（必填）"
  }
]
```

对于上述格式的数据，`dataset_info.json` 中的*数据集描述*应为：

```json
"数据集名称": {
  "file_name": "data.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "kto_tag": "kto_tag"
  }
}
```

### 多模态图像数据集

- [样例数据集](mllm_demo.json)

多模态图像数据集需要额外添加一个 `images` 列，包含输入图像的路径。

注意图片的数量必须与文本中所有 `<image>` 标记的数量严格一致。

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "<image><image>用户指令"
      },
      {
        "from": "gpt",
        "value": "模型回答"
      }
    ],
    "images": [
      "图像路径（必填）",
      "图像路径（必填）"
    ]
  }
]
```

对于上述格式的数据，`dataset_info.json` 中的*数据集描述*应为：

```json
"数据集名称": {
  "file_name": "data.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "images": "images"
  }
}
```

### 多模态视频数据集

- [样例数据集](mllm_video_demo.json)

多模态视频数据集需要额外添加一个 `videos` 列，包含输入视频的路径。

注意视频的数量必须与文本中所有 `<video>` 标记的数量严格一致。

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "<video><video>用户指令"
      },
      {
        "from": "gpt",
        "value": "模型回答"
      }
    ],
    "videos": [
      "视频路径（必填）",
      "视频路径（必填）"
    ]
  }
]
```

对于上述格式的数据，`dataset_info.json` 中的*数据集描述*应为：

```json
"数据集名称": {
  "file_name": "data.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "videos": "videos"
  }
}
```

### 多模态音频数据集

- [样例数据集](mllm_audio_demo.json)

多模态音频数据集需要额外添加一个 `audios` 列，包含输入音频的路径。

注意音频的数量必须与文本中所有 `<audio>` 标记的数量严格一致。

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "<audio><audio>用户指令"
      },
      {
        "from": "gpt",
        "value": "模型回答"
      }
    ],
    "audios": [
      "音频路径（必填）",
      "音频路径（必填）"
    ]
  }
]
```

对于上述格式的数据，`dataset_info.json` 中的*数据集描述*应为：

```json
"数据集名称": {
  "file_name": "data.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "audios": "audios"
  }
}
```


### OpenAI 格式

OpenAI 格式仅仅是 sharegpt 格式的一种特殊情况，其中第一条消息可能是系统提示词。

```json
[
  {
    "messages": [
      {
        "role": "system",
        "content": "系统提示词（选填）"
      },
      {
        "role": "user",
        "content": "用户指令"
      },
      {
        "role": "assistant",
        "content": "模型回答"
      }
    ]
  }
]
```

对于上述格式的数据，`dataset_info.json` 中的*数据集描述*应为：

```json
"数据集名称": {
  "file_name": "data.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "messages"
  },
  "tags": {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant",
    "system_tag": "system"
  }
}
```
