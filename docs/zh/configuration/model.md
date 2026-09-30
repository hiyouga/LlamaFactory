# 模型参数

## ModelArguments

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | `str` | `Qwen/Qwen3-4B-Instruct-2507` | 本地模型路径或 Hub ID |
| `custom_chat_template` | `str \| None` | `None` | 覆盖 tokenizer/processor 的 Jinja2 chat template |
| `trust_remote_code` | `bool` | `false` | 是否加载 Hub 远端代码 |
| `flash_attn` | `str` | `sdpa` | `eager`、`sdpa` 或 `flash_attention_2` |
| `model_class` | `str` | `llm` | `llm`、`cls` 或 `other` |
| `init_config` | `dict \| None` | `None` | 模型初始化配置 |
| `peft_config` | `dict \| None` | `None` | LoRA 或 Freeze 配置 |
| `kernel_config` | `dict \| None` | `None` | 融合算子加速配置 |
| `quant_config` | `dict \| None` | `None` | 量化配置 |

`llm` 加载因果语言模型或 image-to-text 模型，`cls` 加载单标签 token
classification 模型，`other` 使用 `AutoModel`。

## peft_config

### LoRA

设置 `name: lora`：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `r` | `int` | `8` | LoRA rank |
| `lora_alpha` | `int` | `16` | LoRA alpha |
| `lora_dropout` | `float` | `0.05` | LoRA dropout |
| `target_modules` | `list[str] \| str` | `all` | 目标模块 |
| `use_rslora` | `bool` | `false` | 是否启用 RS-LoRA |
| `use_dora` | `bool` | `false` | 是否启用 DoRA |
| `modules_to_save` | `list[str] \| None` | `None` | 额外保存模块 |
| `adapter_name_or_path` | `list[str] \| str \| None` | `None` | adapter 路径 |
| `export_dir` | `str \| None` | `None` | 合并导出目录 |
| `export_size` | `int` | `5` | 导出分片大小，GB |
| `export_hub_model_id` | `str \| None` | `None` | Hub 仓库 ID |
| `infer_dtype` | `str` | `auto` | `auto`、`float16`、`float32` 或 `bfloat16` |
| `export_legacy_format` | `bool` | `false` | 是否使用旧式权重格式 |

### Freeze

设置 `name: freeze`：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `freeze_trainable_layers` | `int` | `2` | 正数选择最后 N 层，负数选择最前 N 层 |
| `freeze_trainable_modules` | `list[str] \| str` | `all` | 层内可训练模块 |
| `freeze_extra_modules` | `list[str] \| str \| None` | `[]` | 额外可训练模块 |
| `cast_trainable_params_to_fp32` | `bool` | `true` | 是否将可训练参数转换为 fp32 |

## quant_config

当前 v1 注册了 `bnb` 和 `auto` 两个量化插件入口。`bnb` 使用 bitsandbytes；`auto` 在指定有效位宽后也转交 `bnb`，当前没有按模型或环境切换到其他量化后端的逻辑。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str` | 必填 | `bnb` 或 `auto` |
| `quantization_bit` | `int \| None` | `None` | 量化加载位宽，4 或 8 |
| `compute_dtype` | `str \| torch.dtype` | `float16` | 4-bit 计算和存储 dtype |
| `double_quantization` | `bool` | `true` | 是否启用 4-bit double quant |
| `quantization_type` | `str` | `nf4` | 4-bit 量化格式，`nf4` 或 `fp4` |

4-bit 分支将 `compute_dtype`、`double_quantization` 和 `quantization_type` 传给 [Hugging Face BitsAndBytesConfig](https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig)，其中 NF4 和 FP4 是 bitsandbytes 的两种 4-bit 格式。8-bit 分支设置 `load_in_8bit=True`，不使用这三个 4-bit 专属字段。

`quantization_bit` 的字段默认值为 `None`，实际行为取决于 `name`：

- 不设置 `quant_config` 或将其设为 `null` 时，不通过此插件添加量化加载配置。
- `name: auto` 且省略 `quantization_bit`（或设为 `null`）时，保持模型加载参数不变。
- `name: bnb` 且省略 `quantization_bit`（或设为 `null`）时，使用 4-bit 量化。
- `name: auto` 且指定 `quantization_bit` 为 `4` 或 `8` 时，转交 `bnb` 实现处理，仍需满足该实现的依赖和运行条件。

以上描述插件构造量化加载配置的行为。[SFT 的 QLoRA 示例](../feature-guide/sft.md#qlora)展示 `bnb`、4-bit、NF4 与 LoRA 的组合。

## init_config

`init_config` 当前只使用 `name` 选择模型初始化设备：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str` | 必填 | 初始化插件名称 |

| `name` 取值 | 初始化设备 |
|-------------|------------|
| `init_on_default` | 当前分布式设备 |
| `init_on_meta` | meta device |
| `init_on_rank0` | rank 0 使用 CPU，其余 rank 使用 meta |

meta 初始化不能与量化同时使用。

## kernel_config

`kernel_config` 可以配置单个融合算子，也可以启用 Liger Kernel 这类包含多项优化的外部加速库。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str` | 必填 | `auto`、单个实现名称或逗号分隔的多个名称 |

可用名称：

- `liger_kernel`
- `cuda_fused_moe`
- `flash-linear-attention`
- `npu_fused_moe`
- `npu_fused_rmsnorm`
- `npu_fused_rope`
- `npu_fused_swiglu`

### Flash Linear Attention

当 `name` 包含 `flash-linear-attention` 时，可以使用以下专属字段：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `include_kernels` | `str` | `auto` | `auto` 或逗号分隔的 FLA 算子名称 |
| `chunk_size` | `int` | `64` | chunk size；可选 `16`、`32`、`64` |

用法见[融合算子加速](../feature-guide/kernel_acceleration.md)。
