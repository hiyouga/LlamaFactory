# 多 Token 预测(MTP)— v1 架构

**English** | [简体中文](v1_mtp_zh.md)

本特性为 v1 架构新增多 Token 预测(Multi-Token Prediction, MTP)支持,改编自
[MindSpeed-LLM](https://gitcode.com/Ascend/MindSpeed-LLM) 中的 FSDP2 MTP 实现
(`mindspeed_llm/fsdp2/models/common/mtp.py`)。

MTP 在 decoder-only 因果语言模型上追加 `K` 个额外预测头。第 `k` 个头预测当前位置
偏移 `k + 2` 处的 token(主头预测偏移 `1` 处)。训练总损失为:

```
total_loss = lm_loss + loss_scale * mtp_loss
```

其中 `mtp_loss` 是各头交叉熵损失的均值(与主 SFT 损失一样按 `loss_weights` 加权),
`loss_scale` 默认为 `0.3`。

## 工作原理

- `MultiTokenPredictionBlock`(`src/llamafactory/v1/plugins/model_plugins/mtp.py`)持有
  `K` 个头。每个头复用基础模型的 decoder layer 类。该 block 拥有共享的
  `enorm`/`hnorm` 归一化层、`e_proj`/`h_proj` 投影层和一个 `final_layernorm`,
  与 MindSpeed-LLM 中的实现完全一致。
- `MTPModelPlugin` 将该 block 挂载为 `model.mtp`,并 patch `model.forward`,
  使训练期间模型输出携带 `mtp_logits`(逐头 logits 列表)。MTP 损失由 trainer 通过
  `compute_mtp_loss` 计算。
- 在 FSDP2 下,MTP 头内部的 decoder layer 会被自动分片:通用的
  `FSDP2Engine.prepare_model` 会包裹基础模型 decoder layer 类的每一个实例,
  其中就包括 `mtp.layers.*.layer`。

## 用法

在 v1 YAML 中加入 `mtp_config` 块:

```yaml
model: Qwen/Qwen3-0.6B
model_class: llm
template: qwen3_nothink

mtp_config:
  name: mtp
  num_layers: 1   # MTP 头的个数(K)
  loss_scale: 0.3 # 可选,默认 0.3

dist_config:
  name: fsdp2
  dcp_path: null

train_dataset: data/v1_sft_demo.yaml
output_dir: outputs/test_mtp
micro_batch_size: 1
cutoff_len: 2048
learning_rate: 1.0e-4
max_steps: 10
```

完整示例(MTP + CP + 检查点保存/续训合一)见
`examples/v1/train_full/train_full_mtp.yaml`。

## 兼容性

MTP 目前面向暴露 `model.model.layers`、`model.model.rotary_emb`、`model.model.norm`
和 `model.lm_head` 的 Llama/Qwen3/Qwen3.5/Mistral 类模型。MTP 头随机初始化;加载不包含
`mtp.*` 键的基础检查点时,这些头保持初始值(出现 missing-key 警告属正常现象)。

### 层选择(混合注意力模型)

每个 MTP 头复用基础模型 decoder layer 的**类**,并从注意力类型为**全局自注意力**的层
克隆而来。对混合注意力模型这一点很关键:Qwen3 混合了 `full_attention` 与
`sliding_attention`;Qwen3.5 混合了 `full_attention` 与 `linear_attention`(GDN)。
MTP 头要在**完整**序列上预测偏移 `k + 2` 处的 token,需要全局上下文,而滑动窗口或
GDN 头(局部/递归视角)是错误的。`_select_layer_idx_for_mtp` 会从
`config.layer_types` 中选取最后一个 `full_attention` 层的下标(对 Llama/Mistral 这类
全 full 模型则回退为最后一层)。被选中的下标会在挂载时打印日志,例如:
`decoder layer cloned from layer_idx=7 [full_attention]`。

## MTP 权重的保存与加载

`mtp.embed_tokens` 与 `mtp.output_layer` 与基础模型的 embedding 和 `lm_head` **共享**,
因此在 `save_pretrained` 之前会先从 state dict 中剥离(`strip_shared_mtp_keys`),
以避免 transformers 报 "shared tensors not properly defined" 错误。最终只有 MTP 专有的
张量(`layers.*`、`enorm`/`hnorm`/`e_proj`/`h_proj`/`final_layernorm`)会随基础模型
权重一起写出。

加载时,`from_pretrained` 会把 `mtp.*` 键当作 unexpected 丢弃(MTP block 是运行时
挂载的)。因此 `ModelEngine` 会调用 `apply_mtp`(重新创建 block 并重新共享
embedding/`lm_head`),再调用 `load_mtp_weights` 从检查点恢复已保存的 MTP 张量。
以上均为自动完成,无需额外配置。FSDP2 meta 路径会通过常规 HF 权重循环加载 `mtp.*`,
DCP 续训则按 FQN 恢复它们。

## 上下文并行(MTP + CP)

MTP 同样支持 Ulysses 上下文并行(CP)。CP 要求 `dist_config.name: fsdp2` 且
`flash_attn: flash_attention_2`(与非 MTP CP 的约束相同)。MTP 与 CP 同时开启时:

- MTP decoder layer 会走与主模型相同的、全局 patch 过的 `_flash_attention_forward`,
  因此自动参与 Ulysses 注意力。(每个 MTP 头都是 `full_attention` 层,所以它总是走
  `_flash_attention_forward`。)
- `BaseTrainer.fit` 会路由到 `sequence_parallel_mtp_loss` 插件,它计算主头的 CP 损失
  (不变)加上缩放后的 MTP 损失。逐头 MTP 损失在完整序列上计算,方式是在 CP 组内
  all-gather `labels` / `loss_weights` / `log_probs`(见 `mtp.py` 中带 `cp_group` 的
  `compute_mtp_loss`),与单头的 `sequence_parallel_loss` 插件保持一致。
- MTP 的输入移位(`shift_input_ids_for_mtp`)是 CP 感知的:每个 rank 的块尾会用
  **下一个 rank 的第一个 token**(在 CP 组内 all-gather 得到)填充,而不是填充
  pad 值,从而保证每个 CP 边界处的 next-token embedding 是正确的。只有全局最后一个
  rank 的块尾(真正的序列结尾)才填充 pad。

```yaml
mtp_config:
  name: mtp
  num_layers: 1
  loss_scale: 0.3

flash_attn: flash_attention_2

dist_config:
  name: fsdp2
  dcp_path: null
  cp_mode: ulysses
  cp_size: 2
```

CP 相关配置见 `examples/v1/train_full/train_full_mtp.yaml` 的 CP 部分。
CP 不支持 DeepSpeed(请使用 FSDP2)。
