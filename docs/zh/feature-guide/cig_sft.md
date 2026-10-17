# CIG-SFT

CIG-SFT 已接入 LLaMA-Factory 现有的 `llamafactory-cli train` Transformers SFT 入口。首版支持文本单轮样本的全参训练和 Freeze 训练；冻结的 reference model 使用现有的 DeepSpeed/FSDP 模型放置逻辑；关闭 `use_cig_sft_loss` 时仍走普通 SFT 损失。

## 快速开始

使用 `examples/extras/cig_sft/qwen25_full_sft.yaml`：

```bash
llamafactory-cli train examples/extras/cig_sft/qwen25_full_sft.yaml
```

配置中的 `cig_erased_prompt` 为 `default`。训练前，冻结的 reference model 分别在原始 user message 和替换 prompt 两个 teacher-forced 视图上前向，得到各位置的 next-token 概率分布；随后提取标准答案 token 的条件概率并比较相应的对数概率，再构造 detached 的 token credit。credit 写入 `cig_credit_cache`，相同 reference model 和 tokenized 数据的后续运行可以复用。`cig_lambda`、`cig_eps` 和 `cig_max_beta` 控制投影和数值保护。

## 数据列

普通 Alpaca/ShareGPT 格式保持不变。若每条样本需要不同的替换 prompt，可以在 `data/dataset_info.json` 中登记额外列：

```json
{
  "my_cig_dataset": {
    "file_name": "my_cig_dataset.jsonl",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "erased_prompt": "erased_instruction"
    }
  }
}
```

非空的行级值优先于数据集级 `cig_erased_prompt`；空值或 null 会回退到数据集级配置。内置 preset 为 `default`、`math`、`code` 和 `medical`，其他字符串按 literal replacement prompt 处理。

## 当前范围和限制

当前实现要求每条样本只有一个 user message 和一个 assistant response。暂不支持 packing、streaming、多模态 erased view、Megatron Bridge、MCA、HyperParallel、LoRA/OFT 以及其他非标准 SFT 后端。reference model 在训练期间保留；credit 写入缓存后，训练步骤会复用缓存，不再重复执行 reference-model 前向。评估和生成使用普通模型输入与普通 SFT loss。

删除或设置 `use_cig_sft_loss: false` 即可回到标准 SFT。CIG-SFT 不能和 DFT、ASFT、EAFT 同时启用。
