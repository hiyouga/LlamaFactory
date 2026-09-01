我们提供了多样化的大模型微调示例脚本。

请确保在 `LLaMA-Factory` 目录下执行下述命令。

## 目录

- [LoRA 微调](#lora-微调)
- [QLoRA 微调](#qlora-微调)
- [全参数微调](#全参数微调)
- [合并 LoRA 适配器与模型量化](#合并-lora-适配器与模型量化)
- [推理 LoRA 模型](#推理-lora-模型)
- [杂项](#杂项)

使用 `CUDA_VISIBLE_DEVICES`（GPU）或 `ASCEND_RT_VISIBLE_DEVICES`（NPU）选择计算设备。

LLaMA-Factory 默认使用所有可见的计算设备。

基础用法：

```bash
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
```

高级用法：

```bash
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml \
    learning_rate=1e-5 \
    logging_steps=1
```

```bash
bash examples/train_lora/qwen3_lora_sft.sh
```

## 示例

### LoRA 微调

#### （增量）预训练

```bash
llamafactory-cli train examples/train_lora/qwen3_lora_pretrain.yaml
```

#### 指令监督微调

```bash
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
```

#### 多模态指令监督微调

```bash
llamafactory-cli train examples/train_lora/qwen3vl_lora_sft.yaml
```

#### DPO/ORPO/SimPO 训练

```bash
llamafactory-cli train examples/train_lora/qwen3_lora_dpo.yaml
```

#### 多模态 DPO/ORPO/SimPO 训练

```bash
llamafactory-cli train examples/train_lora/qwen3vl_lora_dpo.yaml
```

#### 奖励模型训练

```bash
llamafactory-cli train examples/train_lora/qwen3_lora_reward.yaml
```

#### KTO 训练

```bash
llamafactory-cli train examples/train_lora/qwen3_lora_kto.yaml
```

#### 预处理数据集

对于大数据集有帮助，在配置中使用 `tokenized_path` 以加载预处理后的数据集。

```bash
llamafactory-cli train examples/train_lora/qwen3_preprocess.yaml
```

#### 多机指令监督微调

```bash
FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=1 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
```

### 支持弹性和容错的多机指令监督微调

要启动一个支持弹性节点和容错的多机指令微调，在每个节点上执行以下命令。弹性节点数量范围为 `MIN_NNODES:MAX_NNODES`，每个节点最多允许因为错误重启 `MAX_RESTARTS` 次。`RDZV_ID` 应设置为一个唯一的作业 ID（由参与该作业的所有节点共享）。更多细节可以参考官方文档 [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html)。

```bash
FORCE_TORCHRUN=1 MIN_NNODES=1 MAX_NNODES=3 MAX_RESTARTS=3 RDZV_ID=llamafactory MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train examples/train_full/qwen3_full_sft.yaml
```

#### 使用 DeepSpeed ZeRO-3 平均分配显存

```bash
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/qwen3_lora_sft_ds3.yaml
```

#### 使用 Ray 在 4 张 GPU 上微调

```bash
USE_RAY=1 llamafactory-cli train examples/train_lora/qwen3_lora_sft_ray.yaml
```

### QLoRA 微调

#### 基于 4/8 比特 Bitsandbytes/HQQ/EETQ 量化进行指令监督微调（推荐）

```bash
llamafactory-cli train examples/train_qlora/qwen3_lora_sft_otfq.yaml
```

#### 在 NPU 上基于 4 比特 Bitsandbytes 量化进行指令监督微调

```bash
llamafactory-cli train examples/train_qlora/qwen3_lora_sft_bnb_npu.yaml
```

#### 基于 4/8 比特 GPTQ 量化进行指令监督微调

```bash
llamafactory-cli train examples/train_qlora/llama3_lora_sft_gptq.yaml
```

#### 基于 4 比特 AWQ 量化进行指令监督微调

```bash
llamafactory-cli train examples/train_qlora/llama3_lora_sft_awq.yaml
```

#### 基于 2 比特 AQLM 量化进行指令监督微调

```bash
llamafactory-cli train examples/train_qlora/llama3_lora_sft_aqlm.yaml
```

### 全参数微调

#### 在单机上进行指令监督微调

```bash
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen3_full_sft.yaml
```

#### 在多机上进行指令监督微调

```bash
FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train examples/train_full/qwen3_full_sft.yaml
FORCE_TORCHRUN=1 NNODES=2 NODE_RANK=1 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train examples/train_full/qwen3_full_sft.yaml
```

#### 多模态指令监督微调

```bash
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen3vl_full_sft.yaml
```

### 合并 LoRA 适配器与模型量化

#### 合并 LoRA 适配器

注：请勿使用量化后的模型或 `quantization_bit` 参数来合并 LoRA 适配器。

```bash
llamafactory-cli export examples/merge_lora/qwen3_lora_sft.yaml
```

#### 使用 AutoGPTQ 量化模型

```bash
llamafactory-cli export examples/merge_lora/qwen3_gptq.yaml
```

### 保存 Ollama 配置文件

```bash
llamafactory-cli export examples/merge_lora/qwen3_full_sft.yaml
```

### 推理 LoRA 模型

#### 使用 vLLM 多卡推理评估

```
python scripts/vllm_infer.py --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 --template qwen3_nothink --dataset alpaca_en_demo
python scripts/eval_bleu_rouge.py generated_predictions.jsonl
```

#### 使用命令行对话框

```bash
llamafactory-cli chat examples/inference/qwen3_lora_sft.yaml
```

#### 使用浏览器对话框

```bash
llamafactory-cli webchat examples/inference/qwen3_lora_sft.yaml
```

#### 启动 OpenAI 风格 API

```bash
llamafactory-cli api examples/inference/qwen3_lora_sft.yaml
```

### 杂项

#### 使用 GaLore 进行全参数训练

```bash
llamafactory-cli train examples/extras/galore/llama3_full_sft.yaml
```

#### 使用 APOLLO 进行全参数训练

```bash
llamafactory-cli train examples/extras/apollo/llama3_full_sft.yaml
```

#### 使用 BAdam 进行全参数训练

```bash
llamafactory-cli train examples/extras/badam/llama3_full_sft.yaml
```

#### 使用 Adam-mini 进行全参数训练

```bash
llamafactory-cli train examples/extras/adam_mini/qwen2_full_sft.yaml
```

#### 使用 Muon 进行全参数训练

```bash
llamafactory-cli train examples/extras/muon/qwen2_full_sft.yaml
```

#### LoRA+ 微调

```bash
llamafactory-cli train examples/extras/loraplus/llama3_lora_sft.yaml
```

#### PiSSA 微调

```bash
llamafactory-cli train examples/extras/pissa/llama3_lora_sft.yaml
```

#### 深度混合微调

```bash
llamafactory-cli train examples/extras/mod/llama3_full_sft.yaml
```

#### LLaMA-Pro 微调

```bash
bash examples/extras/llama_pro/expand.sh
llamafactory-cli train examples/extras/llama_pro/llama3_freeze_sft.yaml
```

#### FSDP+QLoRA 微调

```bash
bash examples/extras/fsdp_qlora/train.sh
```

#### OFT 微调

```bash
llamafactory-cli train examples/extras/oft/llama3_oft_sft.yaml
```

#### QOFT 微调

```bash
llamafactory-cli train examples/extras/qoft/llama3_oft_sft_bnb_npu.yaml
```

#### 添加新词并初始化词向量

```bash
llamafactory-cli train examples/extras/add_tokens/qwen3_add_tokens_vocab_mean_noise.yaml
```
