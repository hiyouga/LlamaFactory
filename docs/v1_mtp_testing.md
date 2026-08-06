# MTP 测试流程（端到端）

本文档说明如何端到端验证 MTP（Multi-Token Prediction）的三个修复点：权重保存/加载（Q2）、断点续训（DCP）、上下文并行（CP, Q3）。配套三个 yaml，均在 `examples/v1/train_full/` 下。

> 所有命令需在 GPU 机器、仓库根目录执行，且设置 `USE_V1=1` 走 v1 架构。当前 WSL 环境无 GPU，仅能跑单元测试（见末尾「单元测试」一节）。

## 前置准备

- GPU 机器，已安装 flash-attn（路径 C 必需；路径 A/B 单卡 FSDP2 不需要）
- 仓库已 clone，分支 `feature/mtp`（含三处修复）
- `Qwen/Qwen3-0.6B` 可从 HuggingFace 拉取（首次训练会自动下载）

## 三条测试路径总览

| 路径 | yaml | 验证修复点 | 需 flash-attn | 需多卡 |
|---|---|---|---|---|
| A 最终导出 | `train_full_mtp_save_load.yaml` | Q2（save_model 不报错 + mtp.* 可重载） | 否 | 否（单卡） |
| B 断点续训 | `train_full_mtp_resume.yaml` | Q2 续训侧（DCP 存/读 mtp.* + loss 连续） | 否 | 否（单卡） |
| C 上下文并行 | `train_full_mtp_cp.yaml` | Q3（MTP 层 CP：attention + loss + shift 边界） | 是 | 是（≥2 卡） |

三条路径都用 `Qwen/Qwen3-0.6B`（全 `full_attention`，非 GDN 混合模型），按你的要求先不涉及 GDN。

---

## 路径 A：最终导出（save_model）

**验证目标**：训练结束 `save_model` 不再报 `shared tensors` RuntimeError，且导出的 `mtp.*` 权重能被完整读回（等于训练值、不等于随机初值）。

**步骤 1 — 训练 + 导出**

```bash
USE_V1=1 llamafactory-cli train examples/v1/train_full/train_full_mtp_save_load.yaml
```

- 修复前：训练最后一步 `save_model` 抛 `RuntimeError: shared tensors [{'model.embed_tokens.weight','mtp.embed_tokens.weight'}] not properly defined`
- 修复后：正常完成，`outputs/test_mtp_save_load/` 生成 `model.safetensors` + `config.json`

**步骤 2 — 加载 + 对比**（单进程，无需 torchrun）

```bash
PYTHONPATH=src python scripts/verify_mtp_save_load_e2e.py \
    --output_dir outputs/test_mtp_save_load \
    --mtp_num_layers 1 --mtp_loss_scale 0.3
```

脚本走真实加载路径（`from_pretrained` → `apply_mtp` → `load_mtp_weights`），验证四项：

| 检查 | 含义 | 修复前表现 |
|---|---|---|
| `save side` | `mtp.*` 写入 safetensors、共享 key 被剥离 | 保存直接报错，无输出 |
| `load side` | 模型参数与磁盘值逐位相等 | from_pretrained 丢 mtp.*，重建随机 → MISMATCH |
| `non-random` | 加载值 ≠ 随机初值（证明恢复的是训练后权重） | 加载 = 随机初值 |
| `re-tied` | mtp.embed_tokens/output_layer 重新共享主模型 | — |

**通过标准**：四项全 `PASS`。

---

## 路径 B：断点续训（DCP save/resume）

**验证目标**：中途快照用 DCP 格式存 `mtp.*`（按 FQN），续训时恢复，且 loss/mtp_loss 连续不跳变。

**步骤 1 — 训练到第 10 步存快照，继续跑到第 20 步**

```bash
USE_V1=1 llamafactory-cli train examples/v1/train_full/train_full_mtp_resume.yaml
```

yaml 里 `save_steps: 10` 会在第 10 步存 `outputs/test_mtp_resume/checkpoint-10/`（含 `model/`、`optimizer/`、`scheduler.pt` 等 DCP 内容），然后继续跑到 `max_steps: 20`。

**记下第 10 步和第 11 步附近的 `loss` 和 `mtp_loss`**（日志里每 `logging_steps` 打印一次）。

**步骤 2 — 从 checkpoint-10 续训**

编辑 yaml，取消注释 resume 行：
```yaml
resume_from_checkpoint: outputs/test_mtp_resume/checkpoint-10
# 或用 auto 自动找最新：
# resume_from_checkpoint: auto
```

重新跑（建议先删掉 `outputs/test_mtp_resume/` 里除 `checkpoint-10` 外的产物，或换 `output_dir`，避免混淆）：
```bash
USE_V1=1 llamafactory-cli train examples/v1/train_full/train_full_mtp_resume.yaml
```

**通过标准**：
1. 续训不报错（`load_checkpoint` 按 FQN 恢复 mtp.*，DCP 不触发 shared tensor 检查）
2. 续训后第 11 步的 `loss` 和 `mtp_loss` **紧接着**中断前第 10 步的值（不跳回初始高 loss）——证明 optimizer 状态 + mtp 权重都恢复了

> **注意**：两次运行的 `mtp_config`（num_layers/loss_scale）必须完全一致，否则 graft 出的 MTP 结构与 checkpoint FQN 对不上会报错。
>
> 可选：yaml 里取消注释 `save_ckpt_as_hf: true`，会在 `checkpoint-10/hf_model/` 额外存一份 HF 格式——这条也走 Q2 的 strip 逻辑，可顺带验证中途快照的 HF 导出路径。

---

## 路径 C：上下文并行（Ulysses + MTP）

**验证目标**：MTP 在 CP 下正确工作——attention 走 Ulysses、loss 跨 CP group all-gather、`shift_input_ids_for_mtp` 跨 chunk 边界正确。

**前置**：≥2 GPU + flash-attn。CP 要求 `dist_config.name: fsdp2` + `flash_attn: flash_attention_2`。

```bash
# 不要手动 `torchrun -m llamafactory.cli` —— v1 launcher 会自己检测多卡并拉起 torchrun,
# 同时整理 argv 让 yaml 被正确解析。手动 torchrun -m 会污染 sys.argv, 导致
# data_args.train_dataset 变成 None (报 'NoneType' object has no attribute 'endswith').
USE_V1=1 llamafactory-cli train examples/v1/train_full/train_full_mtp_cp.yaml

# 想指定卡数（而非用所有可见卡）, 用环境变量:
# NPROC_PER_NODE=2 USE_V1=1 llamafactory-cli train examples/v1/train_full/train_full_mtp_cp.yaml
```

**通过标准**：
1. 训练正常跑完不报错
2. 日志出现 `Replaced _flash_attention_forward ... for sequence parallel`（Ulysses attention 生效）
3. 日志里 `mtp_loss` 字段健康（finite、大致下降趋势）

**关于 GDN**：本 yaml 用 `Qwen3-0.6B`（全 full_attention），**不涉及 GDN CP**。若要测 Qwen3.5 这类 GDN+full 混合模型的 CP，需：
1. 把 `model` 换成 Qwen3.5 系列（如 `Qwen/Qwen3.5-4B`）
2. cherry-pick PR [#10727](https://github.com/hiyouga/LlamaFactory/pull/10727)（`gdn_attention.py`）—— GDN CP 尚未合入 upstream main

PR #10727 只解决主模型 GDN 层的 CP，与 MTP 层 CP（本修复 Q3）相互独立。

---

## 单元测试（无 GPU 也可跑）

CPU 环境可跑的单元测试，覆盖三个修复的核心逻辑（用 gloo 模拟 2 进程 CP）：

```bash
cd LlamaFactory
WANDB_DISABLED=true PYTHONPATH=src python3 -m pytest -vv \
    --import-mode=importlib tests_v1/plugins/model_plugins/test_mtp.py
```

| 测试 | 验证 |
|---|---|
| `test_mtp_save_load` | 路径 A 的核心逻辑（save→reload 权重恢复） |
| `test_mtp_shift_input_ids_cp` | 路径 C 的 Q3 修复（CP shift 跨边界正确） |
| `test_mtp_cp_alignment` | 路径 C 的 loss 对齐（CP loss = 全序列 loss） |
| 其余 5 个 | MTP 基础功能不回归 |

8 个全过即核心逻辑正确。端到端（A/B/C）需 GPU。

---

## 排查指引

| 现象 | 可能原因 |
|---|---|
| 路径 A `save_model` 报 shared tensors 错 | Q2 的 `strip_shared_mtp_keys` 没生效，检查 `fsdp2.py`/`base_trainer.py` 的 save 路径 |
| 路径 A 脚本 `load side` MISMATCH | `load_mtp_weights` 没读到 mtp.*，检查 `model_engine.py` 是否在 apply_mtp 后调用了它 |
| 路径 B 续训 loss 跳变 | optimizer 或 mtp 权重没恢复，检查 DCP `load_checkpoint` 的 FQN 匹配；确认两次 `mtp_config` 一致 |
| 路径 C 报 `requires flash attention` | 没装 flash-attn 或 yaml 没设 `flash_attn: flash_attention_2` |
| 路径 C 报 qwen3.5 不支持 | 用了 Qwen3.5 但没合 PR #10727；换回 Qwen3-0.6B 或先合 PR |
