# FSDPTurbo EP/EFSDP 与 LlamaFactory FSDP2/CP 设计说明

English version: [FSDPTurbo EP/EFSDP and LlamaFactory FSDP2/CP Design](../../../en/advanced/distributed/fsdpturbo-ep-efsdp.md)

本文描述 `fsdpturbo` distributed plugin 的当前实现。核心原则是保持两侧职责清晰：

- FSDPTurbo 负责专家并行（EP）、专家参数分片（EFSDP）和设备算子注册。
- LlamaFactory 负责进程初始化、基础 DeviceMesh、外层 FSDP2、CP、模型初始化与权重加载。
- LlamaFactory 的集成层负责把两套参数布局组合起来，并处理跨 Mesh 的梯度范数。

## 1. 配置边界

公共并行拓扑放在 `TrainingArguments` 顶层，FSDPTurbo 私有参数保留在 `dist_config`：

```yaml
cp_size: 1

dist_config:
  name: fsdpturbo
  ep_size: 16
  ep_dispatcher: eager
```

最小示例中的字段职责如下：

- `ep_size`：专家并行组大小。
- `ep_dispatcher`：FSDPTurbo EP dispatcher，默认为 `eager`。

`dp_size`、`cp_size`、`cp_mode`、`mp_replicate_size`、`mp_shard_size` 和 `dist_timeout`
属于公共拓扑字段，继续放在顶层。`dist_config` 会被严格解析为 `FSDPTurboParams`；如果把公共拓扑
字段误放进去，会直接报错，而不是静默忽略。

顶层训练参数 `bf16` 同时控制 FSDPTurbo 的参数存储和计算 dtype。backend 会在 FSDP materialization
前完成模型 dtype 转换，因此 `ModelEngine` 不需要读取 distributed backend 配置。

以下高级字段为可选项，因此没有写入上面的最小 YAML 示例：

- `fsdp_ignored_modules`：额外排除在 LlamaFactory 外层 FSDP2 之外的模块。模型规格选中的专家参数
  会被集成层自动加入忽略集合，普通配置无需重复填写。
- `hook_modules`：FSDPTurbo EFSDP hook 的可选模块模式，默认为空列表。
- `fsdp_implementation`：FSDPTurbo EFSDP 实现，可选 `native` 或 `custom`，默认为 `native`。

EFSDP 的目标由模型规格决定。Attention、Embedding、LM Head 等非专家参数不进入 FSDPTurbo
EFSDP plan，而是继续由 LlamaFactory 外层 FSDP2 管理。

模型相关的模块路径和准备逻辑统一由 `FSDPTurboEPModelSpec` 注册表管理。当前内置 `qwen3_moe`
和 `qwen3_5_moe`；未注册的模型会明确报错。`ep_modules` 和 `ep_fsdp_modules` 不属于 YAML
接口，严格参数解析会拒绝这两个字段，避免用户配置与模型实际结构失配。

## 2. Mesh 初始化

LlamaFactory 的 `DistributedInterface` 只初始化自身原有的 model/data mesh。它不感知 EP、EFSDP，
也不为 distributed plugin 提供额外 mesh 注册接口。FSDPTurbo 的专家拓扑由插件文件内的
`FSDPTurboParallelState` 独立创建和持有：

```text
run_sft / run_dpo / run_rm
  -> DistributedInterface(training_args)
     -> 初始化 LlamaFactory model/data mesh
  -> DistributedPlugin("fsdpturbo").shard_model(...)
     -> FSDPTurboFSDP2Engine.__init__()
        -> FSDPTurboParallelState.initialize()
           -> 初始化并保存 expert parent mesh 及其子 mesh
```

`FSDPTurboParallelState` 创建专家侧四维父 Mesh：

```text
(edp, efsdp, ep, expert_cp)
```

当前尺寸计算为：

```text
dp_size       = world_size / cp_size
ep_fsdp_size  = dp_size / ep_size
edp_size      = dp_size / (ep_size * ep_fsdp_size)
mesh_shape    = (edp_size, ep_fsdp_size, ep_size, cp_size)
```

状态对象保存 `edp_mesh`、`efsdp_mesh`、`ep_mesh` 和 `expert_cp_mesh`。插件内部模型切分和梯度范数
都从这个状态对象读取专家通信域；LlamaFactory 其他 backend 不需要实现或感知这些接口。状态初始化
会校验 `ep_size` 为正数且能够整除 `dp_size`，重复初始化时也会拒绝拓扑发生变化。

## 3. 模型切分顺序

模型包装顺序必须保持为“专家侧优先，外层 FSDP2 随后”：

```text
DistributedPlugin("fsdpturbo")
  -> FSDPTurboFSDP2Engine.shard_model(model)
     -> prepare_model_ep(model)
        -> expert_parallelize_modules(model, ep_mesh, ep_plan)
        -> expert_fully_shard_modules(model, efsdp_mesh, ep_plan, fsdp_plan)
        -> 收集专家参数作为 ignored_params
     -> FSDP2Engine.prepare_model(model, ignored_params=...)
        -> 对剩余 Transformer Layer 和根模块执行 outer fully_shard
```

这样可以避免同一专家参数同时被 EFSDP 和外层 FSDP2 管理。外层 FSDP2 仍复用 LlamaFactory
原有的初始化、checkpoint 和保存流程。

LlamaFactory 集成层接受 `eager`、`fused`、`mc2` 和 `domino`，并将选项原样传给
FSDPTurbo。这四种模式的实现边界和当前验证状态不同：

| Dispatcher | 主要路径 | 额外要求 | 本 PR 验证状态 |
| --- | --- | --- | --- |
| `eager` | 使用 PyTorch 实现 permute、unpermute 和 grouped matmul，张量仍在当前加速设备上，通过标准 AllToAll 完成 token dispatch/combine | 依赖最少，用作参考实现 | 已在 A3 上完成精度和性能验证 |
| `fused` | 保持相同的 AllToAll 拓扑，将 permute、unpermute 和 grouped matmul 切换为设备融合算子 | 需要对应的设备算子、dtype 和 layout 支持；存在空专家时可回退到 eager 局部算子 | 已在 A3 上完成精度和性能验证 |
| `mc2` | 使用专用算子融合 AllToAllV 和 grouped matmul，减少通信与计算之间的中间开销 | 依赖 MC2 NPU 算子、HCCL communicator 以及对应的 shape/dtype 约束 | FSDPTurbo 提供实现，本 PR 未做端到端验证 |
| `domino` | 将专家模块输入的第一维分成两片，使用独立通信流和 event 重叠 AllToAll 与专家计算 | 需要异步 stream/event 支持，且两个分片都要有足够的 token 工作量才能覆盖调度开销 | FSDPTurbo 提供实现，本 PR 未做端到端验证 |

当前只验证 `eager` 和 `fused`，是因为它们分别覆盖参考实现和 A3 常用设备融合路径，可用于隔离并验证
LlamaFactory 与 FSDPTurbo 之间的 EP/EFSDP 集成正确性。本次实验矩阵没有继续扩展到 `mc2` 和
`domino`：它们还引入了额外的算子、通信调度和输入形状约束，需要独立比较数值、长步稳定性和 profiler 结果。
因此，它们在配置接口上可选，但不应从本 PR 的实验结果推断为已达到相同的稳定性、精度或性能水平。

## 4. FSDPTurbo 依赖入口

LlamaFactory 从各功能的定义模块直接导入所需对象：

```python
from fsdp_turbo.distributed.expert_parallel.expert_fully_shard_parallel import (
    expert_fully_shard_modules,
)
from fsdp_turbo.distributed.expert_parallel.expert_parallel import expert_parallelize_modules
from fsdp_turbo.fsdp_turbo_config import EPPlanConfig, FSDPPlanConfig
from fsdp_turbo.utils.str_match import module_name_match
```

导入发生在 `prepare_model_ep()` 内，因此没有安装 FSDPTurbo 时，其他 distributed backend 仍可正常导入。
这里不通过 `fsdp_turbo.distributed.__init__` 聚合导出，避免 package 初始化期间的额外依赖和潜在循环导入。

## 5. 梯度范数

外层参数和专家参数可能属于不同 DTensor Mesh，不能直接放入一次标准 `clip_grad_norm_()`。
`fsdpturbo` plugin 按参数所属 Mesh 分组计算局部 p 次方和：

- 非专家参数沿 DP 和 CP group 汇总。
- 专家参数沿 `FSDPTurboParallelState` 保存的 EFSDP、EP 和 expert-CP group 汇总。
- 汇总得到全局范数后，对所有本地梯度应用同一个 clipping coefficient。

启动阶段会执行一次零梯度 warmup，使相关 collective 在正式训练前完成初始化。
当前这是 `fsdpturbo` backend 的专用实现；其他 backend 继续保留原有梯度范数路径，等待上游
distributed plugin 解耦后再统一公共接口。

## 6. 权重加载

LlamaFactory 保留 `init_on_meta` 和 safetensors 加载流程。父类 `FSDP2Engine` 的加载器通过
`self._copy_weights(...)` 动态调用 FSDPTurbo engine 的覆写实现，因此该方法不是未使用代码。
它支持包含多个 `Shard` placement 的 DTensor，按各 Mesh 维度依次计算当前 rank 对应的本地切片。
模型保存和 checkpoint 接口继续复用 LlamaFactory FSDP2 实现。

## 7. Kernel plugin

FLA 算子不属于 distributed config。算子选择通过独立的 `kernel_config` 完成：

```yaml
kernel_config:
  name: auto, flash-linear-attention
  include_kernels: chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
  chunk_size: 32
```

调用链如下：

```text
ModelEngine
  -> apply_kernels("auto, flash-linear-attention")
     -> LlamaFactory 当前加速器对应的 auto kernels
     -> KernelPlugin("flash-linear-attention").apply(...)
        -> fsdp_turbo.ops.get_op()
           -> FSDPTurbo device operator registry
        -> fsdp_turbo.utils.patch.patch_model_members()
           -> FLA backend implementation
```

`chunk_size` 当前支持 `16`、`32` 和 `64`，默认值为 `64`。Kernel plugin 与 distributed plugin
彼此独立。`name: flash-linear-attention` 只安装所选 FLA 算子；逗号分隔的
`name: auto, flash-linear-attention` 会在分布式切分前组合 LlamaFactory 当前加速器的 auto kernels
与 FLA plugin。LlamaFactory 负责算子名到模型属性的映射和 `chunk_size` 参数绑定；FSDPTurbo 负责设备
算子注册、选择和通用 callable patch。FLA 依赖可选的外部三方件，因此保持显式选择，不属于内置
`auto` 集合。FSDPTurbo
随后会替换目标专家模块的 `forward`，所以专家计算的最终路径由 `ep_dispatcher` 决定；auto 阶段
应用的 MoE kernel 不会作为独立的第二条专家执行路径保留下来。

## 8. CP 运行约束与验证范围

`init_on_meta` 构造模型时必须与 `from_pretrained` 路径一样传递 `attn_implementation`，否则模型会退回
非 FlashAttention 实现，Ulysses CP 无法启动。Ulysses 在调用 Hugging Face FlashAttention 前重建全局
attention mask；只有二维 position IDs 才参与 packed-sequence 检测。Qwen3.5 mRoPE 等多轴 position IDs
已经在 rotary embedding 中消费，不应传入 FlashAttention 的 packed-sequence 检测逻辑。

当前实现已在 Atlas 900 A3 SuperPoD 和 Atlas 950 SuperPoD 上用 Qwen3.5-35B-A3B 完成以下
BF16、AdamW full SFT 验证。本次重验证使用 FSDPTurbo `0e96fbc`；A3 环境为 CANN 9.0.0、
PyTorch 2.7.1 和 torch-npu 2.7.1.post4，A5 环境为 CANN 9.1.0-beta.3、PyTorch 2.10.0 和
torch-npu 2.10.0.post2。表中性能按第 1 步至第 100 步的日志时间戳计算，不包含首步前的初始化、
编译和训练后的模型保存时间：

| 机器型号 | CP | EP | EFSDP | Checkpoint | Kernel / Dispatcher | 步数 | Loss（首步 -> 末步） | 性能 | 结果 |
| --- | ---: | ---: | ---: | --- | --- | ---: | --- | ---: | --- |
| Atlas 900 A3 SuperPoD | 1 | 16 | 1 | 关闭 | FLA（chunk size 16）/ eager | 100 | 1.3361 -> 0.0793 | 2.51 s/it | 通过并完成保存 |
| Atlas 900 A3 SuperPoD | 1 | 16 | 1 | 关闭 | FLA（chunk size 16）/ fused | 100 | 1.3354 -> 0.1179 | 2.17 s/it | 通过并完成保存 |
| Atlas 900 A3 SuperPoD | 2 | 4 | 2 | 关闭 | auto + FLA（chunk size 64）/ fused | 100 | 1.8114 -> 0.5260 | 7.65 s/it | 通过并完成保存 |
| Atlas 900 A3 SuperPoD | 2 | 4 | 2 | 关闭 | auto + FLA（chunk size 64）/ eager | 100 | 1.8095 -> 0.5596 | 5.88 s/it | 通过并完成保存 |
| Atlas 950 SuperPoD | 1 | 8 | 1 | 关闭 | 未配置 kernel plugin / eager | 100 | 1.3575 -> 0.4439 | 2.68 s/it | 通过并完成保存 |

五组训练的 loss 和 grad norm 均保持有限，并完成 100 步及模型保存。同一切分下，EP16 eager/fused
的逐步 loss 相关系数为 0.997，CP2/EP4/EFSDP2 eager/fused 为 0.977，说明两种 dispatcher 的
优化轨迹一致。性能收益与切分有关：EP16 下 fused 比 eager 快约 13%，而加入 CP 和 EFSDP 后 fused
比 eager 慢约 30%，因此不能把 fused 视为所有 mesh 的默认最优选择。

EP16 两组使用 global batch 16 和 cutoff length 256；CP2 两组使用 global batch 8 和 cutoff length
128；A5 组使用 global batch 8 和 cutoff length 256。因此，首末 loss 用于验证各组自身的收敛趋势，
不同切分组之间的绝对 loss 不应直接作为精度等价结论。
