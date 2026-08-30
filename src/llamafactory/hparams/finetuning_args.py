# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class FreezeArguments:
    r"""Arguments pertaining to the freeze (partial-parameter) training."""

    freeze_trainable_layers: int = field(
        default=2,
        metadata={
            "help": (
                "The number of trainable layers for freeze (partial-parameter) fine-tuning. "
                "Positive numbers mean the last n layers are set as trainable, "
                "negative numbers mean the first n layers are set as trainable."
            )
        },
    )
    freeze_trainable_modules: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of trainable modules for freeze (partial-parameter) fine-tuning. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the available modules."
            )
        },
    )
    freeze_extra_modules: str | None = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from hidden layers to be set as trainable "
                "for freeze (partial-parameter) fine-tuning. "
                "Use commas to separate multiple modules."
            )
        },
    )


@dataclass
class LoraArguments:
    r"""Arguments pertaining to the LoRA training."""

    additional_target: str | None = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from LoRA layers to be set as trainable "
                "and saved in the final checkpoint. "
                "Use commas to separate multiple modules."
            )
        },
    )
    lora_alpha: int | None = field(
        default=None,
        metadata={"help": "The scale factor for LoRA fine-tuning (default: lora_rank * 2)."},
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."},
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."},
    )
    lora_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of target modules to apply LoRA. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    loraplus_lr_ratio: float | None = field(
        default=None,
        metadata={"help": "LoRA plus learning rate ratio (lr_B / lr_A)."},
    )
    loraplus_lr_embedding: float = field(
        default=1e-6,
        metadata={"help": "LoRA plus learning rate for lora embedding layers."},
    )
    use_rslora: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the rank stabilization scaling factor for LoRA layer."},
    )
    use_dora: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the weight-decomposed lora method (DoRA)."},
    )
    pissa_init: bool = field(
        default=False,
        metadata={"help": "Whether or not to initialize a PiSSA adapter."},
    )
    pissa_iter: int = field(
        default=16,
        metadata={"help": "The number of iteration steps performed by FSVD in PiSSA. Use -1 to disable it."},
    )
    pissa_convert: bool = field(
        default=False,
        metadata={"help": "Whether or not to convert the PiSSA adapter to a normal LoRA adapter."},
    )
    create_new_adapter: bool = field(
        default=False,
        metadata={"help": "Whether or not to create a new adapter with randomly initialized weight."},
    )


@dataclass
class OFTArguments:
    r"""Arguments pertaining to the OFT training."""

    additional_target: str | None = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from LoRA layers to be set as trainable "
                "and saved in the final checkpoint. "
                "Use commas to separate multiple modules."
            )
        },
    )
    module_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for the OFT fine-tuning."},
    )
    oft_rank: int = field(
        default=0,
        metadata={"help": "The intrinsic dimension for OFT fine-tuning."},
    )
    oft_block_size: int = field(
        default=32,
        metadata={"help": "The intrinsic dimension for OFT fine-tuning."},
    )
    oft_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of target modules to apply OFT. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    create_new_adapter: bool = field(
        default=False,
        metadata={"help": "Whether or not to create a new adapter with randomly initialized weight."},
    )


@dataclass
class RLHFArguments:
    r"""Arguments pertaining to the PPO, DPO and KTO training."""

    pref_beta: float = field(
        default=0.1,
        metadata={"help": "The beta parameter in the preference loss."},
    )
    pref_ftx: float = field(
        default=0.0,
        metadata={"help": "The supervised fine-tuning loss coefficient in DPO training."},
    )
    pref_bco_weight: float = field(
        default=0.0,
        metadata={"help": "The Binary Classifier Optimization coefficient in DPO training."},
    )
    pref_loss: Literal["sigmoid", "hinge", "ipo", "kto_pair", "orpo", "simpo"] = field(
        default="sigmoid",
        metadata={"help": "The type of DPO loss to use."},
    )
    dpo_label_smoothing: float = field(
        default=0.0,
        metadata={"help": "The robust DPO label smoothing parameter in cDPO that should be between 0 and 0.5."},
    )
    kto_chosen_weight: float = field(
        default=1.0,
        metadata={"help": "The weight factor of the desirable losses in KTO training."},
    )
    kto_rejected_weight: float = field(
        default=1.0,
        metadata={"help": "The weight factor of the undesirable losses in KTO training."},
    )
    simpo_gamma: float = field(
        default=0.5,
        metadata={"help": "The target reward margin term in SimPO loss."},
    )
    ppo_buffer_size: int = field(
        default=1,
        metadata={"help": "The number of mini-batches to make experience buffer in a PPO optimization step."},
    )
    ppo_epochs: int = field(
        default=4,
        metadata={"help": "The number of epochs to perform in a PPO optimization step."},
    )
    ppo_score_norm: bool = field(
        default=False,
        metadata={"help": "Use score normalization in PPO training."},
    )
    ppo_target: float = field(
        default=6.0,
        metadata={"help": "Target KL value for adaptive KL control in PPO training."},
    )
    ppo_whiten_rewards: bool = field(
        default=False,
        metadata={"help": "Whiten the rewards before compute advantages in PPO training."},
    )
    ref_model: str | None = field(
        default=None,
        metadata={"help": "Path to the reference model used for the PPO or DPO training."},
    )
    ref_model_adapters: str | None = field(
        default=None,
        metadata={"help": "Path to the adapters of the reference model."},
    )
    ref_model_quantization_bit: int | None = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reference model."},
    )
    reward_model: str | None = field(
        default=None,
        metadata={"help": "Path to the reward model used for the PPO training."},
    )
    reward_model_adapters: str | None = field(
        default=None,
        metadata={"help": "Path to the adapters of the reward model."},
    )
    reward_model_quantization_bit: int | None = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reward model."},
    )
    reward_model_type: Literal["lora", "full", "api"] = field(
        default="lora",
        metadata={"help": "The type of the reward model in PPO training. Lora model only supports lora training."},
    )
    ld_alpha: float | None = field(
        default=None,
        metadata={
            "help": (
                "Alpha parameter from the LD-DPO paper, which controls the weighting of"
                " the verbose token log-probabilities in responses."
            )
        },
    )


@dataclass
class GaloreArguments:
    r"""Arguments pertaining to the GaLore algorithm."""

    use_galore: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the gradient low-Rank projection (GaLore)."},
    )
    galore_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of modules to apply GaLore. Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    galore_rank: int = field(
        default=16,
        metadata={"help": "The rank of GaLore gradients."},
    )
    galore_update_interval: int = field(
        default=200,
        metadata={"help": "Number of steps to update the GaLore projection."},
    )
    galore_scale: float = field(
        default=2.0,
        metadata={"help": "GaLore scaling coefficient."},
    )
    galore_proj_type: Literal["std", "reverse_std", "right", "left", "full"] = field(
        default="std",
        metadata={"help": "Type of GaLore projection."},
    )
    galore_layerwise: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable layer-wise update to further save memory."},
    )


@dataclass
class ApolloArguments:
    r"""Arguments pertaining to the APOLLO algorithm."""

    use_apollo: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the APOLLO optimizer."},
    )
    apollo_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of modules to apply APOLLO. Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    apollo_rank: int = field(
        default=16,
        metadata={"help": "The rank of APOLLO gradients."},
    )
    apollo_update_interval: int = field(
        default=200,
        metadata={"help": "Number of steps to update the APOLLO projection."},
    )
    apollo_scale: float = field(
        default=32.0,
        metadata={"help": "APOLLO scaling coefficient."},
    )
    apollo_proj: Literal["svd", "random"] = field(
        default="random",
        metadata={"help": "Type of APOLLO low-rank projection algorithm (svd or random)."},
    )
    apollo_proj_type: Literal["std", "right", "left"] = field(
        default="std",
        metadata={"help": "Type of APOLLO projection."},
    )
    apollo_scale_type: Literal["channel", "tensor"] = field(
        default="channel",
        metadata={"help": "Type of APOLLO scaling (channel or tensor)."},
    )
    apollo_layerwise: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable layer-wise update to further save memory."},
    )
    apollo_scale_front: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the norm-growth limiter in front of gradient scaling."},
    )


@dataclass
class BAdamArgument:
    r"""Arguments pertaining to the BAdam optimizer."""

    use_badam: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the BAdam optimizer."},
    )
    badam_mode: Literal["layer", "ratio"] = field(
        default="layer",
        metadata={"help": "Whether to use layer-wise or ratio-wise BAdam optimizer."},
    )
    badam_start_block: int | None = field(
        default=None,
        metadata={"help": "The starting block index for layer-wise BAdam."},
    )
    badam_switch_mode: Literal["ascending", "descending", "random", "fixed"] | None = field(
        default="ascending",
        metadata={"help": "the strategy of picking block to update for layer-wise BAdam."},
    )
    badam_switch_interval: int | None = field(
        default=50,
        metadata={
            "help": "Number of steps to update the block for layer-wise BAdam. Use -1 to disable the block update."
        },
    )
    badam_update_ratio: float = field(
        default=0.05,
        metadata={"help": "The ratio of the update for ratio-wise BAdam."},
    )
    badam_mask_mode: Literal["adjacent", "scatter"] = field(
        default="adjacent",
        metadata={
            "help": (
                "The mode of the mask for BAdam optimizer. "
                "`adjacent` means that the trainable parameters are adjacent to each other, "
                "`scatter` means that trainable parameters are randomly chosen from the weight."
            )
        },
    )
    badam_verbose: int = field(
        default=0,
        metadata={
            "help": (
                "The verbosity level of BAdam optimizer. "
                "0 for no print, 1 for print the block prefix, 2 for print trainable parameters."
            )
        },
    )


@dataclass
class SwanLabArguments:
    use_swanlab: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the SwanLab (an experiment tracking and visualization tool)."},
    )
    swanlab_project: str | None = field(
        default="llamafactory",
        metadata={"help": "The project name in SwanLab."},
    )
    swanlab_workspace: str | None = field(
        default=None,
        metadata={"help": "The workspace name in SwanLab."},
    )
    swanlab_run_name: str | None = field(
        default=None,
        metadata={"help": "The experiment name in SwanLab."},
    )
    swanlab_mode: Literal["cloud", "local"] = field(
        default="cloud",
        metadata={"help": "The mode of SwanLab."},
    )
    swanlab_api_key: str | None = field(
        default=None,
        metadata={"help": "The API key for SwanLab."},
    )
    swanlab_logdir: str | None = field(
        default=None,
        metadata={"help": "The log directory for SwanLab."},
    )
    swanlab_lark_webhook_url: str | None = field(
        default=None,
        metadata={"help": "The Lark(飞书) webhook URL for SwanLab."},
    )
    swanlab_lark_secret: str | None = field(
        default=None,
        metadata={"help": "The Lark(飞书) secret for SwanLab."},
    )


@dataclass
class FinetuningArguments(
    SwanLabArguments,
    BAdamArgument,
    ApolloArguments,
    GaloreArguments,
    RLHFArguments,
    LoraArguments,
    OFTArguments,
    FreezeArguments,
):
    r"""Arguments pertaining to which techniques we are going to fine-tuning with."""

    pure_bf16: bool = field(
        default=False,
        metadata={"help": "Whether or not to train model in purely bf16 precision (without AMP)."},
    )
    stage: Literal["pt", "sft", "rm", "ppo", "dpo", "kto"] = field(
        default="sft",
        metadata={"help": "Which stage will be performed in training."},
    )
    finetuning_type: Literal["lora", "oft", "freeze", "full"] = field(
        default="lora",
        metadata={"help": "Which fine-tuning method to use."},
    )
    use_llama_pro: bool = field(
        default=False,
        metadata={"help": "Whether or not to make only the parameters in the expanded blocks trainable."},
    )
    use_adam_mini: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the Adam-mini optimizer."},
    )
    use_mca: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to use MCA (Megatron Core Adapter) training. "
                "Controlled by USE_MCA environment variable."
            )
        },
    )
    use_megatron_bridge: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to use Megatron Bridge training backend. "
                "Controlled by USE_MEGATRON_BRIDGE environment variable."
            )
        },
    )
    megatron_bridge_args: Any = field(
        default=None,
        init=False,
        repr=False,
        metadata={"help": "Megatron Bridge specific arguments, set when USE_MEGATRON_BRIDGE=1."},
    )
    use_hyper_parallel: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to use HyperParallel distributed training backend (FSDP/TP). "
                "Only supported for the 'pt' and 'sft' stages with full fine-tuning."
            )
        },
    )
    hyper_parallel_args: str | None = field(
        default=None,
        metadata={
            "help": (
                "Path to a JSON file containing HyperParallel strategy arguments "
                "(e.g., tp_size, param_dtype). Used when use_hyper_parallel=True."
            )
        },
    )
    hyper_parallel_cp_size: int = field(
        default=1,
        metadata={"help": "Context parallel size used when `use_hyper_parallel=True`."},
    )
    use_muon: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the Muon optimizer."},
    )
    use_dft_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use the DFT loss."},
    )
    use_asft_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use the ASFT loss."},
    )
    asft_alpha: float = field(
        default=0.1,
        metadata={"help": "The alpha parameter for ASFT loss to control the power of adaptive weight."},
    )
    use_eaft_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use the EAFT loss."},
    )
    eaft_alpha: float = field(
        default=1.0,
        metadata={"help": "The alpha parameter for EAFT loss to control the power of adaptive weight."},
    )
    freeze_vision_tower: bool = field(
        default=True,
        metadata={"help": "Whether or not to freeze the vision tower in MLLM training."},
    )
    freeze_multi_modal_projector: bool = field(
        default=True,
        metadata={"help": "Whether or not to freeze the multi modal projector in MLLM training."},
    )
    freeze_language_model: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the language model in MLLM training."},
    )
    compute_accuracy: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute the token-level accuracy at evaluation."},
    )
    disable_shuffling: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable the shuffling of the training set."},
    )
    early_stopping_steps: int | None = field(
        default=None,
        metadata={"help": "Number of steps to stop training if the `metric_for_best_model` does not improve."},
    )
    plot_loss: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the training loss curves."},
    )
    include_effective_tokens_per_second: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute effective tokens per second."},
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.freeze_trainable_modules: list[str] = split_arg(self.freeze_trainable_modules)
        self.freeze_extra_modules: list[str] | None = split_arg(self.freeze_extra_modules)
        self.lora_alpha: int = self.lora_alpha or self.lora_rank * 2
        self.lora_target: list[str] = split_arg(self.lora_target)
        self.oft_target: list[str] = split_arg(self.oft_target)
        self.additional_target: list[str] | None = split_arg(self.additional_target)
        self.galore_target: list[str] = split_arg(self.galore_target)
        self.apollo_target: list[str] = split_arg(self.apollo_target)
        self.use_ref_model = self.stage == "dpo" and self.pref_loss not in ["orpo", "simpo"]

        assert self.finetuning_type in ["lora", "oft", "freeze", "full"], "Invalid fine-tuning method."
        assert self.ref_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."
        assert self.reward_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."
        assert self.hyper_parallel_cp_size > 0, "`hyper_parallel_cp_size` must be greater than 0."

        if self.stage == "ppo" and self.reward_model is None:
            raise ValueError("`reward_model` is necessary for PPO training.")

        if self.stage == "ppo" and self.reward_model_type == "lora" and self.finetuning_type != "lora":
            raise ValueError("`reward_model_type` cannot be lora for Freeze/Full PPO training.")

        if self.stage == "ppo" and self.reward_model_type == "oft" and self.finetuning_type != "oft":
            raise ValueError("`reward_model_type` cannot be oft for Freeze/Full PPO training.")

        if self.stage == "dpo" and self.pref_loss != "sigmoid" and self.dpo_label_smoothing > 1e-6:
            raise ValueError("`dpo_label_smoothing` is only valid for sigmoid loss function.")

        if self.use_llama_pro and self.finetuning_type == "full":
            raise ValueError("`use_llama_pro` is only valid for Freeze or LoRA training.")

        if self.finetuning_type == "lora" and (self.use_galore or self.use_apollo or self.use_badam):
            raise ValueError("Cannot use LoRA with GaLore, APOLLO or BAdam together.")

        if int(self.use_galore) + int(self.use_apollo) + (self.use_badam) > 1:
            raise ValueError("Cannot use GaLore, APOLLO or BAdam together.")

        if self.pissa_init and (self.stage in ["ppo", "kto"] or self.use_ref_model):
            raise ValueError("Cannot use PiSSA for current training stage.")

        if self.finetuning_type != "lora":
            if self.loraplus_lr_ratio is not None:
                raise ValueError("`loraplus_lr_ratio` is only valid for LoRA training.")

            if self.use_rslora:
                raise ValueError("`use_rslora` is only valid for LoRA training.")

            if self.use_dora:
                raise ValueError("`use_dora` is only valid for LoRA training.")

            if self.pissa_init:
                raise ValueError("`pissa_init` is only valid for LoRA training.")

    def to_dict(self) -> dict[str, Any]:
        args = asdict(self)
        args = {k: f"<{k.upper()}>" if k.endswith("api_key") else v for k, v in args.items()}
        return args
