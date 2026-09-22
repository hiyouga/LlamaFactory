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

import fnmatch
import csv
import io
import subprocess
import json
import os
import signal
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Optional

import torch
import transformers
from peft import PeftModel
from transformers import PreTrainedModel, ProcessorMixin, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, has_length
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from typing_extensions import override

from ..extras import logging
from ..extras.constants import TRAINER_LOG, V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME
from ..extras.misc import get_peak_memory, is_env_enabled, is_torch_cuda_available, is_torch_npu_available, use_ray
from ..extras.packages import is_safetensors_available


if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.torch import save_file


if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)

_XPU_TELEMETRY_FIELDS: tuple[tuple[tuple[str, ...], str], ...] = (
    # Compute utilization: use metric 31 (Compute engine group utilization) which
    # measures compute engines only, providing the most accurate view of AI/ML workload.
    (("Compute engine group utilization (%)",), "xpu_compute_util_pct"),
    (("GPU Memory Utilization (%)",), "xpu_mem_bandwidth_pct"),
    (("GPU Memory Used (MiB)",), "xpu_mem_in_use_mib"),
)


def _sample_xpu_device_metrics(device_id: int = 0) -> dict[str, float]:
    r"""Sample per-device XPU telemetry via the ``xpu-smi dump`` interface.

    ``xpu-smi dump`` emits CSV (the ``-j`` flag is only honored by ``stats``/``discovery``),
    so we parse the header row to map column names onto our metric keys. Returns an
    empty dict if ``xpu-smi`` is missing, times out, or returns malformed output.
    A single warning is logged per failure mode.

    Metric IDs requested: 5=Mem Util, 18=Mem Used (MiB), 31=Compute Engine Group Util.
    Metric 31 provides the most accurate compute utilization for AI/ML workloads.
    """
    metrics: dict[str, float] = {}
    cmd = ["sudo", "xpu-smi", "dump", "-d", str(device_id), "-m", "5,18,31", "-n", "1"]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
    except FileNotFoundError:
        logger.warning_rank0_once(
            "xpu-smi binary not found on PATH; XPU telemetry will be skipped. "
            "Install Intel XPU Manager or unset RECORD_XPU to silence this warning."
        )
        return metrics
    except subprocess.TimeoutExpired:
        logger.warning_rank0_once("xpu-smi query timed out; XPU telemetry will be skipped for this step.")
        return metrics
    except Exception as err:
        logger.warning_rank0_once(f"xpu-smi query failed ({err!r}); XPU telemetry will be skipped.")
        return metrics

    if proc.returncode != 0:
        logger.warning_rank0_once(
            f"xpu-smi exited with code {proc.returncode}; XPU telemetry will be skipped. "
            f"stderr: {proc.stderr.strip()[:200]}"
        )
        return metrics

    rows = list(csv.reader(io.StringIO(proc.stdout)))
    # Expect a header line plus at least one data row.
    data_rows = [r for r in rows if r and r[0].strip()]
    if len(data_rows) < 2:
        logger.warning_rank0_once("xpu-smi returned no data rows; XPU telemetry will be skipped.")
        return metrics

    header = [col.strip() for col in data_rows[0]]
    values = [col.strip() for col in data_rows[1]]
    row = dict(zip(header, values))

    for src_keys, dst_key in _XPU_TELEMETRY_FIELDS:
        for src_key in src_keys:
            raw = row.get(src_key)
            if raw is None or raw == "" or raw.lower() == "n/a":
                continue
            try:
                metrics[dst_key] = round(float(raw), 2)
                break
            except (ValueError, TypeError):
                continue

    return metrics


def _sample_host_resource_usage() -> dict[str, float]:
    r"""Sample host-side CPU and RAM utilization via ``psutil``.

    Returns an empty dict when ``psutil`` is not installed (warned once). RAM metrics
    are always included when available. The first call to ``psutil.cpu_percent(interval=None)``
    has no prior sample to diff against and would report 0.0, so ``host_cpu_busy_pct``
    is omitted on the first invocation; the counter is primed and reported from the
    second invocation onward.
    """
    try:
        import psutil
    except ImportError:
        logger.warning_rank0_once(
            "psutil is not installed; host CPU/RAM telemetry will be skipped. "
            "Run `pip install psutil` or unset RECORD_CPU to silence this warning."
        )
        return {}

    try:
        first_call = not getattr(_sample_host_resource_usage, "_primed", False)
        cpu_pct = psutil.cpu_percent(interval=None)
        _sample_host_resource_usage._primed = True

        vmem = psutil.virtual_memory()
        result: dict[str, float] = {
            "host_ram_in_use_gib": round(vmem.used / (1024**3), 2),
            "host_ram_full_pct": round(vmem.percent, 2),
        }

        if not first_call:
            result["host_cpu_busy_pct"] = round(cpu_pct, 2)
        return result
    except Exception as err:
        logger.warning_rank0_once(f"psutil host query failed ({err!r}); host telemetry will be skipped.")
        return {}

def fix_valuehead_checkpoint(
    model: "AutoModelForCausalLMWithValueHead", output_dir: str, safe_serialization: bool
) -> None:
    r"""Fix the valuehead checkpoint files.

    The model is already unwrapped.

    There are three cases:
    1. full tuning without ds_zero3: state_dict = {"model.layers.*": ..., "v_head.summary.*": ...}
    2. lora tuning without ds_zero3: state_dict = {"v_head.summary.*": ...}
    3. under deepspeed zero3: state_dict = {"pretrained_model.model.layers.*": ..., "v_head.summary.*": ...}

    We assume `stage3_gather_16bit_weights_on_model_save=true`.
    """
    if not isinstance(model.pretrained_model, (PreTrainedModel, PeftModel)):
        return

    if safe_serialization:
        path_to_checkpoint = os.path.join(output_dir, SAFE_WEIGHTS_NAME)
        with safe_open(path_to_checkpoint, framework="pt", device="cpu") as f:
            state_dict: dict[str, torch.Tensor] = {key: f.get_tensor(key).clone() for key in f.keys()}
    else:
        path_to_checkpoint = os.path.join(output_dir, WEIGHTS_NAME)
        state_dict: dict[str, torch.Tensor] = torch.load(path_to_checkpoint, map_location="cpu", weights_only=True)

    os.remove(path_to_checkpoint)
    decoder_state_dict, v_head_state_dict = {}, {}
    for name, param in state_dict.items():
        if name.startswith("v_head."):
            v_head_state_dict[name] = param
        else:
            decoder_state_dict[name.replace("pretrained_model.", "", 1)] = param

    model.pretrained_model.save_pretrained(
        output_dir, state_dict=decoder_state_dict or None, safe_serialization=safe_serialization
    )

    if safe_serialization:
        save_file(v_head_state_dict, os.path.join(output_dir, V_HEAD_SAFE_WEIGHTS_NAME), metadata={"format": "pt"})
    else:
        torch.save(v_head_state_dict, os.path.join(output_dir, V_HEAD_WEIGHTS_NAME))

    logger.info_rank0(f"Value head model saved at: {output_dir}")


class FixValueHeadModelCallback(TrainerCallback):
    r"""A callback for fixing the checkpoint for valuehead models."""

    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            fix_valuehead_checkpoint(
                model=kwargs.pop("model"),
                output_dir=output_dir,
                safe_serialization=getattr(args, "save_safetensors", True),
            )


class SaveProcessorCallback(TrainerCallback):
    r"""A callback for saving the processor."""

    def __init__(self, processor: "ProcessorMixin") -> None:
        self.processor = processor

    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            self.processor.save_pretrained(output_dir)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            self.processor.save_pretrained(args.output_dir)


class PissaConvertCallback(TrainerCallback):
    r"""A callback for converting the PiSSA adapter to a normal one."""

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            model = kwargs.pop("model")
            pissa_init_dir = os.path.join(args.output_dir, "pissa_init")
            logger.info_rank0(f"Initial PiSSA adapter will be saved at: {pissa_init_dir}.")
            if isinstance(model, PeftModel):
                init_lora_weights = getattr(model.peft_config["default"], "init_lora_weights")
                setattr(model.peft_config["default"], "init_lora_weights", True)
                model.save_pretrained(pissa_init_dir, safe_serialization=getattr(args, "save_safetensors", True))
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            model = kwargs.pop("model")
            pissa_init_dir = os.path.join(args.output_dir, "pissa_init")
            pissa_backup_dir = os.path.join(args.output_dir, "pissa_backup")
            pissa_convert_dir = os.path.join(args.output_dir, "pissa_converted")
            logger.info_rank0(f"Converted PiSSA adapter will be saved at: {pissa_convert_dir}.")
            # 1. save a pissa backup with init_lora_weights: True
            # 2. save a converted lora with init_lora_weights: pissa
            # 3. load the pissa backup with init_lora_weights: True
            # 4. delete the initial adapter and change init_lora_weights to pissa
            if isinstance(model, PeftModel):
                init_lora_weights = getattr(model.peft_config["default"], "init_lora_weights")
                setattr(model.peft_config["default"], "init_lora_weights", True)
                model.save_pretrained(pissa_backup_dir, safe_serialization=getattr(args, "save_safetensors", True))
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)
                model.save_pretrained(
                    pissa_convert_dir,
                    safe_serialization=getattr(args, "save_safetensors", True),
                    path_initial_model_for_weight_conversion=pissa_init_dir,
                )
                model.load_adapter(pissa_backup_dir, "default", is_trainable=True)
                model.set_adapter("default")
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)


class LogCallback(TrainerCallback):
    r"""A callback for logging training and evaluation status."""

    def __init__(self) -> None:
        # Progress
        self.start_time = 0
        self.cur_steps = 0
        self.max_steps = 0
        self.elapsed_time = ""
        self.remaining_time = ""
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        # Status
        self.aborted = False
        self.do_train = False
        # Web UI
        self.webui_mode = is_env_enabled("LLAMABOARD_ENABLED")
        if self.webui_mode and not use_ray():
            signal.signal(signal.SIGABRT, self._set_abort)
            self.logger_handler = logging.LoggerHandler(os.getenv("LLAMABOARD_WORKDIR"))
            logging.add_handler(self.logger_handler)
            transformers.logging.add_handler(self.logger_handler)

    def _set_abort(self, signum, frame) -> None:
        self.aborted = True

    def _reset(self, max_steps: int = 0) -> None:
        self.start_time = time.time()
        self.cur_steps = 0
        self.max_steps = max_steps
        self.elapsed_time = ""
        self.remaining_time = ""

    def _timing(self, cur_steps: int) -> None:
        cur_time = time.time()
        elapsed_time = cur_time - self.start_time
        avg_time_per_step = elapsed_time / cur_steps if cur_steps != 0 else 0
        remaining_time = (self.max_steps - cur_steps) * avg_time_per_step
        self.cur_steps = cur_steps
        self.elapsed_time = str(timedelta(seconds=int(elapsed_time)))
        self.remaining_time = str(timedelta(seconds=int(remaining_time)))

    def _write_log(self, output_dir: str, logs: dict[str, Any]) -> None:
        with open(os.path.join(output_dir, TRAINER_LOG), "a", encoding="utf-8") as f:
            f.write(json.dumps(logs) + "\n")

    def _sample_and_write_log(self, output_dir: str, logs: dict[str, Any], device_id: int) -> None:
        r"""Sample device/host metrics on the worker thread and append to the log file.

        ``_sample_xpu_device_metrics`` shells out to ``xpu-smi`` (subprocess + IO) and
        ``_sample_host_resource_usage`` calls ``psutil``; both are blocking. Running them
        here keeps the main training thread free, which matters when ``logging_steps`` is
        small. Metrics are not used by the Web UI's inline log line, so deferring them
        only affects the persisted ``trainer_log.jsonl`` content.
        """
        if is_env_enabled("RECORD_XPU"):
            logs.update(_sample_xpu_device_metrics(device_id))
        if is_env_enabled("RECORD_CPU"):
            logs.update(_sample_host_resource_usage())
        logs = {k: v for k, v in logs.items() if v is not None}
        self._write_log(output_dir, logs)

    def _create_thread_pool(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.thread_pool = ThreadPoolExecutor(max_workers=1)

    def _close_thread_pool(self) -> None:
        if self.thread_pool is not None:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None

    @override
    def on_init_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if (
            args.should_save
            and os.path.exists(os.path.join(args.output_dir, TRAINER_LOG))
            and getattr(args, "overwrite_output_dir", False)
        ):
            logger.warning_rank0_once("Previous trainer log in this folder will be deleted.")
            os.remove(os.path.join(args.output_dir, TRAINER_LOG))

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            self.do_train = True
            self._reset(max_steps=state.max_steps)
            self._create_thread_pool(output_dir=args.output_dir)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        self._close_thread_pool()

    @override
    def on_substep_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    @override
    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    @override
    def on_evaluate(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_predict(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_log(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not args.should_save:
            return

        self._timing(cur_steps=state.global_step)
        logs = dict(
            current_steps=self.cur_steps,
            total_steps=self.max_steps,
            loss=state.log_history[-1].get("loss"),
            eval_loss=state.log_history[-1].get("eval_loss"),
            predict_loss=state.log_history[-1].get("predict_loss"),
            reward=state.log_history[-1].get("reward"),
            accuracy=state.log_history[-1].get("rewards/accuracies"),
            lr=state.log_history[-1].get("learning_rate"),
            epoch=state.log_history[-1].get("epoch"),
            percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
            elapsed_time=self.elapsed_time,
            remaining_time=self.remaining_time,
        )
        if state.num_input_tokens_seen:
            logs["throughput"] = round(state.num_input_tokens_seen / (time.time() - self.start_time), 2)
            logs["total_tokens"] = state.num_input_tokens_seen

        if is_env_enabled("RECORD_VRAM"):
            vram_allocated, vram_reserved = get_peak_memory()
            logs["vram_allocated"] = round(vram_allocated / (1024**3), 2)
            logs["vram_reserved"] = round(vram_reserved / (1024**3), 2)

        # Resolve the active XPU device on the main thread (cheap torch query) so the
        # worker can sample without touching torch state. Actual xpu-smi / psutil
        # sampling is offloaded below to avoid blocking training on logging steps.
        xpu_device_id = 0
        if is_env_enabled("RECORD_XPU") and hasattr(torch, "xpu") and torch.xpu.is_available():
            try:
                xpu_device_id = torch.xpu.current_device()
            except Exception:
                xpu_device_id = 0

        if self.webui_mode and all(key in logs for key in ("loss", "lr", "epoch")):
            log_str = f"'loss': {logs['loss']:.4f}, 'learning_rate': {logs['lr']:2.4e}, 'epoch': {logs['epoch']:.2f}"
            for extra_key in ("reward", "accuracy", "throughput"):
                if logs.get(extra_key):
                    log_str += f", '{extra_key}': {logs[extra_key]:.2f}"

            logger.info_rank0("{" + log_str + "}")

        if self.thread_pool is not None:
            # Offload XPU/host sampling (xpu-smi subprocess + psutil) plus the file write
            # onto the single-worker pool so they cannot stall the training step.
            self.thread_pool.submit(self._sample_and_write_log, args.output_dir, logs, xpu_device_id)

    @override
    def on_prediction_step(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ):
        if self.do_train:
            return

        if self.aborted:
            sys.exit(0)

        if not args.should_save:
            return

        eval_dataloader = kwargs.pop("eval_dataloader", None)
        if has_length(eval_dataloader):
            if self.max_steps == 0:
                self._reset(max_steps=len(eval_dataloader))
                self._create_thread_pool(output_dir=args.output_dir)

            self._timing(cur_steps=self.cur_steps + 1)
            if self.cur_steps % 5 == 0 and self.thread_pool is not None:
                logs = dict(
                    current_steps=self.cur_steps,
                    total_steps=self.max_steps,
                    percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
                    elapsed_time=self.elapsed_time,
                    remaining_time=self.remaining_time,
                )
                self.thread_pool.submit(self._write_log, args.output_dir, logs)


class TorchProfilerCallback(TrainerCallback):
    r"""A callback for collecting torch.profiler traces during training.

    Activated by setting ``enable_torch_profiler: true`` in the YAML config.

    Configuration fields (in YAML):
      profiler_output_dir     – where to write traces (default: <output_dir>/profiler)
      profiler_wait_steps     – steps to skip at start of each cycle (default: 1)
      profiler_warmup_steps   – profiler warm-up steps per cycle       (default: 1)
      profiler_active_steps   – steps to record per cycle              (default: 1)
      profiler_repeat         – number of cycles; 0 = forever          (default: 1)
      profiler_record_shapes  – record tensor shapes (default: true)
      profiler_profile_memory – profile memory usage (default: true)
      profiler_with_stack     – record stack traces (default: true)

    Trace files (one per rank, Chrome / TensorBoard JSON format) are written to
    ``<profiler_output_dir>/rank_<N>/``.
    """

    def __init__(self, training_args: "TrainingArguments") -> None:
        self.profiler = None
        self.profiler_args = training_args

    @staticmethod
    def _get_rank() -> int:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    @override
    def on_train_begin(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ) -> None:
        if self.profiler is not None:
            self.profiler.stop()
            self.profiler = None

        pa = self.profiler_args
        output_dir = pa.profiler_output_dir or os.path.join(args.output_dir, "profiler")
        rank = self._get_rank()
        trace_dir = os.path.join(output_dir, f"rank_{rank}")
        os.makedirs(trace_dir, exist_ok=True)

        activities = [torch.profiler.ProfilerActivity.CPU]
        try:
            if is_torch_cuda_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            if is_torch_npu_available():
                activities.append(torch.profiler.ProfilerActivity.NPU)
        except Exception:
            pass

        self.profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=pa.profiler_wait_steps,
                warmup=pa.profiler_warmup_steps,
                active=pa.profiler_active_steps,
                repeat=pa.profiler_repeat,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
            record_shapes=pa.profiler_record_shapes,
            profile_memory=pa.profiler_profile_memory,
            with_stack=pa.profiler_with_stack,
        )
        self.profiler.start()
        logger.info_rank0(
            f"TorchProfiler started — schedule: wait={pa.profiler_wait_steps}, warmup={pa.profiler_warmup_steps}, "
            f"active={pa.profiler_active_steps}, repeat={pa.profiler_repeat}. Traces → {output_dir}"
        )

    @override
    def on_step_end(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ) -> None:
        if self.profiler is not None:
            self.profiler.step()

    @override
    def on_train_end(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ) -> None:
        if self.profiler is not None:
            self.profiler.stop()
            self.profiler = None
            logger.info_rank0("TorchProfiler stopped.")


class ReporterCallback(TrainerCallback):
    r"""A callback for reporting training status to external logger."""

    def __init__(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
    ) -> None:
        self.model_args = model_args
        self.data_args = data_args
        self.finetuning_args = finetuning_args
        self.generating_args = generating_args
        os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT", "llamafactory")

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not state.is_world_process_zero:
            return

        if "wandb" in args.report_to:
            import wandb

            wandb.config.update(
                {
                    "model_args": self.model_args.to_dict(),
                    "data_args": self.data_args.to_dict(),
                    "finetuning_args": self.finetuning_args.to_dict(),
                    "generating_args": self.generating_args.to_dict(),
                }
            )

        if "trackio" in args.report_to:
            import trackio

            trackio.config.update(
                {
                    "model_args": self.model_args.to_dict(),
                    "data_args": self.data_args.to_dict(),
                    "finetuning_args": self.finetuning_args.to_dict(),
                    "generating_args": self.generating_args.to_dict(),
                }
            )

        if self.finetuning_args.use_swanlab:
            import swanlab  # type: ignore

            swanlab.config.update(
                {
                    "model_args": self.model_args.to_dict(),
                    "data_args": self.data_args.to_dict(),
                    "finetuning_args": self.finetuning_args.to_dict(),
                    "generating_args": self.generating_args.to_dict(),
                }
            )


class ModuleProfilerCallback(TrainerCallback):
    r"""Profile forward/backward time of specified modules using accelerator events.

    Hooks are registered on modules matching the user-provided name patterns.
    Timing statistics are logged at each trainer logging step.

    Usage in YAML config:
        profile_modules: "*.layers.0.self_attn,*.layers.0.mlp"

    Supports fnmatch wildcards:
        profile_modules: "*.layers.*.self_attn,*.layers.*.mlp.experts"
    """

    @staticmethod
    def _get_accelerator():
        """Detect available accelerator and return (event_factory, synchronize_fn)."""
        if is_torch_cuda_available():
            return torch.cuda.Event, torch.cuda.synchronize
        if is_torch_npu_available():
            return torch.npu.Event, torch.npu.synchronize
        return None, None

    def __init__(self, profile_modules: str) -> None:
        self.patterns = [p.strip() for p in profile_modules.split(",") if p.strip()]
        self._create_event, self._synchronize = self._get_accelerator()
        self._handles: list[Any] = []
        self._forward_times: dict[str, list[float]] = defaultdict(list)
        self._backward_times: dict[str, list[float]] = defaultdict(list)
        self._pending_forward: dict[str, tuple] = {}
        self._pending_backward: dict[str, tuple] = {}

    @property
    def enabled(self) -> bool:
        return self._create_event is not None

    def _match(self, name: str) -> bool:
        return any(fnmatch.fnmatch(name, pat) for pat in self.patterns)

    def _make_forward_pre_hook(self, name: str):
        def hook(module, input):
            start = self._create_event(enable_timing=True)
            end = self._create_event(enable_timing=True)
            start.record()
            self._pending_forward[name] = (start, end)

        return hook

    def _make_forward_hook(self, name: str):
        def hook(module, input, output):
            pair = self._pending_forward.get(name)
            if pair is not None:
                pair[1].record()

        return hook

    def _make_backward_pre_hook(self, name: str):
        def hook(module, grad_output):
            start = self._create_event(enable_timing=True)
            end = self._create_event(enable_timing=True)
            start.record()
            self._pending_backward[name] = (start, end)

        return hook

    def _make_backward_hook(self, name: str):
        def hook(module, grad_input, grad_output):
            pair = self._pending_backward.get(name)
            if pair is not None:
                pair[1].record()

        return hook

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.enabled:
            logger.warning_rank0("ModuleProfiler: no supported accelerator (CUDA/NPU) found, profiling disabled.")
            return

        model = kwargs.get("model")
        if model is None:
            return

        matched = []
        for name, module in model.named_modules():
            if not name or not self._match(name):
                continue
            self._handles.append(module.register_forward_pre_hook(self._make_forward_pre_hook(name)))
            self._handles.append(module.register_forward_hook(self._make_forward_hook(name)))
            self._handles.append(module.register_full_backward_pre_hook(self._make_backward_pre_hook(name)))
            self._handles.append(module.register_full_backward_hook(self._make_backward_hook(name)))
            matched.append(name)

        if matched:
            logger.info_rank0(
                f"ModuleProfiler: registered hooks on {len(matched)} modules: {matched[:5]}"
                + (f" ... (+{len(matched) - 5} more)" if len(matched) > 5 else "")
            )
        else:
            logger.warning_rank0(f"ModuleProfiler: no modules matched patterns {self.patterns}")

    @override
    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.enabled:
            return

        self._synchronize()

        for name, (start, end) in self._pending_forward.items():
            self._forward_times[name].append(start.elapsed_time(end))
        self._pending_forward.clear()

        for name, (start, end) in self._pending_backward.items():
            self._backward_times[name].append(start.elapsed_time(end))
        self._pending_backward.clear()

    @override
    def on_log(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self._forward_times and not self._backward_times:
            return

        lines = ["[ModuleProfiler] Timing (ms):"]
        all_names = sorted(set(list(self._forward_times.keys()) + list(self._backward_times.keys())))
        for name in all_names:
            fwd = self._forward_times.get(name, [])
            bwd = self._backward_times.get(name, [])
            fwd_mean = sum(fwd) / len(fwd) if fwd else 0.0
            bwd_mean = sum(bwd) / len(bwd) if bwd else 0.0
            lines.append(f"  {name}: fwd={fwd_mean:.3f}, bwd={bwd_mean:.3f}, total={fwd_mean + bwd_mean:.3f}")

        logger.info_rank0("\n".join(lines))
        self._forward_times.clear()
        self._backward_times.clear()

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
