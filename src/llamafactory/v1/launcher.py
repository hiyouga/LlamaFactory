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

import os
import subprocess
import sys
from copy import deepcopy


USAGE = (
    "-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "|   llamafactory-cli sft -h: train models                            |\n"
    + "|   llamafactory-cli version: show version info                      |\n"
    + "| Hint: You can use `lmf` as a shortcut for `llamafactory-cli`.      |\n"
    + "-" * 70
)

_DIST_TRAIN_COMMANDS = ("train", "sft", "dpo", "rm")


def launch():
    from .accelerator.helper import get_device_count
    from .utils.env import find_available_port, is_env_enabled, use_kt, use_ray
    from .utils.logging import get_logger

    logger = get_logger(__name__)

    # NOTE:
    # `llamafactory-cli <command> ...` enters here first.
    # We may re-launch via `torchrun` for distributed training. In that case we must
    # forward `<command>` as argv[1] to the re-executed script, otherwise the script
    # will misinterpret the first user argument (e.g. yaml config) as the command.
    command = sys.argv.pop(1) if len(sys.argv) > 1 else "help"

    if command in _DIST_TRAIN_COMMANDS and (
        is_env_enabled("FORCE_TORCHRUN") or (get_device_count() > 1 and not use_ray() and not use_kt())
    ):
        nnodes = os.getenv("NNODES", "1")
        node_rank = os.getenv("NODE_RANK", "0")
        nproc_per_node = os.getenv("NPROC_PER_NODE", str(get_device_count()))
        master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
        master_port = os.getenv("MASTER_PORT", str(find_available_port()))
        logger.info_rank0(f"Initializing {nproc_per_node} distributed tasks at: {master_addr}:{master_port}")
        if int(nnodes) > 1:
            logger.info_rank0(f"Multi-node training enabled: num nodes: {nnodes}, node rank: {node_rank}")

        # elastic launch support
        max_restarts = os.getenv("MAX_RESTARTS", "0")
        rdzv_id = os.getenv("RDZV_ID")
        min_nnodes = os.getenv("MIN_NNODES")
        max_nnodes = os.getenv("MAX_NNODES")

        env = deepcopy(os.environ)
        if is_env_enabled("OPTIM_TORCH", "1"):
            # optimize DDP, see https://zhuanlan.zhihu.com/p/671834539
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            env["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

        torchrun_args = [
            "torchrun",
            "--nproc-per-node",
            nproc_per_node,
        ]
        if rdzv_id is not None:
            # launch elastic job with fault tolerant support when possible
            # see also https://docs.pytorch.org/docs/stable/elastic/train_script.html
            rdzv_nnodes = nnodes
            # elastic number of nodes if MIN_NNODES and MAX_NNODES are set
            if min_nnodes is not None and max_nnodes is not None:
                rdzv_nnodes = f"{min_nnodes}:{max_nnodes}"

            torchrun_args.extend(
                [
                    "--nnodes",
                    rdzv_nnodes,
                    "--rdzv-id",
                    rdzv_id,
                    "--rdzv-backend",
                    "c10d",
                    "--rdzv-endpoint",
                    f"{master_addr}:{master_port}",
                    "--max-restarts",
                    max_restarts,
                ]
            )
        else:
            # NOTE: DO NOT USE shell=True to avoid security risk
            torchrun_args.extend(
                [
                    "--nnodes",
                    nnodes,
                    "--node_rank",
                    node_rank,
                    "--master_addr",
                    master_addr,
                    "--master_port",
                    master_port,
                ]
            )

        script_args = [__file__, command] + sys.argv[1:]
        process = subprocess.run(
            torchrun_args + script_args,
            env=env,
            check=True,
        )

        sys.exit(process.returncode)

    elif command == "chat":
        from .samplers.cli_sampler import run_chat

        run_chat()

    elif command == "merge":
        from llamafactory.v1.plugins.model_plugins.peft import merge_and_export_model

        merge_and_export_model()

    elif command == "env":
        raise NotImplementedError("Environment information is not implemented yet.")

    elif command == "version":
        raise NotImplementedError("Version information is not implemented yet.")

    elif command == "help":
        print(USAGE)

    elif command in _DIST_TRAIN_COMMANDS:
        # Single GPU training without torchrun
        if command in ("train", "sft"):
            from llamafactory.v1.trainers.sft_trainer import run_sft

            run_sft()
        elif command == "dpo":
            from llamafactory.v1.trainers.dpo_trainer import run_dpo

            run_dpo()
        elif command == "rm":
            from llamafactory.v1.trainers.rm_trainer import run_rm

            run_rm()

    else:
        print(f"Unknown command: {command}.\n{USAGE}")


def main():
    # sys.argv[1] contains the command (sft/dpo/rm/train), sys.argv[2:] contains the rest args
    command = sys.argv[1] if len(sys.argv) > 1 else "sft"

    # Routing needs the sub-command, but downstream trainers usually expect argv without it.
    if command in _DIST_TRAIN_COMMANDS:
        sys.argv.pop(1)
    else:
        # Backward-compat: if someone runs `torchrun launcher.py config.yaml`,
        # treat it as sft by default.
        if len(sys.argv) > 1 and sys.argv[1].endswith((".yaml", ".yml")):
            command = "sft"
    if command in ("train", "sft"):
        from llamafactory.v1.trainers.sft_trainer import run_sft

        run_sft()
    elif command == "dpo":
        from llamafactory.v1.trainers.dpo_trainer import run_dpo

        run_dpo()
    elif command == "rm":
        from llamafactory.v1.trainers.rm_trainer import run_rm

        run_rm()


if __name__ == "__main__":
    main()
