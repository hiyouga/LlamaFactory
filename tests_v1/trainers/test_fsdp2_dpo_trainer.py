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
from pathlib import Path

import pytest


@pytest.mark.xfail(reason="CI machines may OOM when heavily loaded.")
@pytest.mark.runs_on(["cuda", "npu", "xpu"])
def test_fsdp2_dpo_trainer(tmp_path: Path):
    """Test FSDP2 DPO trainer with sigmoid loss by simulating `llamafactory-cli dpo config.yaml`."""
    config_yaml = """\
model: Qwen/Qwen3-0.6B
trust_remote_code: true
model_class: llm

kernel_config:
    name: auto

quant_config: null

dist_config:
    name: fsdp2
    dcp_path: null

init_config:
    name: init_on_default

# PEFT Configuration
peft_config:
    name: lora
    r: 8
    lora_alpha: 16
    lora_dropout: 0.0
    target_modules: all

### data
train_dataset: data/v1_dpo_demo.yaml

### training
output_dir: {output_dir}
micro_batch_size: 1
global_batch_size: 1
cutoff_len: 2048
learning_rate: 1.0e-4
bf16: false
max_steps: 1
pref_loss: sigmoid
pref_beta: 0.1
dpo_label_smoothing: 0.0

### sample
sample_backend: hf
max_new_tokens: 128
"""
    # Create output directory
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_yaml.format(output_dir=str(output_dir)))

    # Set up environment variables
    env = os.environ.copy()
    env["USE_V1"] = "1"  # Use v1 launcher
    env["FORCE_TORCHRUN"] = "1"  # Force distributed training via torchrun

    # Run the CLI command via subprocess
    result = subprocess.run(
        [sys.executable, "-m", "llamafactory.cli", "dpo", str(config_file)],
        env=env,
        capture_output=True,
        cwd=str(Path(__file__).parent.parent.parent),  # LLaMA-Factory root
    )

    # Decode output with error handling (progress bars may contain non-UTF-8 bytes)
    stderr = result.stderr.decode("utf-8", errors="replace")

    # Check the result
    assert result.returncode == 0, f"DPO training failed with return code {result.returncode}\nSTDERR: {stderr}"
