# Copyright 2026 the LlamaFactory team.
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

import json
from pathlib import Path

from llamafactory.v1.trainers.sft_trainer import run_sft


DEMO_DATA = Path(__file__).parents[2] / "data" / "v1_sft_demo.yaml"


def test_sft_trainer_writes_timing_metrics(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(Path(__file__).parents[2])  # dataset paths in the demo yaml are repo-relative
    output_dir = tmp_path / "outputs"
    run_sft(
        {
            "model": "llamafactory/tiny-random-qwen3",
            "train_dataset": str(DEMO_DATA),
            "output_dir": str(output_dir),
            "micro_batch_size": 1,
            "global_batch_size": 2,
            "cutoff_len": 128,
            "batching_workers": 1,
            "bf16": False,
            "max_steps": 4,
            "logging_steps": 2,
        }
    )

    logs = [json.loads(line) for line in (output_dir / "trainer_log.jsonl").read_text().splitlines()]
    assert [log["step"] for log in logs] == [2, 4]
    for log in logs:
        assert {"elapsed_time", "remaining_time", "step_time", "tokens_per_second"} <= log.keys()
        assert log["step_time"] > 0
        assert log["tokens_per_second"] > 0
    assert logs[-1]["remaining_time"] == "0:00:00"

    results = json.loads((output_dir / "train_results.json").read_text())
    assert results["global_step"] == 4
    assert results["train_runtime"] > 0
    assert results["train_tokens"] > 0
    assert results["train_tokens_per_second"] > 0
    assert results["train_steps_per_second"] > 0
