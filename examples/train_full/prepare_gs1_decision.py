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

"""Intersect sampled TRAIN records with the native disjoint two-stage plans."""

import argparse
import hashlib
import json
import random
from pathlib import Path


def prepare(sample, corpus, output, seed=41):
    questions = {}
    with Path(sample).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("partition", "train") != "train":
                raise ValueError("Only TRAIN records may enter the stage datasets.")
            for qid in row["request"]["questions"]:
                key = (row["id"], qid)
                if key in questions:
                    raise ValueError(f"Duplicate sampled question: {key}")
                questions[key] = row
    stages, seen = {}, set()
    for stage, filename in (("nll", "warmup_plan.jsonl"), ("joint", "sft_plan.jsonl")):
        selected = []
        with (Path(corpus) / filename).open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                key = (item["record_id"], item["question_id"])
                if key in questions:
                    if key in seen:
                        raise ValueError(f"Repeated or overlapping stage question: {key}")
                    seen.add(key)
                    selected.append(key)
        if not selected:
            raise ValueError(f"The sample has no {stage} questions.")
        if stage == "joint":
            identity = json.dumps(["yes1.rng.v2", seed, "v15/joint_full_finetune"], separators=(",", ":"))
            named_seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big") % (2**63)
            random.Random(named_seed).shuffle(selected)
        stages[stage] = selected
    if seen != set(questions):
        raise ValueError("Some sampled questions are absent from the native stage plans.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for stage, selected in stages.items():
        with (output / f"{stage}.jsonl").open("w", encoding="utf-8") as stream:
            for rid, qid in selected:
                row = questions[rid, qid]
                record = dict(
                    id=rid,
                    request={**row["request"], "questions": {qid: row["request"]["questions"][qid]}},
                    targets={qid: row["targets"][qid]},
                )
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    registry = {f"gs1_{stage}": dict(file_name=f"{stage}.jsonl", formatting="gs1") for stage in stages}
    (output / "dataset_info.json").write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    return {stage: len(selected) for stage, selected in stages.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sample", "corpus", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    print(prepare(**vars(args)))
