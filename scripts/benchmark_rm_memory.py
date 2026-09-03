"""Benchmark: Memory and speed impact of RM trainer optimizations.

Runs two separate training processes (baseline vs optimized) and compares results.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_rm_memory.py
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_rm_memory.py --model Qwen/Qwen3-4B-Instruct-2507
"""

import argparse
import json
import os
import subprocess
import sys


def run_single_benchmark(mode: str, model: str, dataset: str, template: str,
                         steps: int, batch_size: int, cutoff_len: int) -> dict:
    """Run a single benchmark in a subprocess and return stats."""
    output_dir = f"output/bench_rm_{mode}"
    script = f"""
import gc, json, os, sys, time
import torch

# Patch the trainer BEFORE importing run_exp
if "{mode}" == "baseline":
    import llamafactory.train.rm.trainer as rm_trainer
    from transformers import Trainer
    from types import MethodType

    _OrigInit = rm_trainer.PairwiseTrainer.__init__

    def _baseline_init(self, finetuning_args, processor=None, **kwargs):
        # Skip the norm hook optimization: call the Trainer constructor directly
        from llamafactory.extras.packages import is_transformers_version_greater_than
        from llamafactory.train.callbacks import FixValueHeadModelCallback, SaveProcessorCallback

        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        # Cast v_head dtype (keep this — it's needed for correctness)
        model = kwargs["model"]
        model_dtype = next(model.pretrained_model.parameters()).dtype
        model.v_head = model.v_head.to(dtype=model_dtype)

        Trainer.__init__(self, **kwargs)
        self.model_accepts_loss_kwargs = False
        self.finetuning_args = finetuning_args
        self.can_return_loss = True
        self.add_callback(FixValueHeadModelCallback)
        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

    rm_trainer.PairwiseTrainer.__init__ = _baseline_init

    def _baseline_loss(self, model, inputs, return_outputs=False, **kwargs):
        # TRL's default: calls model with output_hidden_states=True
        _, _, values = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        batch_size = inputs["input_ids"].size(0) // 2
        chosen_masks, rejected_masks = torch.split(inputs["attention_mask"], batch_size, dim=0)
        chosen_rewards, rejected_rewards = torch.split(values, batch_size, dim=0)
        chosen_scores = chosen_rewards.gather(dim=-1, index=(chosen_masks.sum(dim=-1, keepdim=True) - 1))
        rejected_scores = rejected_rewards.gather(dim=-1, index=(rejected_masks.sum(dim=-1, keepdim=True) - 1))
        chosen_scores, rejected_scores = chosen_scores.squeeze(), rejected_scores.squeeze()
        loss = -torch.nn.functional.logsigmoid(chosen_scores.float() - rejected_scores.float()).mean()
        if return_outputs:
            return loss, (loss, chosen_scores, rejected_scores)
        return loss

    rm_trainer.PairwiseTrainer.compute_loss = _baseline_loss

from llamafactory.train.tuner import run_exp

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.perf_counter()

run_exp({{
    "stage": "rm",
    "model_name_or_path": "{model}",
    "do_train": True,
    "finetuning_type": "lora",
    "dataset": "{dataset}",
    "template": "{template}",
    "cutoff_len": {cutoff_len},
    "overwrite_output_dir": True,
    "per_device_train_batch_size": {batch_size},
    "max_steps": {steps},
    "bf16": True,
    "report_to": "none",
    "logging_steps": 1,
    "save_steps": 999999,
    "output_dir": "{output_dir}",
}})

torch.cuda.synchronize()
elapsed = time.perf_counter() - start
peak_mem = torch.cuda.max_memory_allocated() / 1024**3

stats = {{
    "elapsed_sec": round(elapsed, 2),
    "sec_per_step": round(elapsed / {steps}, 3),
    "peak_memory_gib": round(peak_mem, 2),
}}
# Write stats to a file
with open("{output_dir}/bench_stats.json", "w") as f:
    json.dump(stats, f)
print("BENCH_STATS:" + json.dumps(stats))
"""
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env, capture_output=True, text=True, timeout=3600,
    )

    # Print subprocess output for visibility
    if result.stdout:
        for line in result.stdout.strip().split("\n")[-5:]:
            print(f"  [{mode}] {line}")
    if result.returncode != 0:
        print(f"  [{mode}] STDERR (last 10 lines):")
        for line in result.stderr.strip().split("\n")[-10:]:
            print(f"  [{mode}]   {line}")

    # Extract stats
    for line in result.stdout.split("\n"):
        if line.startswith("BENCH_STATS:"):
            return json.loads(line[len("BENCH_STATS:"):])

    # Fallback: try reading from file
    stats_file = f"{output_dir}/bench_stats.json"
    if os.path.exists(stats_file):
        with open(stats_file) as f:
            return json.loads(f.read())

    raise RuntimeError(f"{mode} benchmark failed. Exit code: {result.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark RM trainer optimizations")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--dataset", default="ultrafeedback")
    parser.add_argument("--template", default="qwen3")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cutoff-len", type=int, default=512)
    args = parser.parse_args()

    import torch
    print("=" * 70)
    print("RM Trainer Optimization Benchmark")
    print("=" * 70)
    print(f"Model:      {args.model}")
    print(f"Dataset:    {args.dataset}")
    print(f"Steps:      {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Cutoff len: {args.cutoff_len}")
    if torch.cuda.is_available():
        print(f"GPU:        {torch.cuda.get_device_name()}")
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {total:.1f} GiB")
    print("=" * 70)

    # Run baseline
    print("\n>>> BASELINE (output_hidden_states=True + full lm_head)")
    baseline = run_single_benchmark(
        "baseline", args.model, args.dataset, args.template,
        args.steps, args.batch_size, args.cutoff_len,
    )
    print(f"  Peak Memory: {baseline['peak_memory_gib']} GiB")
    print(f"  Time/step:   {baseline['sec_per_step']}s")

    # Run optimized
    print("\n>>> OPTIMIZED (norm hook + output_hidden_states=False)")
    optimized = run_single_benchmark(
        "optimized", args.model, args.dataset, args.template,
        args.steps, args.batch_size, args.cutoff_len,
    )
    print(f"  Peak Memory: {optimized['peak_memory_gib']} GiB")
    print(f"  Time/step:   {optimized['sec_per_step']}s")

    # Comparison
    mem_saved = baseline["peak_memory_gib"] - optimized["peak_memory_gib"]
    mem_pct = (mem_saved / baseline["peak_memory_gib"]) * 100 if baseline["peak_memory_gib"] > 0 else 0
    speedup = baseline["sec_per_step"] / optimized["sec_per_step"] if optimized["sec_per_step"] > 0 else 0

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Metric':<25} {'Baseline':>12} {'Optimized':>12} {'Delta':>15}")
    print("-" * 70)
    print(f"{'Peak GPU Memory (GiB)':<25} {baseline['peak_memory_gib']:>12.2f} {optimized['peak_memory_gib']:>12.2f} {mem_saved:>+12.2f} GiB ({mem_pct:.1f}%)")
    print(f"{'Time/step (sec)':<25} {baseline['sec_per_step']:>12.3f} {optimized['sec_per_step']:>12.3f}   {speedup:.2f}x speedup")
    print(f"{'Total time (sec)':<25} {baseline['elapsed_sec']:>12.2f} {optimized['elapsed_sec']:>12.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
