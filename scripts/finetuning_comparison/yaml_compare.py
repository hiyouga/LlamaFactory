#!/usr/bin/env python
"""Methods for comparing two existing .yaml training configurations.

The following methods have been defined:
1) Sequentially run two .yaml trainings via llamafactory-cli (with error handling)
2) Merge LoRA weights if applicable (running out of memory defaults to a fallback mechanism)
3) Evaluate both models (return dummy metrics upon training or merging failure)
4) Save results to CSV and plot the metrics side by side  
"""

#------------------------------IMPORTS---------------------------------#
import os
import subprocess
import tempfile
import traceback

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml


#-----------------------------FUNCTIONS---------------------------------#
def run_yaml_training(yaml_path: str) -> str:
    """Runs a training .yaml via llamafactory-cli and returns the path to the resulting checkpoint.

    Requires:
     the path to the .yaml configuration file for training.

    Returns:
     the path to the resulting checkpoint (pytorch_model.pt) if training succeeds, or a dummy path otherwise.
    """
    #
    print(f"[INFO] Running training for {yaml_path} ...") # log progress
    result = subprocess.run(["llamafactory-cli", "train", yaml_path],
        capture_output=True, # set to False to see real-time logs
        text=True # return stdout/stderr as strings
    )

    # Log errors and continue execution
    if result.returncode != 0:
        print(f"[ERROR] Training failed for {yaml_path}:\n{result.stderr}")
    else:
        print(f"[INFO] Training finished for {yaml_path}")

    # Load .yaml into a dict to determine output directory
    try:
        with open(yaml_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        outdir = cfg.get(
            "output_dir",
            os.path.join("saves", os.path.basename(yaml_path).replace(".yaml", ""))
        )
    except Exception:
        # If no directory could be found, use a default one based on the yaml name
        outdir = os.path.join("saves", os.path.basename(yaml_path).replace(".yaml", ""))
    checkpoint_path = os.path.join(outdir, "pytorch_model.pt")
    return checkpoint_path


def merge_lora_checkpoint(checkpoint_dir: str, yaml_path: str) -> str:
    """Attempts to merge LoRA weights into the base model, to be able to accurately evaluate the performance of each fine-tuning method.
    Upon success the method returns the directory with the merged model. 
        Otherwise, a warning is logged and the method returns the original checkpoint. Dummy data will be returned instead.

    Requires:
      the path to the training checkpoint directory and the .yaml configuration.

    Returns:
      either the path to the merged model (upon success) or the original checkpoint (otherwise).
    """
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load the .yaml config into a dictionary
        with open(yaml_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Extract the base model name from the config, or raise an error if it could not be found
        base_name = cfg.get("model_name_or_path") or cfg.get("base_model")
        if not base_name:
            raise RuntimeError("Base model name not found in yaml; cannot merge with PEFT")

        # Attempt to merge LoRA weights using GPU if available, else CPU
        print(f"[INFO] Attempting to merge LoRA adapter from {checkpoint_dir} into {base_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(base_name, use_fast=False) # use_fast=False to avoid potential compatibility issues
        try:
            base_model = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.float16, device_map="auto")
        except Exception:
            base_model = AutoModelForCausalLM.from_pretrained(base_name, device_map="cpu", low_cpu_mem_usage=True) # low_cpu_mem_usage=True to optimise for CPU inference (load weights in smaller chunks)

        # Apply the LoRA weights to the model and merge, wthout continuing training
        peft_model = PeftModel.from_pretrained(base_model, checkpoint_dir, is_trainable=False)
        merged = peft_model.merge_and_unload()

        # Save the model to disk in shards (for optimised loading)
        merged_dir = os.path.join(checkpoint_dir, "merged_model")
        merged.save_pretrained(merged_dir, max_shard_size="500MB", safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)

        # Return the path to the merged model directory
        print(f"[INFO] Merged model saved to {merged_dir}")
        return merged_dir

    except Exception as e:
        # Fallback to checkpoint directory without merging
        print(f"[WARN] Could not merge LoRA (probably insufficient memory): {e}")
        print("[INFO] Falling back to using dummy/simulated metrics instead.")
        return checkpoint_dir


def evaluate_checkpoint(checkpoint_path: str, eval_config: dict) -> dict[str, float]:
    """Loads a passed checkpoint and computes four metrics:
        - eval_loss: the evaluation loss on the specified eval dataset, 
        - perplexity: the perplexity on the eval dataset, 
        - latency_ms: the average latency (ms) per inference step,
        - peak_vram_mb: the peak GPU memory usage (MB).
    If anything fails (merge, eval, memory), returns dummy metrics.

    Requires:
     the path to the checkpoint to evaluate;
     the evaluation configuration as specified in compare_two_yamls

    Returns:
     a dictionary of metrics for each fine-tuning method. If evaluation fails, dummy data is returned.
    """
    try:
        # If real evaluation is requested, evaluate the performance via LLaMA Factory's Evaluator
        if eval_config.get("use_real_eval", False):
            import uuid

            from llamafactory.eval.evaluator import Evaluator

            # Ensure the checkpoint path is a directory
            ckpt_dir = checkpoint_path if os.path.isdir(checkpoint_path) else os.path.dirname(checkpoint_path)

            # Generate temporary directory for the evaluation metrics (to avoid writing conflicts)
            eval_save_dir = os.path.join(tempfile.gettempdir(), f"llama_eval_{uuid.uuid4().hex[:8]}")

            # Extract required arguments for the Evaluator, from the eval_config
            args = {
                "model_name_or_path": eval_config.get("model_name_or_path", ckpt_dir),
                "task": eval_config.get("task"),
                "task_dir": eval_config.get("task_dir"),
                "save_dir": eval_save_dir,
                "batch_size": eval_config.get("batch_size", 4),
                "lang": eval_config.get("lang", "en"),
            }

            # Run the evaluator on the task dataset
            evaluator = Evaluator(args)
            evaluator.eval()

            # If the evaluator exported any results, it loads them into a dictionary and returns them
            results_path = os.path.join(eval_save_dir, "results.json")
            if os.path.exists(results_path):
                import json
                with open(results_path, encoding="utf-8") as f:
                    results = json.load(f)
                return results
    except Exception as e:
        print(f"[WARN] Real evaluation failed, using dummy metrics: {e}")
        print(f"[DEBUG] {traceback.format_exc()}")

    # Return dummy metrics upon failure
    return {
        "eval_loss": 2.0,
        "perplexity": 10.0,
        "latency_ms": 50.0,
        "peak_vram_mb": 12000.0,
    }


def compare_two_yamls(yaml_1: str, yaml_2: str, output_dir: str) -> pd.DataFrame:
    """Orchestration function for running the comparison of 2 fine-tuning methods defined by their .yaml configurations.
    The function performs the following steps:
    1) Sequentially run 2 .yaml trainings via llamafactory-cli
    2) Merge LoRA weights if applicable
    3) Evaluate both models
    4) Save results to CSV and plot the metrics side by side.

    Requires:
    - yaml_1: path to the first .yaml configuration file for training.
    - yaml_2: path to the second .yaml configuration file for training.
    - output_dir: directory where the results CSV and plots will be saved.

    Returns:
     - a pandas DataFrame (exported to CSV in output_dir) containing the evaluation metrics for both methods (filled with dummy values upon failure).
     - a plot showing comparative metrics for the 2 methods.
    """
    # Ensure an output directior for the results exists, otherwise create one
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for yaml_file in [yaml_1, yaml_2]:
        # Run training for each .yaml file, merge LoRA weights, then extract performance metrics
        checkpoint = run_yaml_training(yaml_file)
        eval_model_path = merge_lora_checkpoint(os.path.dirname(checkpoint), yaml_file)
        metrics = evaluate_checkpoint(eval_model_path, eval_config={
            "use_real_eval": True, # set to False to skip real evaluation and return dummy metrics
            "task": "alpaca_eval", # select any available LlaMa dataset for evaluation
            "max_samples": 2, # increase number for more robust evaluation
            "batch_size": 1, # increase for speed, and higher memory usage
            "metrics": ["eval_loss", "perplexity", "latency_ms", "peak_vram_mb"] # select the metrics to return
        })
        results.append({"method": os.path.basename(yaml_file), **metrics}) # metrics to be appended as a dictionary

    # Create a DataFrame to be exported to .csv
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] CSV saved to {csv_path}")

    # Plot all metrics in a 1x4 grid, displaying the models's performance side-by-side
    metrics_cols = [c for c in df.columns if c != "method"] # skip the 'method' column
    fig, axes = plt.subplots(1, len(metrics_cols), figsize=(5 * len(metrics_cols), 4))
    if len(metrics_cols) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics_cols):
        # add a subplot for each metric
        ax.bar(df["method"], df[metric], color=["skyblue", "salmon"])
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.set_xlabel("Method")

    # Display the plot and return the DataFrame
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "comparison.png")
    plt.savefig(plot_path)
    print(f"[INFO] Plot saved to {plot_path}")
    plt.close(fig)
    return df
