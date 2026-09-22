#!/usr/bin/env python
#------------------------------IMPORTS---------------------------------#
import os
import sys

# Add the finetuning_comparison folder to path for imports
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(root_path, "scripts", "finetuning_comparison"))

from yaml_compare import compare_two_yamls

#-----------------------------FUNCTIONS---------------------------------#
def main():
    """
    Call the comparison function in yaml_compare.py.
    This is where you can specify:
    - the two models to compare by selecting the paths to their YAML configs;
    - the output directory for obtaining evaluation metrics (and plots).
    """
    compare_two_yamls(
        yaml_1=os.path.join(root_path, "examples/train_lora/qwen3_lora_sft.yaml"), # first model to be fine-tuned
        yaml_2=os.path.join(root_path, "examples/train_qlora/qwen3_lora_sft_bnb_npu.yaml"), # second model to be fine-tuned
        output_dir=os.path.join(root_path, "data/ft_comparison_results"), # output directory for evaluation metrics
    )

#-----------------------------MAIN---------------------------------#
if __name__ == "__main__":
    main()
