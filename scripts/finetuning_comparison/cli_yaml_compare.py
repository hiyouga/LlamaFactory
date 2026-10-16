#!/usr/bin/env python
"""Mapping of the CLI command to compare 2 LLaMA-Factory .yaml files, to the actual function.
"""

#------------------------------IMPORTS---------------------------------#
import argparse

from yaml_compare import compare_two_yamls


#-----------------------------FUNCTIONS---------------------------------#
def parse_args():
    """CLI flag definition:
    --first: path to the first model to fine-tune
    --second: path to the second model to fine-tune
    --out: output directory for evaluation metrics and plots (default: data/ft_comparison_results)
    """
    p = argparse.ArgumentParser(description="Compare two YAML trainings.")
    p.add_argument("--first", required=True, help="Path to first YAML")
    p.add_argument("--second", required=True, help="Path to second YAML")
    p.add_argument("--out", default="data/ft_comparison_results", help="Output directory")
    return p.parse_args()

def main():
    """Pass the arguments extracted from the CLI to the comparison function.
    """
    args = parse_args()
    compare_two_yamls(args.first, args.second, args.out)

#-----------------------------MAIN---------------------------------#
if __name__ == "__main__":
    main()
