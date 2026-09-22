import os
import sys
import unittest
from unittest.mock import patch

import pandas as pd


sys.path.insert(0, os.path.abspath("scripts/finetuning_comparison"))
from yaml_compare import compare_two_yamls


class TestYamlCompare(unittest.TestCase):
    """
    Test the fallback functionality for comparing two YAML configs when real training/eval is not possible.
    """
    @patch("yaml_compare.run_yaml_training")
    @patch("yaml_compare.evaluate_checkpoint")
    def test_compare_two_yamls(self, mock_eval, mock_train):
        # Generate mock checkpoint paths
        mock_train.side_effect = lambda yaml_path: f"/fake/checkpoint/{yaml_path}"
        # Generate mock evaluation metrics
        mock_eval.side_effect = lambda ckpt, eval_config: {
            "eval_loss": 1.0,
            "perplexity": 10.0,
            "latency_ms": 50.0,
            "peak_vram_mb": 12000.0
        }

        df = compare_two_yamls("first.yaml", "second.yaml", "data/ft_comparison_results")
        self.assertIsInstance(df, pd.DataFrame) # check output type (should be a DataFrame)
        self.assertEqual(len(df), 2) # check there are results for both models
        # Check that all metrics are logged
        self.assertIn("eval_loss", df.columns)
        self.assertIn("perplexity", df.columns)
        self.assertIn("latency_ms", df.columns)
        self.assertIn("peak_vram_mb", df.columns)

if __name__ == "__main__":
    unittest.main()
