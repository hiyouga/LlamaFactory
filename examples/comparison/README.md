# Fine-Tuning Comparison Feature 
The purpose of this update is to allow people to easily compare fine-tuning strategies given a specific fine-tuning configuration.
This new feature extends the existing system and is meant to work with pre-defined datasets, algorithms, and metrics.

To compare training metrics, you need to have already defined .yaml configuration files that include the LLM, the dataset, and the fine-tuning strategy. For examples of such configurations, see /examples.

## EXPERIMENTAL FEATURE
⚠️ This model evaluation feature has undergone limited testing. Default settings run a very small number of samples for quick tests. Increasing sample size or batch size may increase memory usage or runtime. ⚠️

---

## Installations

Follow the installation guide on LLaMa's main page (also written below for convenience):

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e .
pip install -r requirements/metrics.txt
```

### IMPORTANT (for Windows users)
#### Install PyTorch

You need to manually install the GPU version of PyTorch on Windows. Please refer to the official website and run the following:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available())"
```

#### Install BitsAndBytes

If you want to enable QLoRA on Windows, install a pre-built version of the bitsandbytes library that supports CUDA 11.1 to 12.2. Choose the appropriate release version based on your CUDA version:

```bash
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.2.post2-py3-none-win_amd64.whl
```

## Structure

- **Demo:** `/examples/comparison/ft_comparison_demo.py`  
  Runs sequential fine-tuning algorithms (LoRA and QLoRA) and saves 4 metrics at each checkpoint.

- **Core logic:** `/scripts/finetuning_comparison`  
  Ensures models are loaded correctly and metrics are computed properly.

- **Tests:** `tests/eval`  
  Minimal tests to ensure compatibility with existing functionality.

## Running the Demos
### Option 1: Using ft_comparison_demo.py

Open `ft_comparison_demo.py` and adjust the following paths:

- `ft_yaml_1` and `ft_yaml_2`: point to existing training configuration files
- `output_dir`: path to save the results (e.g., `data/ft_comparison_results` or `outputs`)

Save and run the script:

```bash
python examples/comparison/ft_comparison_demo.py
```

### Option 2: Using the CLI
```bash
python scripts/finetuning_comparison/cli_yaml_compare.py \
    --first examples/train_lora/qwen3_lora_sft.yaml \
    --second examples/train_qlora/qwen3_lora_sft_bnb_npu.yaml \
    --out data/ft_comparison_results
```

Adjust the models as needed.

## Features

- Compare 2 existing training configurations and view these metrics:
    - Evaluation loss: model accuracy indicator
    - Perplexity: reinterpretation of loss, also indicates accuracy
    - Latency (ms): responsiveness, relevant for real-time applications
    - Peak VRAM (MB): maximum GPU memory usage during inference
- Export a CSV summarizing these metrics
- Generate a plot comparing all metrics side-by-side. If GPU resources are insufficient, generate CSV and plots with dummy data
- Unit tests include fallback strategy for dummy data

## Possible Expansions

- Further testing of the feature in environments with sufficient memory to handle LoRA finetuning and merging weights into the base model.
- Add more relevant metrics (e.g., ROUGE, human evaluation)
- Make input more flexible: accept a dataset, LLM, 2 fine-tuning algorithms (or None), and automatically generate .yaml configs before comparison
- Improve plotting functionality for finer-grained analysis during training