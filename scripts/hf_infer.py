import gc
import json
from typing import Optional

import torch
import fire
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    Seq2SeqTrainingArguments,
)
from peft import PeftModel

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer


def hf_infer(
    model_name_or_path: str,
    adapter_name_or_path: str = None,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 2048,
    max_samples: Optional[int] = None,
    save_name: str = "generated_predictions.jsonl",
    temperature: float = 0.95,
    top_p: float = 0.7,
    top_k: int = 50,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.0,
    skip_special_tokens: bool = True,
    default_system: Optional[str] = None,
    enable_thinking: bool = True,
    seed: Optional[int] = None,
    batch_size: int = 4,
    device: str = None,
):
    """
    Perform batch generation using Hugging Face transformers backend.

    Usage:
        python hf_infer.py --model_name_or_path meta-llama/Llama-2-7b-hf --template llama --dataset alpaca_en_demo --batch_size 16 --save_name test.jsonl 
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model_args, data_args, _, _ = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=8,
            default_system=default_system,
            enable_thinking=enable_thinking,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    )

    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False

    # --- Load model ---
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=model_args.infer_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_name_or_path is not None:
        model = PeftModel.from_pretrained(model, adapter_name_or_path)

    model.eval()

    # --- Load dataset ---
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)
    train_dataset = dataset_module["train_dataset"]

    # --- Generation configuration ---
    gen_cfg = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        do_sample=True,
    )

    all_prompts, all_preds, all_labels = [], [], []

    for i in tqdm(range(0, len(train_dataset), batch_size), desc="Processing batched inference"):
        batch = train_dataset[i : min(i + batch_size, len(train_dataset))]

        input_ids = [torch.tensor(x) for x in batch["input_ids"]]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        ).to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                **gen_cfg.to_dict(),
            )

        # Decode predictions
        for j in range(len(batch["input_ids"])):
            prompt = tokenizer.decode(batch["input_ids"][j], skip_special_tokens=skip_special_tokens)
            label = tokenizer.decode(
                list(filter(lambda x: x != IGNORE_INDEX, batch["labels"][j])),
                skip_special_tokens=skip_special_tokens,
            )
            pred = tokenizer.decode(outputs[j][len(batch["input_ids"][j]) :], skip_special_tokens=skip_special_tokens)

            all_prompts.append(prompt)
            all_preds.append(pred)
            all_labels.append(label)

        gc.collect()

    # Save all results
    with open(save_name, "w", encoding="utf-8") as f:
        for text, pred, label in zip(all_prompts, all_preds, all_labels):
            f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

    print("*" * 70)
    print(f"{len(all_prompts)} total generated results have been saved at {save_name}.")
    print("*" * 70)


if __name__ == "__main__":
    fire.Fire(hf_infer)
