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

import inspect
import json
import os
import re
import typing
from typing import TYPE_CHECKING, Any, Optional

from ...extras.logging import get_logger


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset


logger = get_logger(__name__)
_GENERATION_REGEX = re.compile(r"\{%-?\s+generation\s+-?%\}")
_END_GENERATION_REGEX = re.compile(r"\{%-?\s+endgeneration\s+-?%\}")
_ASSISTANT_ELIF_REGEX = re.compile(
    r"(\{%\s*elif\s+message\['role'\]\s*==\s*'assistant'\s*%\})"
    r"(.*?)"
    r"(\{%\s*endif\s*%\})",
    flags=re.DOTALL,
)


def supports_hf_chat_template() -> bool:
    r"""Return whether the installed Megatron Bridge supports HF chat templates."""
    try:
        from megatron.bridge.data.datasets.sft import GPTSFTChatDataset

        return "use_hf_tokenizer_chat_template" in inspect.signature(GPTSFTChatDataset.__init__).parameters
    except Exception:
        return False


def _inject_generation_block(chat_template: str) -> str:
    r"""Wrap assistant content with ``{% generation %}`` when missing."""
    if _GENERATION_REGEX.search(chat_template):
        return chat_template

    match = _ASSISTANT_ELIF_REGEX.search(chat_template)
    if match is None:
        raise ValueError(
            "Cannot inject {% generation %} into chat template: "
            "no `{% elif message['role'] == 'assistant' %}` block found."
        )

    body = match.group(2)
    if _END_GENERATION_REGEX.search(body):
        return chat_template

    patched = (
        chat_template[: match.start()]
        + match.group(1)
        + "{% generation %}"
        + body
        + "{% endgeneration %}"
        + match.group(3)
        + chat_template[match.end() :]
    )
    if not _GENERATION_REGEX.search(patched):
        raise ValueError("Failed to inject {% generation %} into chat template.")
    return patched


def build_chat_template_with_generation(
    tokenizer_path: str | None,
    trust_remote_code: bool = False,
    template_name: str | None = None,
) -> str | None:
    r"""Build a chat template that supports assistant-only loss masks.

    Prefers the LLaMA-Factory registered template (same formatting as HF Trainer),
    then falls back to patching the tokenizer's native chat template.
    """
    if not tokenizer_path:
        return None

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)
    except Exception as exc:
        logger.warning_rank0(f"Failed to load tokenizer for chat template patching: {exc}")
        return None

    if template_name:
        try:
            from ...data.template import TEMPLATES

            template = TEMPLATES.get(template_name)
            if template is not None:
                return _inject_generation_block(template._get_jinja_template(tokenizer))
        except Exception as exc:
            logger.warning_rank0(f"Failed to build LLaMA-Factory chat template '{template_name}': {exc}")

    native = tokenizer.chat_template
    if not isinstance(native, str) or not native:
        return None

    try:
        return _inject_generation_block(native)
    except Exception as exc:
        logger.warning_rank0(f"Failed to inject {{% generation %}} into native chat template: {exc}")
        return None


def tokenizer_supports_hf_chat_template(
    tokenizer_path: str | None,
    trust_remote_code: bool = False,
    template_name: str | None = None,
) -> bool:
    r"""Return whether Megatron Bridge can use HF chat templates for this tokenizer."""
    if not supports_hf_chat_template() or not tokenizer_path:
        return False

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)
        if _GENERATION_REGEX.search(tokenizer.chat_template or ""):
            return True
    except Exception as exc:
        logger.warning_rank0(f"Failed to inspect tokenizer chat template: {exc}")
        return False

    return build_chat_template_with_generation(tokenizer_path, trust_remote_code, template_name) is not None


def get_sft_dataset_kwargs(
    tokenizer_path: str | None = None,
    trust_remote_code: bool = False,
    template_name: str | None = None,
) -> dict[str, Any]:
    r"""Return dataset kwargs compatible with the installed Megatron Bridge version."""
    kwargs: dict[str, Any] = {"chat": True}
    if not tokenizer_supports_hf_chat_template(tokenizer_path, trust_remote_code, template_name):
        return kwargs

    chat_template = build_chat_template_with_generation(tokenizer_path, trust_remote_code, template_name)
    if chat_template is None:
        return kwargs

    kwargs["use_hf_tokenizer_chat_template"] = True
    kwargs["chat_template"] = chat_template
    # Chat templates already include BOS/EOS / turn markers.
    kwargs["add_bos"] = False
    kwargs["add_eos"] = False
    logger.info_rank0(
        "Using HuggingFace chat template with {% generation %} for Megatron Bridge SFT "
        f"(template={template_name or 'native'})."
    )
    return kwargs


def get_finetuning_dataloader_type(disable_shuffling: bool = False) -> str:
    r"""Return a finetuning dataloader type supported by the installed Megatron Bridge.

    Prefer sequential ``batch``/``single`` samplers. When shuffling is disabled for
    HF loss comparison, avoid the random ``cyclic`` sampler.
    """
    try:
        from megatron.bridge.training.config import FinetuningDatasetConfig

        field = FinetuningDatasetConfig.__dataclass_fields__.get("dataloader_type")
        if field is None:
            return "single"

        choices: set[str] = set()
        for arg in typing.get_args(field.type):
            for choice in typing.get_args(arg):
                if isinstance(choice, str):
                    choices.add(choice)

        if disable_shuffling:
            if "batch" in choices:
                return "batch"
            if "single" in choices:
                return "single"
            return next(iter(choices), "single")

        if "batch" in choices:
            return "batch"
        if "single" in choices:
            return "single"
        return next(iter(choices), "single")
    except Exception:
        return "single"


def _role_to_sharegpt(role: str) -> str:
    mapping = {"user": "User", "assistant": "Assistant", "system": "System"}
    return mapping.get(role, role.capitalize())


def _example_to_record(
    example: dict[str, Any], stage: str = "sft", use_messages_format: bool = True
) -> dict[str, Any] | None:
    r"""Convert an aligned LLaMA-Factory example to Megatron Bridge JSONL format."""
    if example.get("text") is not None:
        return {"text": example["text"]}

    prompt = example.get("_prompt")
    if stage == "pt":
        if not prompt:
            return None
        return {"text": prompt[0]["content"]}

    response = example.get("_response")
    if not prompt or not response:
        return None

    if use_messages_format:
        messages = []
        system = example.get("_system")
        if system:
            messages.append({"role": "system", "content": system})
        for message in prompt:
            messages.append({"role": message["role"], "content": message["content"]})
        for message in response:
            messages.append({"role": message["role"], "content": message["content"]})
        record: dict[str, Any] = {"messages": messages}
        tools = example.get("_tools")
        if tools:
            record["tools"] = tools
        return record

    conversations = []
    for message in prompt:
        conversations.append({"from": _role_to_sharegpt(message["role"]), "value": message["content"]})
    for message in response:
        conversations.append({"from": _role_to_sharegpt(message["role"]), "value": message["content"]})

    return {
        "system": example.get("_system") or "",
        "conversations": conversations,
        "mask": "User",
    }


def _remove_stale_memmap_index(path: str) -> None:
    r"""Remove cached memmap index files after rewriting a JSONL dataset."""
    for suffix in (".idx.npy", ".idx.info"):
        index_path = path + suffix
        if os.path.exists(index_path):
            os.remove(index_path)
            logger.info_rank0(f"Removed stale Megatron dataset index: {index_path}")


def _write_jsonl(
    path: str,
    dataset: "Dataset | IterableDataset",
    stage: str = "sft",
    use_messages_format: bool = True,
) -> int:
    count = 0
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for example in dataset:
            record = _example_to_record(example, stage=stage, use_messages_format=use_messages_format)
            if record is None:
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    _remove_stale_memmap_index(path)
    return count


def export_dataset_for_megatron_bridge(
    train_dataset: "Dataset | IterableDataset",
    output_dir: str,
    eval_dataset: Optional["Dataset | IterableDataset | dict[str, Dataset]"] = None,
    val_size: float = 0.0,
    seed: int = 42,
    stage: str = "sft",
    model_name_or_path: str | None = None,
    trust_remote_code: bool = False,
    template_name: str | None = None,
) -> str:
    r"""Export aligned LLaMA-Factory datasets to Megatron Bridge JSONL files."""
    os.makedirs(output_dir, exist_ok=True)
    use_messages_format = stage != "sft" or tokenizer_supports_hf_chat_template(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        template_name=template_name,
    )
    if stage == "sft" and not use_messages_format:
        logger.info_rank0(
            "Cannot enable HuggingFace chat template for Megatron Bridge; "
            "exporting ShareGPT conversations for legacy preprocessing."
        )

    if val_size > 0 and eval_dataset is None:
        split = train_dataset.train_test_split(test_size=val_size, seed=seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]

    train_path = os.path.join(output_dir, "training.jsonl")
    train_count = _write_jsonl(
        train_path,
        train_dataset,
        stage=stage,
        use_messages_format=use_messages_format,
    )
    logger.info_rank0(f"Exported {train_count} training samples to {train_path}.")

    if isinstance(eval_dataset, dict):
        for name, dataset in eval_dataset.items():
            split_name = "validation" if name == "validation" else name
            eval_path = os.path.join(output_dir, f"{split_name}.jsonl")
            eval_count = _write_jsonl(
                eval_path,
                dataset,
                stage=stage,
                use_messages_format=use_messages_format,
            )
            logger.info_rank0(f"Exported {eval_count} {split_name} samples to {eval_path}.")
    elif eval_dataset is not None:
        eval_path = os.path.join(output_dir, "validation.jsonl")
        eval_count = _write_jsonl(
            eval_path,
            eval_dataset,
            stage=stage,
            use_messages_format=use_messages_format,
        )
        logger.info_rank0(f"Exported {eval_count} validation samples to {eval_path}.")

    return output_dir
