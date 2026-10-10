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

r"""Architecture summary of the selected model, shown on the Model page.

Reads ``config.json`` from a local model directory or from the Hugging Face cache (never downloads),
and falls back to a few built-in presets when neither is available.
"""

import json
import os
from dataclasses import dataclass


@dataclass
class Arch:
    layers: int = 28
    hidden: int = 1024
    inter: int = 3072
    heads: int = 16
    kv_heads: int = 8
    head_dim: int = 128
    vocab: int = 151936
    tie: bool = True


PRESETS: dict[str, Arch] = {
    "Qwen3-0.6B": Arch(28, 1024, 3072, 16, 8, 128, 151936, True),
    "Qwen3-1.7B": Arch(28, 2048, 6144, 16, 8, 128, 151936, True),
    "Qwen3-4B": Arch(36, 2560, 9728, 32, 8, 128, 151936, True),
    "Qwen3-8B": Arch(36, 4096, 12288, 32, 8, 128, 151936, False),
    "Qwen2.5-7B": Arch(28, 3584, 18944, 28, 4, 128, 152064, False),
    "Llama-3.2-1B": Arch(16, 2048, 8192, 32, 8, 64, 128256, True),
    "Llama-3.2-3B": Arch(28, 3072, 8192, 24, 8, 128, 128256, True),
    "Llama-3.1-8B": Arch(32, 4096, 14336, 32, 8, 128, 128256, False),
}


def arch_from_config(cfg: dict) -> Arch | None:
    if isinstance(cfg.get("text_config"), dict):
        base = dict(cfg["text_config"])
        base.setdefault("tie_word_embeddings", cfg.get("tie_word_embeddings"))
        cfg = base

    try:
        hidden = int(cfg["hidden_size"])
        heads = int(cfg["num_attention_heads"])
        return Arch(
            layers=int(cfg["num_hidden_layers"]),
            hidden=hidden,
            inter=int(cfg.get("intermediate_size") or 4 * hidden),
            heads=heads,
            kv_heads=int(cfg.get("num_key_value_heads") or heads),
            head_dim=int(cfg.get("head_dim") or hidden // heads),
            vocab=int(cfg["vocab_size"]),
            tie=bool(cfg.get("tie_word_embeddings", False)),
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def _config_file(model_path: str | None) -> str | None:
    path = (model_path or "").strip()
    if path and os.path.isdir(path):
        cfg_file = os.path.join(path, "config.json")
        return cfg_file if os.path.isfile(cfg_file) else None

    if path and "/" in path and not os.path.isabs(path):
        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(path, "config.json")
            if isinstance(cached, str) and os.path.isfile(cached):
                return cached
        except Exception:
            return None

    return None


def _read_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def find_arch(model_name: str | None, model_path: str | None) -> tuple[Arch | None, str]:
    r"""Look up the architecture: local config.json first, then the Hugging Face cache, then a preset."""
    cfg_file = _config_file(model_path)
    if cfg_file:
        cfg = _read_json(cfg_file)
        arch = arch_from_config(cfg) if cfg else None
        if arch:
            return arch, cfg_file

    name = f"{model_name or ''} {model_path or ''}".lower()
    for key in sorted(PRESETS, key=len, reverse=True):
        if key.lower() in name:
            return PRESETS[key], f"preset:{key}"

    return None, ""


def counts(arch: Arch, lora_rank: int = 8) -> dict[str, int]:
    r"""Parameter counts of a decoder-only transformer (gated MLP, untied or tied embeddings)."""
    attn = arch.hidden * arch.head_dim * (2 * arch.heads + 2 * arch.kv_heads)
    mlp = 3 * arch.hidden * arch.inter
    layer = attn + mlp + 2 * arch.hidden
    emb = arch.vocab * arch.hidden
    head = 0 if arch.tie else arch.vocab * arch.hidden
    total = emb + head + arch.layers * layer + arch.hidden
    linear = arch.layers * (attn + mlp)
    lora_layer = lora_rank * (
        2 * (arch.hidden + arch.heads * arch.head_dim)
        + 2 * (arch.hidden + arch.kv_heads * arch.head_dim)
        + 3 * (arch.hidden + arch.inter)
    )
    return dict(
        attn=attn, mlp=mlp, layer=layer, emb=emb, head=head, total=total, linear=linear, lora=arch.layers * lora_layer
    )


def model_card(model_name: str | None, model_path: str | None, lora_rank: int = 8) -> dict | None:
    r"""Return the architecture plus a few descriptive fields for the model overview."""
    arch, source = find_arch(model_name, model_path)
    if arch is None:
        return None

    card = {
        "arch": arch,
        "source": source,
        "counts": counts(arch, int(lora_rank or 8)),
        "max_pos": None,
        "dtype": None,
    }
    if source and not source.startswith("preset:"):
        cfg = _read_json(source) or {}
        text = cfg["text_config"] if isinstance(cfg.get("text_config"), dict) else cfg
        card["max_pos"] = text.get("max_position_embeddings") or cfg.get("max_position_embeddings")
        card["dtype"] = cfg.get("torch_dtype") or cfg.get("dtype") or text.get("torch_dtype")

    return card
