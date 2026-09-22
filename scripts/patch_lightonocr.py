#!/usr/bin/env python3
"""
Patch LightOnOCR-2 model configs for compatibility with transformers >= 5.1.

LightOnOCR-2 models on HuggingFace Hub ship with ``model_type: "mistral3"`` in
their config.json, but transformers >= 5.1 provides a native ``lighton_ocr``
model type with the correct weight naming.  Without this patch, the vision
encoder weights fail to load (they appear as MISSING in the load report).

Additionally, the processor_config.json stores ``patch_size`` as a bare integer,
which causes thousands of INFO-level log messages during training.

Usage
-----
    # Patch a specific model (downloads if not cached)
    python scripts/patch_lightonocr.py lightonai/LightOnOCR-2-1B-base

    # Patch a local model directory
    python scripts/patch_lightonocr.py /path/to/local/model

    # Patch all cached LightOnOCR-2 models
    python scripts/patch_lightonocr.py --all

Note: This script is idempotent — running it multiple times is safe.
"""

import argparse
import glob
import json
import os
import sys


def _needs_config_patch(data: dict) -> bool:
    architectures = data.get("architectures", [])
    model_type = data.get("model_type", "")
    has_lightonocr = any("lightonocr" in a.lower() or "light_on_ocr" in a.lower() for a in architectures)
    return has_lightonocr and model_type == "mistral3"


def _needs_processor_patch(data: dict) -> bool:
    image_processor = data.get("image_processor", {})
    patch_size = image_processor.get("patch_size")
    return isinstance(patch_size, int)


def patch_config_json(filepath: str) -> bool:
    with open(filepath, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not _needs_config_patch(data):
        return False

    old_type = data["model_type"]
    data["model_type"] = "lighton_ocr"
    data["architectures"] = [
        "LightOnOcrForConditionalGeneration" if a == "LightOnOCRForConditionalGeneration" else a
        for a in data.get("architectures", [])
    ]

    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(f"  [PATCHED] {filepath}")
    print(f"           model_type: '{old_type}' -> 'lighton_ocr'")
    print(f"           architectures: -> 'LightOnOcrForConditionalGeneration'")
    return True


def patch_processor_config_json(filepath: str) -> bool:
    with open(filepath, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not _needs_processor_patch(data):
        return False

    ps = data["image_processor"]["patch_size"]
    data["image_processor"]["patch_size"] = {"height": ps, "width": ps}

    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(f"  [PATCHED] {filepath}")
    print(f"           patch_size: {ps} -> {{'height': {ps}, 'width': {ps}}}")
    return True


def patch_model(model_path: str) -> int:
    """Patch a single model directory or HF hub model.  Returns count of patches applied."""
    patches = 0

    # Resolve path: local dir or HF cache
    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, "config.json")
        processor_path = os.path.join(model_path, "processor_config.json")
    else:
        try:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(model_path, "config.json", local_files_only=True)
            try:
                processor_path = hf_hub_download(model_path, "processor_config.json", local_files_only=True)
            except Exception:
                processor_path = None
        except Exception:
            print(f"  [SKIP] Model '{model_path}' not found in cache. Download it first:")
            print(f"         huggingface-cli download {model_path}")
            return 0

    if os.path.isfile(config_path):
        if patch_config_json(config_path):
            patches += 1
        else:
            print(f"  [OK] {config_path} (already patched or not LightOnOCR)")

    if processor_path and os.path.isfile(processor_path):
        if patch_processor_config_json(processor_path):
            patches += 1
        else:
            print(f"  [OK] {processor_path} (already patched or no fix needed)")

    return patches


def find_all_cached_lightonocr() -> list[str]:
    """Find all LightOnOCR model snapshot dirs in the HF cache."""
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    pattern = os.path.join(cache_dir, "models--*LightOnOCR*", "snapshots", "*")
    return sorted(glob.glob(pattern))


def main():
    parser = argparse.ArgumentParser(
        description="Patch LightOnOCR-2 configs for transformers >= 5.1 compatibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Model name or path (e.g. lightonai/LightOnOCR-2-1B-base or /path/to/model)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Patch all LightOnOCR models found in the HuggingFace cache",
    )
    args = parser.parse_args()

    if not args.model and not args.all:
        parser.print_help()
        sys.exit(1)

    total_patches = 0

    if args.all:
        dirs = find_all_cached_lightonocr()
        if not dirs:
            print("No cached LightOnOCR models found in ~/.cache/huggingface/hub/")
            sys.exit(0)

        print(f"Found {len(dirs)} cached LightOnOCR snapshot(s):\n")
        for d in dirs:
            print(f"--- {d} ---")
            total_patches += patch_model(d)
            print()
    else:
        print(f"--- {args.model} ---")
        total_patches = patch_model(args.model)

    print(f"\nDone. Applied {total_patches} patch(es).")


if __name__ == "__main__":
    main()
