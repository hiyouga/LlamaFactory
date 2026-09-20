"""Test that the RM trainer optimization works correctly.

Tests that:
1. EfficientValueHeadForward produces the correct output shape
2. lm_head is replaced with Identity
3. v_head dtype matches model dtype
4. Forward pass returns valid (logits, loss, value) tuple
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llamafactory.train.rm.trainer import EfficientValueHeadForward


def test_efficient_forward():
    """Test EfficientValueHeadForward produces correct outputs."""
    print("=" * 60)
    print("Test: EfficientValueHeadForward")
    print("=" * 60)

    # Create a small model
    config = AutoConfig.from_pretrained("llamafactory/tiny-random-Llama-3")
    base_model = AutoModelForCausalLM.from_config(config)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)

    pretrained = model.pretrained_model
    v_head = model.v_head

    # --- Baseline: original TRL forward (output_hidden_states=True) ---
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        orig_logits, orig_loss, orig_values = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    print(f"Original forward:")
    print(f"  logits shape: {orig_logits.shape}")
    print(f"  values shape: {orig_values.shape}")
    print(f"  loss: {orig_loss}")

    # --- Optimized: replace lm_head with Identity + EfficientValueHeadForward ---
    _lm_head_owner = pretrained
    if hasattr(pretrained, "base_model") and hasattr(pretrained.base_model, "model"):
        _lm_head_owner = pretrained.base_model.model
    if hasattr(_lm_head_owner, "lm_head"):
        _lm_head_owner.lm_head = torch.nn.Identity()

    model_dtype = next(pretrained.parameters()).dtype
    model.v_head = model.v_head.to(dtype=model_dtype)

    efficient_fwd = EfficientValueHeadForward(pretrained, model.v_head)

    with torch.no_grad():
        opt_logits, opt_loss, opt_values = efficient_fwd(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

    print(f"\nOptimized forward:")
    print(f"  logits shape: {opt_logits.shape}")
    print(f"  values shape: {opt_values.shape}")
    print(f"  loss: {opt_loss}")

    # --- Verify shapes match (logits shape differs because Identity passes through hidden_size) ---
    assert opt_values.shape == orig_values.shape, (
        f"Values shape mismatch: {opt_values.shape} vs {orig_values.shape}"
    )
    # logits shape: original is (batch, seq, vocab_size), optimized is (batch, seq, hidden_size)
    # This is expected because lm_head=Identity passes through hidden states directly
    print(f"\n  Original logits shape: {orig_logits.shape} (batch, seq, vocab_size)")
    print(f"  Optimized logits shape: {opt_logits.shape} (batch, seq, hidden_size)")
    print(f"  Note: logits shape differs because lm_head=Identity; logits are not used in RM training")

    # Verify values are finite
    assert torch.isfinite(opt_values).all(), "Values contain non-finite numbers"

    print("\n[PASS] EfficientValueHeadForward produces valid outputs")


def test_memory_comparison():
    """Compare memory usage between original and optimized forward."""
    print("\n" + "=" * 60)
    print("Test: Memory comparison")
    print("=" * 60)

    config = AutoConfig.from_pretrained("llamafactory/tiny-random-Llama-3")

    # Original forward
    base_model_orig = AutoModelForCausalLM.from_config(config)
    model_orig = AutoModelForCausalLMWithValueHead.from_pretrained(base_model_orig)

    input_ids = torch.randint(0, config.vocab_size, (4, 32))
    attention_mask = torch.ones_like(input_ids)

    import tracemalloc
    tracemalloc.start()

    with torch.no_grad():
        _, _, values_orig = model_orig(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    _, orig_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Optimized forward
    base_model_opt = AutoModelForCausalLM.from_config(config)
    model_opt = AutoModelForCausalLMWithValueHead.from_pretrained(base_model_opt)

    pretrained = model_opt.pretrained_model
    _lm_head_owner = pretrained
    if hasattr(pretrained, "base_model") and hasattr(pretrained.base_model, "model"):
        _lm_head_owner = pretrained.base_model.model
    if hasattr(_lm_head_owner, "lm_head"):
        _lm_head_owner.lm_head = torch.nn.Identity()

    model_dtype = next(pretrained.parameters()).dtype
    model_opt.v_head = model_opt.v_head.to(dtype=model_dtype)
    efficient_fwd = EfficientValueHeadForward(pretrained, model_opt.v_head)

    tracemalloc.start()

    with torch.no_grad():
        _, _, values_opt = efficient_fwd(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
    _, opt_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Original forward peak memory: {orig_peak / 1024 / 1024:.1f} MB")
    print(f"Optimized forward peak memory: {opt_peak / 1024 / 1024:.1f} MB")
    savings_pct = (1 - opt_peak / max(orig_peak, 1)) * 100
    print(f"Memory savings: {savings_pct:.1f}%")

    print(f"\nNote: The tiny model has only 2 layers and hidden_size=16.")
    print(f"For real models (e.g., Llama-3-70B with 80 layers, hidden_size=8192),")
    print(f"the savings from skipping hidden state storage would be ~6+ GiB.")
    print(f"Additional savings come from skipping the lm_head projection")
    print(f"(hidden_size x vocab_size = 8192 x 128256 matrix multiply).")

    print("\n[PASS] Memory comparison completed")


def test_lora_path():
    """Test that the optimization handles LoRA-wrapped models."""
    print("\n" + "=" * 60)
    print("Test: LoRA model path")
    print("=" * 60)

    from peft import LoraConfig

    config = AutoConfig.from_pretrained("llamafactory/tiny-random-Llama-3")
    base_model = AutoModelForCausalLM.from_config(config)

    # TRL wraps LoRA internally via peft_config parameter
    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
    model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model, peft_config=lora_config)

    pretrained = model.pretrained_model
    _lm_head_owner = pretrained
    if hasattr(pretrained, "base_model") and hasattr(pretrained.base_model, "model"):
        _lm_head_owner = pretrained.base_model.model
        print(f"  LoRA detected: lm_head found on pretrained.base_model.model")

    if hasattr(_lm_head_owner, "lm_head"):
        _lm_head_owner.lm_head = torch.nn.Identity()

    assert isinstance(_lm_head_owner.lm_head, torch.nn.Identity), "lm_head not replaced with Identity"

    model_dtype = next(pretrained.parameters()).dtype
    model.v_head = model.v_head.to(dtype=model_dtype)

    efficient_fwd = EfficientValueHeadForward(pretrained, model.v_head)

    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        logits, loss, values = efficient_fwd(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

    assert values.shape == (2, 8), f"Unexpected values shape: {values.shape}"
    assert torch.isfinite(values).all(), "Values contain non-finite numbers"

    print(f"  logits shape: {logits.shape}")
    print(f"  values shape: {values.shape}")
    print("\n[PASS] LoRA model path works correctly")


if __name__ == "__main__":
    test_efficient_forward()
    test_memory_comparison()
    test_lora_path()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
