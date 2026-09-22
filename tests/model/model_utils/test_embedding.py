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

import torch

from llamafactory.model.model_utils.embedding import (
    INIT_METHOD_ALIASES,
    _description_initialization,
    _existing_embeddings,
    _make_embedding_freeze_hook,
    _noise_initialization,
    _resolve_new_token_ids,
    _vocab_mean_initialization,
    _vocab_mean_noise_initialization,
)


class _StubTokenizer:
    """Minimal tokenizer stub mapping token strings to fixed IDs."""

    unk_token_id = 0

    def __init__(self, mapping: dict[str, int], desc_ids: list[int] | None = None):
        self._mapping = mapping
        self._desc_ids = desc_ids or []

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._mapping.get(token, self.unk_token_id)

    def __call__(self, desc, return_tensors=None, add_special_tokens=False):
        return {"input_ids": torch.tensor([self._desc_ids], dtype=torch.long)}


class _StubModel:
    """Wraps an embedding matrix so ``get_input_embeddings()`` is a usable lookup."""

    def __init__(self, embed_weight: "torch.Tensor"):
        self._emb = torch.nn.Embedding.from_pretrained(embed_weight.clone(), freeze=True)

    def get_input_embeddings(self):
        return self._emb


def test_resolve_new_token_ids_returns_none_without_config():
    tokenizer = _StubTokenizer({})
    assert _resolve_new_token_ids(None, tokenizer, embed_size=100) is None
    assert _resolve_new_token_ids([], tokenizer, embed_size=100) is None


def test_resolve_new_token_ids_filters_invalid_and_dedups():
    # "<a>" valid, "<unk_like>" maps to unk_token_id (skipped), "<oob>" out of range (skipped)
    tokenizer = _StubTokenizer({"<a>": 10, "<unk_like>": 0, "<oob>": 999, "<b>": 5})
    # duplicates and unsorted input -> sorted unique in-range IDs
    tokens = ["<a>", "<a>", "<unk_like>", "<oob>", "<b>"]
    assert _resolve_new_token_ids(tokens, tokenizer, embed_size=100) == [5, 10]
    # passing a dict iterates its keys (config compatibility)
    assert _resolve_new_token_ids({"<a>": "desc"}, tokenizer, embed_size=100) == [10]


def test_existing_embeddings_excludes_new_token_ids():
    embed_weight = torch.arange(10 * 2, dtype=torch.float32).reshape(10, 2)
    # explicit ids take precedence and drop exactly those rows
    existing = _existing_embeddings(embed_weight, num_new_tokens=3, new_token_ids=[2, 5])
    assert existing.size(0) == 8
    # tail fallback when no explicit ids
    tail = _existing_embeddings(embed_weight, num_new_tokens=3, new_token_ids=None)
    assert torch.allclose(tail, embed_weight[:-3])
    # no resize and no ids -> use everything
    everything = _existing_embeddings(embed_weight, num_new_tokens=0, new_token_ids=None)
    assert torch.allclose(everything, embed_weight)


def test_vocab_mean_noise_initialization_with_token_ids_targets_exact_rows():
    """New tokens placed by explicit IDs must hit those rows, even inside the padding zone."""
    torch.manual_seed(0)
    vocab_size, embedding_dim = 20, 8
    embed_weight = torch.zeros(vocab_size, embedding_dim)
    # existing rows carry a constant so the mean is well-defined and non-zero
    embed_weight[:16] = 1.0

    # num_new_tokens reflects the embedding resize delta (4 padded rows),
    # but the real new tokens sit at IDs 16 and 17 (inside what the tail slice would miss/over-cover).
    target_ids = [16, 17]
    _vocab_mean_noise_initialization(embed_weight, num_new_tokens=4, token_ids=target_ids)

    # targeted rows are initialized around the mean (~1.0) and not left at zero
    for tid in target_ids:
        assert not torch.allclose(embed_weight[tid], torch.zeros(embedding_dim))
        assert abs(embed_weight[tid].mean().item() - 1.0) < 0.5

    # untouched padding rows (18, 19) must remain zero
    assert torch.allclose(embed_weight[18], torch.zeros(embedding_dim))
    assert torch.allclose(embed_weight[19], torch.zeros(embedding_dim))


def test_vocab_mean_noise_initialization_tail_fallback():
    """Without token_ids, falls back to the last num_new_tokens rows."""
    torch.manual_seed(0)
    vocab_size, embedding_dim = 12, 8
    embed_weight = torch.zeros(vocab_size, embedding_dim)
    embed_weight[:10] = 1.0

    _vocab_mean_noise_initialization(embed_weight, num_new_tokens=2, token_ids=None)

    # last two rows initialized, earlier rows untouched
    assert not torch.allclose(embed_weight[-1], torch.zeros(embedding_dim))
    assert not torch.allclose(embed_weight[-2], torch.zeros(embedding_dim))
    assert torch.allclose(embed_weight[0], torch.ones(embedding_dim))


def test_noise_initialization_with_token_ids():
    """Pure-noise init writes only the targeted rows and matches the existing std scale."""
    torch.manual_seed(0)
    vocab_size, embedding_dim = 64, 16
    embed_weight = torch.zeros(vocab_size, embedding_dim)
    embed_weight[:60].normal_(mean=0.0, std=0.02)

    _noise_initialization(embed_weight, num_new_tokens=4, token_ids=[60, 61])

    for tid in (60, 61):
        assert not torch.allclose(embed_weight[tid], torch.zeros(embedding_dim))
    # untouched rows stay zero
    assert torch.allclose(embed_weight[62], torch.zeros(embedding_dim))
    assert torch.allclose(embed_weight[63], torch.zeros(embedding_dim))


def test_vocab_mean_initialization_assigns_identical_mean():
    """Pure-mean init assigns the same mean vector to every targeted row."""
    vocab_size, embedding_dim = 16, 8
    # All rows are existing (2.0) except the two new tokens, which are excluded from the baseline.
    embed_weight = torch.full((vocab_size, embedding_dim), 2.0)
    embed_weight[12] = 0.0
    embed_weight[13] = 0.0

    _vocab_mean_initialization(embed_weight, num_new_tokens=4, token_ids=[12, 13])

    expected = torch.full((embedding_dim,), 2.0)
    assert torch.allclose(embed_weight[12], expected)
    assert torch.allclose(embed_weight[13], expected)
    assert torch.allclose(embed_weight[12], embed_weight[13])


def test_description_init_excludes_new_token_ids_from_average():
    """Description tokens that are themselves new (uninitialized) must be excluded.

    Reproduces the padding-zone bug: id 17 is a new token and must not pollute the
    semantic average for id 16; only the valid existing token (id 5) should be used.
    """
    vocab_size, embedding_dim = 20, 4
    embed_weight = torch.zeros(vocab_size, embedding_dim)
    embed_weight[5] = 3.0  # the only valid description token

    # description for "<x>" tokenizes to [5 (existing), 17 (new -> must be skipped)]
    tokenizer = _StubTokenizer({"<x>": 16}, desc_ids=[5, 17])
    model = _StubModel(embed_weight)

    _description_initialization(
        embed_weight,
        num_new_tokens=4,
        descriptions={"<x>": "ignored, ids come from the stub"},
        tokenizer=tokenizer,
        model=model,
        new_token_ids=[16, 17],
        add_noise=False,
    )

    # row 16 must equal embedding of id 5 only (3.0), not the (5,17) average (1.5)
    assert torch.allclose(embed_weight[16], torch.full((embedding_dim,), 3.0))


def test_embedding_freeze_hook_zeros_original_rows():
    """The freeze hook must zero the gradient of original rows and keep new rows intact."""
    vocab_size, embedding_dim, orig_size = 8, 4, 6
    embed = torch.nn.Embedding(vocab_size, embedding_dim)
    embed.weight.register_hook(_make_embedding_freeze_hook(orig_size, vocab_size))

    # touch every row so each gets a gradient, then backprop
    embed(torch.arange(vocab_size)).sum().backward()

    grad = embed.weight.grad
    assert torch.allclose(grad[:orig_size], torch.zeros(orig_size, embedding_dim))  # frozen
    assert torch.all(grad[orig_size:] != 0.0)  # new rows still trainable


def test_embedding_freeze_hook_passes_through_sharded_gradient():
    """A partitioned/flattened gradient (ZeRO-3 / FSDP) is returned unchanged, never crashes."""
    vocab_size, embedding_dim, orig_size = 8, 4, 6
    hook = _make_embedding_freeze_hook(orig_size, vocab_size)

    # 1D flat shard (FSDP-style) and a 2D shard with the wrong row count are both passed through
    flat_shard = torch.ones(vocab_size * embedding_dim // 2)
    assert torch.equal(hook(flat_shard), flat_shard)

    row_shard = torch.ones(vocab_size // 2, embedding_dim)
    assert torch.equal(hook(row_shard), row_shard)


def test_legacy_init_method_aliases():
    """Legacy init-method names map to their canonical equivalents."""
    assert INIT_METHOD_ALIASES == {
        "noise_init": "vocab_mean_noise",
        "desc_init": "description",
        "desc_init_w_noise": "description_noise",
    }


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
