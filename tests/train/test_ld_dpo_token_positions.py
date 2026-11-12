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

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, Seq2SeqTrainingArguments

from llamafactory.data.mm_plugin import MossVLPlugin
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import FinetuningArguments
from llamafactory.train.dpo.trainer import CustomDPOTrainer
from llamafactory.train.trainer_utils import get_batch_logps


@pytest.fixture(autouse=True)
def _restore_rng():
    with torch.random.fork_rng():
        torch.manual_seed(42)
        yield


def _reference_logps(logits, labels, alpha):
    """Weight the ordered list of supervised tokens, independent of their offsets."""
    log_probs = logits[:, :-1].log_softmax(-1)
    targets = labels[:, 1:]
    tokens = [
        [log_probs[row, column, label] for column, label in enumerate(sequence) if label != IGNORE_INDEX]
        for row, sequence in enumerate(targets)
    ]
    pair_count = len(tokens) // 2
    scores = []
    for row, sequence in enumerate(tokens):
        pair = row % pair_count
        shared = min(len(tokens[pair]), len(tokens[pair + pair_count]))
        scores.append(
            sum(score * (1 if alpha is None or index < shared else alpha) for index, score in enumerate(sequence))
        )
    return torch.stack(scores)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("alpha", [None, 0.0, 0.25, 1.0])
@pytest.mark.parametrize(
    "labels",
    [
        [[-100, 1, 2, 3, 4, -100, -100], [-100, 1, 4, -100, -100, -100, -100]],
        [[-100, 1, -100, 2, 3, -100, 4], [-100, 1, -100, 4, -100, -100, -100]],
        [[-100, -100, 1, -100, 2, 3, 4], [-100, -100, -100, 1, -100, 4, -100]],
    ],
    ids=["contiguous", "internal-masks", "different-prefix-lengths"],
)
def test_ld_dpo_counts_supervised_token_positions(labels, alpha, dtype):
    labels = torch.tensor(labels)
    original_labels = labels.clone()
    logits = torch.randn(*labels.shape, 7, dtype=dtype, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_()

    actual, lengths = get_batch_logps(logits, labels, ld_alpha=alpha)
    expected = _reference_logps(reference_logits, labels, alpha)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(lengths, torch.tensor([4, 2]))
    torch.testing.assert_close(labels, original_labels)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad)


@pytest.mark.parametrize("pair_count", [1, 2])
@pytest.mark.parametrize("alpha", [0.0, 0.3])
def test_ld_dpo_trainer_update_with_internal_masks(tmp_path, pair_count, alpha):
    config = GPT2Config(
        vocab_size=7,
        n_positions=8,
        n_embd=8,
        n_layer=1,
        n_head=2,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
    )
    policy = GPT2LMHeadModel(config)
    expected_policy = deepcopy(policy)
    reference = deepcopy(policy)
    with torch.no_grad():
        reference.transformer.wte.weight.mul_(0.7)
    labels = torch.tensor(
        [[-100, 1, -100, 2, 3, -100, 4]] * pair_count + [[-100, 1, -100, 4, -100, -100, -100]] * pair_count
    )
    model_inputs = {
        "input_ids": labels.masked_fill(labels == IGNORE_INDEX, 0),
        "attention_mask": torch.ones_like(labels),
    }
    batch = {**model_inputs, "labels": labels}
    finetuning_args = FinetuningArguments(stage="dpo", pref_loss="sigmoid", ld_alpha=alpha)
    trainer = CustomDPOTrainer(
        model=policy,
        ref_model=reference,
        finetuning_args=finetuning_args,
        processor=None,
        tokenizer=None,
        args=Seq2SeqTrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to="none"),
    )
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.05)
    reference_optimizer = torch.optim.SGD(expected_policy.parameters(), lr=0.05)

    for _ in range(2):
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        actual, _ = trainer.get_batch_loss_metrics(policy, batch)
        expected_scores = _reference_logps(expected_policy(**model_inputs).logits, labels, alpha)
        with torch.no_grad():
            reference_scores = _reference_logps(reference(**model_inputs).logits, labels, None)
        policy_margin = expected_scores[:pair_count] - expected_scores[pair_count:]
        reference_margin = reference_scores[:pair_count] - reference_scores[pair_count:]
        expected = -F.logsigmoid(finetuning_args.pref_beta * (policy_margin - reference_margin)).mean()

        torch.testing.assert_close(actual, expected)
        actual.backward()
        expected.backward()
        for parameter, expected_parameter in zip(policy.parameters(), expected_policy.parameters()):
            torch.testing.assert_close(parameter.grad, expected_parameter.grad)
        optimizer.step()
        reference_optimizer.step()
        for parameter, expected_parameter in zip(policy.parameters(), expected_policy.parameters()):
            torch.testing.assert_close(parameter, expected_parameter)


def test_ld_dpo_with_moss_vl_control_token_masks():
    vocabulary = [
        "<pad>",
        "<unk>",
        "<image>",
        "<video>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|time_start|>",
        "<|time_end|>",
        "<|im_end|>",
        "\n",
        "first",
        "second",
        "third",
    ]
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(dict(zip(vocabulary, range(len(vocabulary)))), unk_token="<unk>")),
        pad_token="<pad>",
        unk_token="<unk>",
    )
    processor = SimpleNamespace(
        tokenizer=tokenizer, image_token_id=2, video_token_id=3, vision_start_token_id=4, vision_end_token_id=5
    )
    plugin = MossVLPlugin(image_token="<image>", video_token="<video>", audio_token=None)
    input_ids = torch.tensor([[0, 10, 4, 2, 5, 11, 12, 8, 9], [0, 10, 4, 2, 5, 11, 8, 9, 0]])
    labels = input_ids.masked_fill(input_ids == 0, IGNORE_INDEX)
    features = {"input_ids": input_ids, "attention_mask": input_ids != 0, "labels": labels}
    plugin.post_process_mossvl_inputs(
        features, {"grid_thw": torch.ones(2, 3, dtype=torch.long), "media_nums_per_sample": [1, 1]}, processor
    )
    torch.testing.assert_close(
        labels,
        torch.tensor([[-100, 10, -100, -100, -100, 11, 12, 8, -100], [-100, 10, -100, -100, -100, 11, 8, -100, -100]]),
    )
    logits = torch.randn(*labels.shape, len(vocabulary))
    actual, lengths = get_batch_logps(logits, labels, ld_alpha=0.25)
    torch.testing.assert_close(actual, _reference_logps(logits, labels, 0.25))
    torch.testing.assert_close(lengths, torch.tensor([4, 3]))
