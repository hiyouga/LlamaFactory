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

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from llamafactory.hparams import DataArguments, FinetuningArguments, ModelArguments, TrainingArguments
from llamafactory.train.decision.data import DecisionDataCollator, load_decision_dataset
from llamafactory.train.decision.trainer import DecisionTrainer, decision_loss
from llamafactory.train.decision.workflow import validate_decision_args


# Stand-in for the request schema exported by a native/remote GS1 compiler.
class DecisionRequest:
    @classmethod
    def model_validate(cls, value):
        return SimpleNamespace(**value)


class Compiler:
    def compile(self, request):
        branches = []
        for qid, question in request.questions.items():
            candidates = tuple(question["criteria"]) if question["type"] == "choice" else ("false", "true")
            branches.append(
                SimpleNamespace(question_id=qid, candidates=candidates, suffix=(21, 22, 23), visual_inputs=None)
            )
        return SimpleNamespace(state=(11, 12), branches=branches)


def record():
    return {
        "id": "sample",
        "request": {
            "model": "GS1",
            "state": "A customer requests a refund.",
            "questions": {
                "team": {"type": "choice", "criteria": {"billing": "Payments", "technical": "Bugs"}},
                "refund": {"type": "noul"},
            },
        },
        "targets": {
            "team": {"candidate_ids": ["technical", "billing"], "distribution": [0.25, 0.75]},
            "refund": {"candidate_ids": ["true", "false"], "distribution": [1.0, 0.0]},
        },
        "provenance": {"sample_weight": 0.5, "question_loss_weights": {"team": 0.8}},
    }


def dataset_args(tmp_path, records, **kwargs):
    (tmp_path / "native.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    (tmp_path / "dataset_info.json").write_text(
        json.dumps({"native": {"file_name": "native.jsonl", "formatting": "gs1"}})
    )
    return DataArguments(dataset="native", dataset_dir=str(tmp_path), **kwargs)


def test_native_records_preserve_soft_targets_order_and_record_limit(tmp_path):
    source = record()
    data_args = dataset_args(tmp_path, [source, source], max_samples=1)
    examples = load_decision_dataset(data_args.dataset, data_args, Compiler())
    assert len(examples) == 2  # One record contains two logical decisions.
    assert examples[0]["labels"] == [0.75, 0.25]
    assert examples[1]["labels"] == [0.0, 1.0]
    assert examples[0]["weight"] == 1.0
    assert examples[1]["weight"] == 1.0
    assert examples[0]["input_ids"] == [11, 12, 21, 22, 23]


@pytest.mark.parametrize(
    "target",
    [
        {"candidate_ids": ["billing", "billing"], "distribution": [0.5, 0.5]},
        {"candidate_ids": ["billing", "other"], "distribution": [0.5, 0.5]},
        {"candidate_ids": ["billing", "technical"], "distribution": [0.8]},
        {"candidate_ids": ["billing", "technical"], "distribution": [0.1, 0.1]},
        {"candidate_ids": ["billing", "technical"], "distribution": [-0.1, 1.1]},
        {"candidate_ids": ["billing", "technical"], "distribution": [float("nan"), 0.0]},
    ],
)
def test_invalid_targets_fail_with_record_context(tmp_path, target):
    row = record()
    row["targets"]["team"] = target
    data_args = dataset_args(tmp_path, [row])
    with pytest.raises(ValueError, match="Invalid GS1 record sample"):
        load_decision_dataset(data_args.dataset, data_args, Compiler())


def test_context_errors_are_not_silently_truncated(tmp_path):
    data_args = dataset_args(tmp_path, [record()])
    compiler = Compiler()
    compiler.compile = Mock(side_effect=ValueError("Branch exceeds 16 tokens; no truncation"))
    with pytest.raises(ValueError, match="no truncation"):
        load_decision_dataset(data_args.dataset, data_args, compiler)


@pytest.fixture
def tokenizer():
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({str(i): i for i in range(32)}, unk_token="0")),
        pad_token="0",
        padding_side="left",
    )


@pytest.mark.parametrize("pad_token_id", [0, 7])
def test_collator_left_padding_positions_and_candidate_padding(tokenizer, pad_token_id):
    tokenizer.pad_token = str(pad_token_id)
    features = [
        {"input_ids": [10, pad_token_id], "labels": [0.3, 0.7], "weight": 1.0, "branch": "a", "state_length": 1},
        {"input_ids": [12, 13, 14], "labels": [0.0, 0.0, 1.0], "weight": 0.5, "branch": "b", "state_length": 2},
    ]
    batch = DecisionDataCollator(tokenizer=tokenizer)(features)
    assert batch["input_ids"].tolist() == [[pad_token_id, 10, pad_token_id], [12, 13, 14]]
    assert batch["attention_mask"].tolist() == [[0, 1, 1], [1, 1, 1]]
    assert batch["position_ids"].tolist() == [[0, 0, 1], [0, 1, 2]]
    assert batch["branches"] == ["a", "b"]
    assert batch["state_lengths"] == [1, 2]
    torch.testing.assert_close(batch["labels"], torch.tensor([[0.3, 0.7, 0.0], [0.0, 0.0, 1.0]]))


@pytest.mark.parametrize("brier_weight", [0.0, 0.5, 1.0])
def test_soft_distribution_loss_and_padded_gradients(brier_weight):
    logits = torch.tensor([[0.3, -0.2, -torch.inf], [0.4, 0.0, -0.3]], requires_grad=True)
    labels = torch.tensor([[0.25, 0.75, 0.0], [0.0, 1.0, 0.0]])
    weights = torch.tensor([0.5, 1.0])
    expected = []
    for i, width in enumerate([2, 3]):
        lp = logits[i, :width].detach().log_softmax(-1)
        target = labels[i, :width]
        brier = (lp.exp().square() - 2 * target * lp.exp() + target).sum()
        expected.append((-(target * lp).sum() + brier_weight * brier) * weights[i])
    loss = decision_loss(logits, labels, weights, brier_weight)
    torch.testing.assert_close(loss.detach(), torch.stack(expected).mean())
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 2] == 0


def test_trainer_uses_wrapped_forward_without_mutating_inputs_or_returning_hidden_states():
    trainer = object.__new__(DecisionTrainer)
    trainer.finetuning_args = FinetuningArguments(stage="decision", finetuning_type="full")
    model = Mock(return_value=SimpleNamespace(logits=torch.tensor([[0.0, 1.0]]), last_hidden_state="large"))
    inputs = {
        "input_ids": torch.tensor([[2, 3]]),
        "labels": torch.tensor([[0.0, 1.0]]),
        "question_weights": torch.ones(1),
    }
    loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)
    assert torch.isfinite(loss)
    assert "labels" in inputs and "question_weights" in inputs
    assert set(outputs) == {"logits"}
    assert set(model.call_args.kwargs) == {"input_ids"}


def test_decision_arguments_and_rejected_chat_settings(tmp_path):
    model_args = ModelArguments(model_name_or_path="bigai/GS1-0.8B")
    data_args = DataArguments()
    training_args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to="none")
    fine_args = FinetuningArguments(stage="decision", finetuning_type="full")
    validate_decision_args(model_args, data_args, training_args, fine_args)
    for obj, name, value in [
        (fine_args, "finetuning_type", "lora"),
        (data_args, "packing", True),
        (data_args, "template", "qwen3"),
        (training_args, "do_predict", True),
        (fine_args, "decision_brier_weight", -1.0),
        (fine_args, "decision_brier_weight", float("nan")),
    ]:
        changed = deepcopy(obj)
        setattr(changed, name, value)
        args = [model_args, data_args, training_args, fine_args]
        args[next(i for i, original in enumerate(args) if original is obj)] = changed
        with pytest.raises(ValueError):
            validate_decision_args(*args)


def test_native_processor_assets_are_saved_without_rewriting(tmp_path):
    from llamafactory.train.decision.workflow import DecisionProcessor

    source, output = tmp_path / "source", tmp_path / "checkpoint-1"
    source.mkdir()
    for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
        (source / name).write_text('{"native": true}\n')
    DecisionProcessor(source).save_pretrained(output)
    for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
        assert (output / name).read_bytes() == (source / name).read_bytes()


@pytest.mark.parametrize("fail_loading", [False, True])
def test_native_loading_defers_zero3_and_restores_config(fail_loading):
    from transformers.integrations.deepspeed import (
        is_deepspeed_zero3_enabled,
        set_hf_deepspeed_config,
        unset_hf_deepspeed_config,
    )

    from llamafactory.train.decision.workflow import native_load_context

    config = Mock()
    config.is_zero3.return_value = True
    args = SimpleNamespace(deepspeed_plugin=SimpleNamespace(hf_ds_config=config))
    set_hf_deepspeed_config(config)
    try:
        try:
            with native_load_context(args):
                assert not is_deepspeed_zero3_enabled()
                if fail_loading:
                    raise RuntimeError("native loader failed")
        except RuntimeError:
            assert fail_loading
        assert is_deepspeed_zero3_enabled()
    finally:
        unset_hf_deepspeed_config()


def test_decision_accepts_standard_distributed_backends(tmp_path):
    model_args = ModelArguments(model_name_or_path="bigai/GS1-0.8B")
    data_args = DataArguments()
    fine_args = FinetuningArguments(stage="decision", finetuning_type="full")
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to="none")
    # Backend construction belongs to TrainingArguments/Accelerate, not this stage.
    args.deepspeed = "examples/deepspeed/ds_z3_config.json"
    validate_decision_args(model_args, data_args, args, fine_args)
    args.deepspeed = None
    args.fsdp = ["full_shard", "auto_wrap"]
    validate_decision_args(model_args, data_args, args, fine_args)


def test_standard_tuner_dispatches_decision_stage(monkeypatch, tmp_path):
    from llamafactory.train import decision, tuner

    parsed = (
        ModelArguments(model_name_or_path="GS1"),
        DataArguments(),
        TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to="none"),
        FinetuningArguments(stage="decision", finetuning_type="full"),
        SimpleNamespace(),
    )
    monkeypatch.setattr(tuner, "get_train_args", lambda args: parsed)
    run = Mock()
    monkeypatch.setattr(decision, "run_decision", run)
    callbacks = []
    tuner._training_function({"args": {}, "callbacks": callbacks})
    run.assert_called_once_with(*parsed[:4], callbacks)


@pytest.mark.parametrize("fail_saving", [False, True])
def test_fsdp2_export_preserves_native_class_and_restores_wrapper(monkeypatch, tmp_path, fail_saving):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    from llamafactory.train.decision.trainer import CustomSeq2SeqTrainer

    class NativeModel(torch.nn.Module):
        _auto_class = "AutoModel"

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))

    model = NativeModel()
    dist.init_process_group("gloo", init_method=(tmp_path / "rendezvous").as_uri(), rank=0, world_size=1)
    try:
        fully_shard(model, mesh=init_device_mesh("cpu", (1,)))
    finally:
        dist.destroy_process_group()
    wrapper_class = type(model)
    assert wrapper_class is not NativeModel
    trainer = object.__new__(DecisionTrainer)
    trainer.model = model

    def save(self, output_dir, state_dict):
        assert type(self.model) is NativeModel  # HF custom_object_save inspects this class.
        if fail_saving:
            raise OSError("disk full")

    monkeypatch.setattr(CustomSeq2SeqTrainer, "_save", save)
    if fail_saving:
        with pytest.raises(OSError, match="disk full"):
            trainer._save(str(tmp_path), state_dict={})
    else:
        trainer._save(str(tmp_path), state_dict={})
    assert type(model) is wrapper_class


def test_workflow_exports_native_model_after_sharded_training(monkeypatch, tmp_path):
    from llamafactory.train.decision import workflow

    plugin = SimpleNamespace(state_dict_type="SHARDED_STATE_DICT")
    plugin.set_state_dict_type = lambda value: setattr(plugin, "state_dict_type", value)
    trainer = Mock(is_fsdp_enabled=True)
    trainer.accelerator.state.fsdp_plugin = plugin

    def train(**kwargs):
        assert plugin.state_dict_type == "SHARDED_STATE_DICT"
        return SimpleNamespace(metrics={})

    def save():
        # Trainer.save_model silently skips export with a sharded state dict.
        assert plugin.state_dict_type == "FULL_STATE_DICT"
        (tmp_path / "config.json").write_text('{"model_type": "gs1"}')

    trainer.train.side_effect = train
    trainer.save_model.side_effect = save
    model = Mock()
    model.named_parameters.return_value = []
    monkeypatch.setattr(workflow, "check_version", Mock())
    monkeypatch.setattr(workflow, "load_config", lambda args: SimpleNamespace(model_type="gs1"))
    monkeypatch.setattr(workflow, "load_tokenizer", lambda args: {"tokenizer": Mock(), "processor": Mock()})
    monkeypatch.setattr(workflow.AutoModel, "from_pretrained", Mock(return_value=model))
    monkeypatch.setattr(workflow, "load_decision_dataset", Mock(return_value=[]))
    monkeypatch.setattr(workflow, "DecisionTrainer", Mock(return_value=trainer))
    workflow.run_decision(
        ModelArguments(model_name_or_path="GS1"),
        DataArguments(dataset="native"),
        TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to="none", do_train=True),
        FinetuningArguments(stage="decision", finetuning_type="full"),
    )
    assert json.loads((tmp_path / "config.json").read_text())["model_type"] == "gs1"
    trainer.save_model.assert_called_once()


def test_train_loop_accumulates_decision_gradients_on_cpu(tmp_path, tokenizer):
    from transformers import PretrainedConfig

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = PretrainedConfig()
            self.weight = torch.nn.Parameter(torch.zeros(2))

        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(logits=self.weight.unsqueeze(0).expand(input_ids.shape[0], -1))

    dataset = [
        {"input_ids": [1, 2], "branch": None, "state_length": 1, "labels": [0.0, 1.0], "weight": 1.0} for _ in range(4)
    ]
    model = Model()
    trainer = DecisionTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            report_to="none",
            max_steps=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=0.01,
            save_strategy="no",
            disable_tqdm=True,
        ),
        finetuning_args=FinetuningArguments(stage="decision", finetuning_type="full"),
        train_dataset=dataset,
        data_collator=DecisionDataCollator(tokenizer=tokenizer),
        tokenizer=None,
        processor=None,
    )
    assert trainer.model_accepts_loss_kwargs is True
    trainer.train()
    assert model.weight[1] > model.weight[0]


class TinyDecisionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = torch.nn.Module()
        self.language_model.embed_tokens = torch.nn.Embedding(32, 4)
        self.language_model.layer = torch.nn.Linear(4, 4)
        self.language_model.attnres_routers = torch.nn.Linear(4, 4, bias=False)
        self.decision_head = torch.nn.Linear(4, 2)
        self.calibrator = torch.nn.Linear(2, 2)

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def forward(self, input_ids, **kwargs):
        hidden = self.language_model.layer(self.get_input_embeddings()(input_ids))
        return SimpleNamespace(logits=self.decision_head(self.language_model.attnres_routers(hidden).mean(1)))


@pytest.mark.parametrize("warmup", [False, True])
def test_native_recipe_parameter_groups_and_structural_updates(tmp_path, warmup):
    from llamafactory.train.decision.workflow import configure_decision_model

    model = TinyDecisionModel()
    args = TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        report_to="none",
        do_train=True,
        learning_rate=2e-4 if warmup else 2e-5,
        weight_decay=0.01,
    )
    fine = FinetuningArguments(
        stage="decision", finetuning_type="full", decision_warmup=warmup, decision_brier_weight=0.0 if warmup else 1.0
    )
    config = SimpleNamespace(structural_token_ids={str(i): i for i in range(25, 32)})
    configure_decision_model(model, config, args, fine)
    trainer = DecisionTrainer(model=model, args=args, finetuning_args=fine, tokenizer=None, processor=None)
    optimizer = trainer.create_optimizer(model)
    ownership = {id(p): group for group in optimizer.param_groups for p in group["params"]}
    for name, parameter in model.named_parameters():
        expected = not name.startswith("calibrator.") and (
            not warmup or "decision_head." in name or "attnres_routers." in name or "embed_tokens." in name
        )
        assert parameter.requires_grad == expected
        if expected:
            group = ownership[id(parameter)]
            ratio = 0.2 if warmup and "embed_tokens." in name else 0.5 if name.startswith("language_model.") else 1.0
            assert group["lr"] == pytest.approx(args.learning_rate * ratio)
            assert group["weight_decay"] == (
                0.0 if "attnres_routers." in name or warmup and "embed_tokens." in name else 0.01
            )
    before = model.get_input_embeddings().weight.detach().clone()
    for _ in range(2):
        loss = model(torch.tensor([[1, 25], [2, 26]])).logits.square().sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    if warmup:
        torch.testing.assert_close(model.get_input_embeddings().weight[:25], before[:25], rtol=0, atol=0)
        assert not torch.equal(model.get_input_embeddings().weight[25:27], before[25:27])


def test_accumulated_loss_counts_questions_and_ignores_zero_loss_fillers():
    trainer = object.__new__(DecisionTrainer)
    trainer.finetuning_args = FinetuningArguments(stage="decision", finetuning_type="full")
    trainer.args = SimpleNamespace(world_size=2)
    trainer.is_deepspeed_enabled = False
    trainer.accelerator = SimpleNamespace(
        gradient_accumulation_steps=1, gather=lambda count: torch.stack([count, count])
    )
    batches = [
        {"labels": torch.zeros(3, 2), "question_weights": torch.tensor([1.0, 1.0, 0.0])},
        {"labels": torch.zeros(1, 5), "question_weights": torch.ones(1)},
    ]
    assert trainer._get_num_items_in_batch(batches, "cpu").item() == 6
    model = Mock(return_value=SimpleNamespace(logits=torch.zeros(3, 2)))
    inputs = {"labels": torch.tensor([[1.0, 0.0]] * 3), "question_weights": torch.tensor([1.0, 1.0, 0.0])}
    loss = trainer.compute_loss(model, inputs, num_items_in_batch=torch.tensor(6))
    torch.testing.assert_close(loss, (torch.log(torch.tensor(2.0)) + 0.5) * 2 / 3)


def test_preparation_preserves_native_stage_membership_and_rejects_overlap(tmp_path):
    import runpy
    from pathlib import Path

    prepare = runpy.run_path(str(Path(__file__).resolve().parents[2] / "examples/train_full/prepare_gs1_decision.py"))[
        "prepare"
    ]
    sample = tmp_path / "sample.jsonl"
    sample.write_text(json.dumps(record()) + "\n")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for filename, qid in (("warmup_plan.jsonl", "team"), ("sft_plan.jsonl", "refund")):
        (corpus / filename).write_text(json.dumps(dict(record_id="sample", question_id=qid)) + "\n")
    output = tmp_path / "output"
    assert prepare(sample, corpus, output) == {"nll": 1, "joint": 1}
    for stage, qid in (("nll", "team"), ("joint", "refund")):
        row = json.loads((output / f"{stage}.jsonl").read_text())
        assert set(row["request"]["questions"]) == set(row["targets"]) == {qid}
        assert row["targets"][qid] == record()["targets"][qid]
    (corpus / "sft_plan.jsonl").write_text(json.dumps(dict(record_id="sample", question_id="team")) + "\n")
    with pytest.raises(ValueError, match="overlapping"):
        prepare(sample, corpus, tmp_path / "invalid")
