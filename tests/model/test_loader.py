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

from types import SimpleNamespace

import torch

from llamafactory.hparams import get_train_args
from llamafactory.model import loader


TRAIN_ARGS = {
    "model_name_or_path": "dummy",
    "stage": "sft",
    "finetuning_type": "lora",
    "dataset": "dummy",
    "template": "llama3",
    "output_dir": "dummy_dir",
    "overwrite_output_dir": True,
    "report_to": "none",
    "use_cpu": True,
}


def test_full_precision_dtype_is_set_only_for_training():
    model_args, *_ = get_train_args({**TRAIN_ARGS, "do_train": True})
    assert model_args.compute_dtype == torch.float32

    model_args, *_ = get_train_args({**TRAIN_ARGS, "do_train": False})
    assert model_args.compute_dtype is None

    model_args, *_ = get_train_args({**TRAIN_ARGS, "do_train": True, "fp8": True})
    assert model_args.compute_dtype is None


def test_full_precision_training_overrides_checkpoint_dtype(monkeypatch):
    captured_kwargs = {}
    config = SimpleNamespace(model_type="dummy")

    class DummyModel:
        def __init__(self):
            self.config = config

        def train(self):
            return self

    def fake_from_pretrained(**kwargs):
        captured_kwargs.update(kwargs)
        return DummyModel()

    model_args = SimpleNamespace(
        adapter_name_or_path=None,
        compute_dtype=torch.float32,
        mixture_of_depths=None,
        model_name_or_path="dummy",
        print_param_status=False,
        train_from_scratch=False,
        trust_remote_code=False,
        use_kt=False,
        use_unsloth=False,
        use_v1_kernels=False,
    )
    finetuning_args = SimpleNamespace(stage="sft")

    monkeypatch.setattr(loader, "_get_init_kwargs", lambda _model_args: {})
    monkeypatch.setattr(loader, "load_config", lambda _model_args: config)
    monkeypatch.setattr(loader, "patch_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(loader, "apply_liger_kernel", lambda *args, **kwargs: None)
    monkeypatch.setattr(loader.AutoModelForCausalLM, "from_pretrained", staticmethod(fake_from_pretrained))
    monkeypatch.setattr(loader, "patch_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(loader, "register_autoclass", lambda *args, **kwargs: None)
    monkeypatch.setattr(loader, "init_adapter", lambda _config, model, *args, **kwargs: model)
    monkeypatch.setattr(loader, "is_torch_version_greater_than", lambda _version: False)
    monkeypatch.setattr(loader, "count_parameters", lambda _model: (1, 1))

    loader.load_model(object(), model_args, finetuning_args, is_trainable=True)

    assert captured_kwargs["torch_dtype"] == torch.float32
