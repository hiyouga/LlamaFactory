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

import os

import pytest

from llamafactory.data.data_utils import configure_preprocessing_thread_limits


THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TOKENIZERS_PARALLELISM",
)


@pytest.fixture(autouse=True)
def _clean_thread_env(monkeypatch):
    for env_var in THREAD_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


def _mock_cpu_count(monkeypatch, count: int):
    monkeypatch.setattr(os, "cpu_count", lambda: count)
    if hasattr(os, "sched_getaffinity"):
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(count)))


@pytest.mark.runs_on(["cpu", "mps"])
@pytest.mark.parametrize("num_proc", [None, 1])
def test_no_capping_for_single_process(num_proc):
    r"""The default single-process path must be left untouched."""
    configure_preprocessing_thread_limits(num_proc)
    for env_var in THREAD_ENV_VARS:
        assert env_var not in os.environ


@pytest.mark.runs_on(["cpu", "mps"])
def test_caps_threads_proportionally(monkeypatch):
    r"""Threads should be split evenly across the requested worker processes."""
    _mock_cpu_count(monkeypatch, 32)
    configure_preprocessing_thread_limits(8)
    for env_var in THREAD_ENV_VARS[:-1]:
        assert os.environ[env_var] == "4"  # 32 cpus // 8 workers

    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"


@pytest.mark.runs_on(["cpu", "mps"])
def test_does_not_clobber_user_override(monkeypatch):
    r"""An explicit user setting must always win over our default."""
    monkeypatch.setenv("OMP_NUM_THREADS", "16")
    _mock_cpu_count(monkeypatch, 32)
    configure_preprocessing_thread_limits(8)
    assert os.environ["OMP_NUM_THREADS"] == "16"
    assert os.environ["MKL_NUM_THREADS"] == "4"


@pytest.mark.runs_on(["cpu", "mps"])
def test_floor_is_one_thread_when_num_proc_exceeds_cpus(monkeypatch):
    r"""Never compute a thread count below 1, even when oversubscribing workers."""
    _mock_cpu_count(monkeypatch, 4)
    configure_preprocessing_thread_limits(32)
    assert os.environ["OMP_NUM_THREADS"] == "1"
