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

import pytest
import torch

from llamafactory.train.trainer_utils import log1mexp
from llamafactory.v1.utils.helper import log1mexp as log1mexp_v1


IMPLEMENTATIONS = [log1mexp, log1mexp_v1]


@pytest.mark.parametrize("fn", IMPLEMENTATIONS, ids=["legacy", "v1"])
def test_log1mexp_matches_float64_reference(fn):
    r"""Both tails must match the reference, not only the p -> 1 one.

    ``atol`` is set to zero on purpose: the expm1-only form returns 0 instead of ``-p`` once
    ``p < 2 ** -25``, and an absolute tolerance would swallow that error completely.

    """
    xs = torch.tensor([-1e-12, -1e-9, -1e-7, -0.5, -0.7, -2.0, -10.0, -30.0])
    expected = torch.log1p(-torch.exp(xs.double()))

    assert torch.allclose(fn(xs).double(), expected, rtol=1e-5, atol=0.0)


@pytest.mark.parametrize("fn", IMPLEMENTATIONS, ids=["legacy", "v1"])
def test_log1mexp_is_finite_for_saturated_logps(fn):
    r"""The naive log1p(-exp(x)) returns -inf here, which makes ORPO's loss NaN."""
    xs = torch.tensor([-1e-9, -1e-12, -30.0])

    assert torch.isfinite(fn(xs)).all()


@pytest.mark.parametrize("fn", IMPLEMENTATIONS, ids=["legacy", "v1"])
def test_log1mexp_gradients_are_finite(fn):
    r"""torch.where evaluates both branches, so an unclamped one leaks -inf into the backward."""
    xs = torch.tensor([-1e-9, -1e-7, -30.0], requires_grad=True)

    fn(xs).sum().backward()

    assert torch.isfinite(xs.grad).all()


def test_log1mexp_implementations_stay_in_sync():
    r"""The v1 tree does not import from the legacy one, so the two copies must not drift."""
    xs = torch.linspace(-40.0, 0.0, 1024)

    torch.testing.assert_close(log1mexp(xs), log1mexp_v1(xs), rtol=0.0, atol=0.0)
