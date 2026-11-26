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

import math
from decimal import Decimal, localcontext

import pytest
import torch

from llamafactory.train.trainer_utils import log1mexp
from llamafactory.v1.utils.helper import log1mexp as log1mexp_v1


@pytest.mark.parametrize("implementation", [log1mexp, log1mexp_v1], ids=["legacy", "v1"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_log1mexp_values_and_gradients_match_decimal(implementation, dtype):
    epsilon = 1e-8 if dtype == torch.float32 else 1e-18
    cutoff = torch.tensor(-math.log(2), dtype=dtype)
    below = torch.nextafter(cutoff, torch.tensor(-torch.inf, dtype=dtype))
    above = torch.nextafter(cutoff, torch.tensor(torch.inf, dtype=dtype))
    inputs = torch.tensor([-epsilon, -10.0, -40.0, below.item(), cutoff.item(), above.item()], dtype=dtype)
    inputs = inputs.reshape(2, 3).requires_grad_()

    with localcontext() as context:
        context.prec = 100
        probabilities = [Decimal.from_float(value).exp() for value in inputs.detach().flatten().tolist()]
        expected_values = [float((1 - probability).ln()) for probability in probabilities]
        expected_gradients = [float(-probability / (1 - probability)) for probability in probabilities]

    actual = implementation(inputs)
    actual.sum().backward()
    assert torch.isfinite(actual).all()
    assert torch.isfinite(inputs.grad).all()
    tolerance = 5e-6 if dtype == torch.float32 else 2e-14
    # An absolute tolerance would hide a zero value or gradient in the tail.
    torch.testing.assert_close(
        actual, torch.tensor(expected_values, dtype=dtype).reshape_as(inputs), rtol=tolerance, atol=0.0
    )
    torch.testing.assert_close(
        inputs.grad, torch.tensor(expected_gradients, dtype=dtype).reshape_as(inputs), rtol=tolerance, atol=0.0
    )


@pytest.mark.parametrize("implementation", [log1mexp, log1mexp_v1], ids=["legacy", "v1"])
def test_log1mexp_preserves_endpoint_limits(implementation):
    actual = implementation(torch.tensor([-torch.inf, 0.0], dtype=torch.float64))
    torch.testing.assert_close(actual, torch.tensor([-0.0, -torch.inf], dtype=torch.float64), rtol=0.0, atol=0.0)
