# Copyright 2026 FlagOS Contributors
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

"""v7 Iluvatar GEMM/Linear regression tests (requires BI-V150 or Iluvatar GPU)."""

import pytest
import torch

import flag_gems


pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "iluvatar" or not torch.cuda.is_available(),
    reason="v7 kernels require an Iluvatar GPU",
)

# Aligned and ragged tiles, small decode M, and 4K/16K prefill-like M.
SHAPES = [
    (1, 128, 128),
    (7, 133, 131),
    (65, 129, 96),
    (128, 256, 256),
    (16, 2560, 512),
    (4096, 128, 128),
    (16384, 128, 128),
]


def _assert_close(actual, expected, dtype):
    if dtype == torch.bfloat16:
        torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.05)
    else:
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)


@pytest.mark.parametrize("m,k,n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_bias", [False, True])
def test_v7_iluvatar_linear(m, k, n, dtype, with_bias):
    """Directly exercise the optimized vendor implementation, not a fallback."""
    from flag_gems.runtime.backend._iluvatar.ops.linear import linear

    torch.manual_seed(22466)
    x = torch.randn((m, k), device="cuda", dtype=dtype)
    weight = torch.randn((n, k), device="cuda", dtype=dtype)
    bias = (
        torch.randn(n, device="cuda", dtype=dtype)
        if with_bias
        else None
    )
    expected = torch.nn.functional.linear(x, weight, bias)
    actual = linear(x, weight, bias)
    assert actual.shape == expected.shape
    _assert_close(actual, expected, dtype)


@pytest.mark.parametrize("m,k,n", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("transposed_weight", [False, True])
def test_v7_iluvatar_mm_and_mm_out(m, k, n, dtype, transposed_weight):
    """Exercise both matrix layouts and the caller-provided output path."""
    from flag_gems.runtime.backend._iluvatar.ops.mm import mm, mm_out

    torch.manual_seed(22466)
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    if transposed_weight:
        storage = torch.randn((n, k), device="cuda", dtype=dtype)
        b = storage.t()
        assert b.stride() == (1, k), "Expected a column-major weight"
    else:
        b = torch.randn((k, n), device="cuda", dtype=dtype)
        assert b.stride() == (n, 1), "Expected a row-major weight"

    expected = torch.mm(a, b)
    _assert_close(mm(a, b), expected, dtype)

    out = torch.empty((m, n), device="cuda", dtype=dtype)
    returned = mm_out(a, b, out=out)
    assert returned is out
    _assert_close(out, expected, dtype)


@pytest.mark.parametrize("with_bias", [False, True])
def test_v7_iluvatar_linear_batch_dimensions(with_bias):
    """Keep the baseline's one- and three-dimensional Linear API behavior."""
    from flag_gems.runtime.backend._iluvatar.ops.linear import linear

    dtype = torch.float16
    weight = torch.randn((96, 128), dtype=dtype, device="cuda")
    bias = torch.randn(96, dtype=dtype, device="cuda") if with_bias else None
    for shape in [(128,), (2, 3, 128)]:
        x = torch.randn(shape, dtype=dtype, device="cuda")
        _assert_close(
            linear(x, weight, bias),
            torch.nn.functional.linear(x, weight, bias),
            dtype,
        )
