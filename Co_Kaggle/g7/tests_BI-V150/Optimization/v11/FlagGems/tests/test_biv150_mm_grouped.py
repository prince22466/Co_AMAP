"""GPU screen for the provisional v11 grouped-MM candidate; not yet promoted."""
import importlib

import pytest
import torch


@pytest.mark.parametrize(
    ("m", "k", "n", "dtype", "layout", "grouped"),
    [
        (2048, 2048, 12288, torch.bfloat16, "transposed_weight", True),
        (2048, 2048, 12288, torch.bfloat16, "transposed_input", True),
        (2048, 2048, 12288, torch.float16, "transposed_weight", True),
        (57, 2048, 12288, torch.bfloat16, "transposed_weight", False),
        (7, 2051, 257, torch.bfloat16, "strided_input", False),
        (9, 33, 65, torch.float16, "contiguous", False),
    ],
)
def test_candidate_matches_torch_and_uses_expected_mapping(
    m, k, n, dtype, layout, grouped, monkeypatch
):
    if not torch.cuda.is_available():
        pytest.skip("Requires the authorized BI-V150 GPU")
    mm_module = importlib.import_module("flag_gems.runtime.backend._iluvatar.ops.mm")
    torch.manual_seed(20261111)
    if layout == "transposed_input":
        a = torch.randn((k, m), device="cuda", dtype=dtype).T
    elif layout == "strided_input":
        a = torch.randn((m * 2, k * 2), device="cuda", dtype=dtype)[::2, ::2]
    else:
        a = torch.randn((m, k), device="cuda", dtype=dtype)
    if layout == "transposed_weight":
        b = torch.randn((n, k), device="cuda", dtype=dtype).T
    else:
        b = torch.randn((k, n), device="cuda", dtype=dtype)

    launches = []
    original_kernel = mm_module.mm_kernel

    class KernelSpy:
        def __getitem__(self, grid):
            launch = original_kernel[grid]

            def capture(*args, **kwargs):
                launches.append(kwargs["GROUPED_MAPPING"])
                return launch(*args, **kwargs)

            return capture

    monkeypatch.setattr(mm_module, "mm_kernel", KernelSpy())
    with torch.inference_mode():
        actual = mm_module.mm(a, b)
        expected = torch.mm(a, b)
        torch.cuda.synchronize()
    assert launches == [grouped]
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.25, rtol=0.03)


def test_mm_out_strided_output_and_tails():
    if not torch.cuda.is_available():
        pytest.skip("Requires the authorized BI-V150 GPU")
    mm_module = importlib.import_module("flag_gems.runtime.backend._iluvatar.ops.mm")
    torch.manual_seed(20261112)
    a = torch.randn((7, 33), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((65, 33), device="cuda", dtype=torch.bfloat16).T
    backing = torch.empty((7, 130), device="cuda", dtype=torch.bfloat16)
    out = backing[:, ::2]
    with torch.inference_mode():
        result = mm_module.mm_out(a, b, out=out)
        expected = torch.mm(a, b)
        torch.cuda.synchronize()
    assert result is out
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, expected, atol=0.25, rtol=0.03)
