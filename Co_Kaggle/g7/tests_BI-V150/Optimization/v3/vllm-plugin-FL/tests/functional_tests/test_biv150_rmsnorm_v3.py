# SPDX-License-Identifier: Apache-2.0
"""BI-V150 v3 RMSNorm correctness and conservative dispatch guards."""

import pytest
import torch


class NormLayer:
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        self.weight = weight
        self.variance_epsilon = eps


def test_opt_out_keeps_original_reference(monkeypatch):
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.normalization import (
        rms_norm_iluvatar,
    )

    monkeypatch.delenv("VLLM_FL_BIV150_RMS_V3", raising=False)
    weight = torch.randn(256)
    x = torch.randn(3, 256)
    residual = torch.randn_like(x)
    layer = NormLayer(weight)
    expected_r = x + residual
    expected_y = weight * (expected_r * torch.rsqrt(
        expected_r.pow(2).mean(-1, keepdim=True) + layer.variance_epsilon
    ))
    actual_y, actual_r = rms_norm_iluvatar(layer, x, residual)
    torch.testing.assert_close(actual_r, expected_r, atol=0, rtol=0)
    torch.testing.assert_close(actual_y, expected_y, atol=0, rtol=0)


def test_opt_in_cpu_falls_back(monkeypatch):
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.normalization import (
        rms_norm_iluvatar,
    )
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.normalization_biv150_v3 import (
        is_eligible,
    )

    monkeypatch.setenv("VLLM_FL_BIV150_RMS_V3", "1")
    x = torch.randn(2, 256)
    weight = torch.ones(256)
    assert not is_eligible(x, None, weight)
    actual = rms_norm_iluvatar(NormLayer(weight), x)
    reference = weight * (
        x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    )
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)


@pytest.mark.parametrize("with_residual", [False, True])
@pytest.mark.parametrize(
    ("rows", "hidden", "dtype"),
    [
        (1, 2048, torch.float32),
        (32, 2560, torch.float16),
        (127, 2560, torch.bfloat16),
        (4096, 2048, torch.bfloat16),
    ],
)
def test_fused_norm_matches_fp32_reference(
    monkeypatch, rows, hidden, dtype, with_residual
):
    if not torch.cuda.is_available():
        pytest.skip("Requires BI-V150 / CUDA-compatible GPU")

    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.normalization import (
        rms_norm_iluvatar,
    )
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.normalization_biv150_v3 import (
        is_eligible,
    )

    monkeypatch.setenv("VLLM_FL_BIV150_RMS_V3", "1")
    torch.manual_seed(2026)
    x = torch.randn(rows, hidden, device="cuda", dtype=dtype)
    weight = torch.randn(hidden, device="cuda", dtype=dtype)
    residual = torch.randn_like(x) if with_residual else None
    assert is_eligible(x, residual, weight)

    with torch.inference_mode():
        actual = rms_norm_iluvatar(NormLayer(weight), x, residual)
        combined = (
            (x.float() + residual.float()).to(dtype)
            if residual is not None
            else x
        )
        inv_rms = torch.rsqrt(combined.float().square().mean(-1, keepdim=True) + 1e-6)
        expected = (combined.float() * inv_rms * weight.float()).to(dtype)
        torch.cuda.synchronize()

    if with_residual:
        actual, returned_residual = actual
        torch.testing.assert_close(returned_residual, combined, atol=0, rtol=0)

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


def test_noncontiguous_input_and_wrong_weight_dtype_fall_back(monkeypatch):
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.normalization_biv150_v3 import (
        is_eligible,
    )

    monkeypatch.setenv("VLLM_FL_BIV150_RMS_V3", "1")
    x = torch.randn(4, 256)
    weight = torch.ones(256)
    assert not is_eligible(x.t(), None, weight)
    if torch.cuda.is_available():
        device = "cuda"
        xx = x.to(device)
        assert not is_eligible(xx, None, weight.to(device, dtype=torch.float16))
        assert not is_eligible(xx, xx[:, ::-1] if False else xx[:, :128], weight.to(device))
