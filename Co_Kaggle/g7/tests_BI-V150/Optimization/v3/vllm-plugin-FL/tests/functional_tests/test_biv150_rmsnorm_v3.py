# SPDX-License-Identifier: Apache-2.0
"""BI-V150 v3: guards, FlagGems equivalence, in-place ABI and OOT integration."""

import pytest
import torch


class NormLayer:
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        self.weight = weight
        self.variance_epsilon = eps


def _require_biv150():
    if not torch.cuda.is_available():
        pytest.skip("Requires BI-V150 / CUDA-compatible GPU")
    from vllm.platforms import current_platform
    if getattr(current_platform, "vendor_name", None) != "iluvatar":
        pytest.skip("FlagGems v3 opt-in is restricted to Iluvatar GPUs")


def test_inference_parameter_guard_and_autograd_fallback(monkeypatch):
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.normalization_biv150_v3 import (
        is_eligible,
    )

    x = torch.randn(2, 2048)
    weight = torch.nn.Parameter(torch.ones(2048))
    assert not is_eligible(x, None, weight)  # CPU not supported

    if not torch.cuda.is_available():
        return

    x, weight = x.cuda(), torch.nn.Parameter(weight.detach().cuda())
    assert not is_eligible(x, None, weight)  # gradient recording is on
    with torch.inference_mode():
        assert is_eligible(x, None, weight)
    with torch.no_grad():
        assert is_eligible(x, None, weight)

    with torch.inference_mode():
        residual = torch.randn_like(x)
        assert is_eligible(x, residual, weight)
        assert not is_eligible(x, x, weight)  # in-place aliasing unsupported
        assert not is_eligible(x, residual[:, :1024], weight)
        assert not is_eligible(x.t(), None, weight)
        assert not is_eligible(x, None, weight.to(torch.float16))
        assert not is_eligible(
            torch.empty((2, 8192), device="cuda"), None,
            torch.ones(8192, device="cuda"),
        )


def test_opt_out_preserves_original_vendor_reference(monkeypatch):
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.normalization import (
        rms_norm_iluvatar,
    )

    monkeypatch.delenv("VLLM_FL_BIV150_RMS_V3", raising=False)
    weight = torch.randn(256)
    x = torch.randn(3, 256)
    residual = torch.randn_like(x)
    expected_r = x + residual
    expected_y = weight * (
        expected_r * torch.rsqrt(expected_r.pow(2).mean(-1, keepdim=True) + 1e-6)
    )
    y, r = rms_norm_iluvatar(NormLayer(weight), x, residual)
    assert r is not residual
    torch.testing.assert_close(r, expected_r, atol=0, rtol=0)
    torch.testing.assert_close(y, expected_y, atol=0, rtol=0)


def test_opt_in_cpu_falls_back(monkeypatch):
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.normalization import (
        rms_norm_iluvatar,
    )

    monkeypatch.setenv("VLLM_FL_BIV150_RMS_V3", "1")
    x = torch.randn(2, 256)
    weight = torch.ones(256)
    y = rms_norm_iluvatar(NormLayer(weight), x)
    reference = weight * (
        x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    )
    torch.testing.assert_close(y, reference, atol=0, rtol=0)


@pytest.mark.parametrize("with_residual", [False, True])
@pytest.mark.parametrize(
    ("rows", "hidden", "dtype"),
    [
        (1, 2048, torch.float32),
        (32, 2560, torch.float16),
        (127, 2560, torch.bfloat16),
        (256, 2048, torch.bfloat16),
        (32, 128, torch.bfloat16),
        (16, 4096, torch.float16),
    ],
)
def test_v3_matches_actual_flaggems_and_inplace_contract(
    monkeypatch, rows, hidden, dtype, with_residual
):
    _require_biv150()
    pytest.importorskip("flag_gems")
    from vllm_fl.dispatch.backends.flaggems.impl.normalization import (
        rms_norm_flaggems,
    )

    torch.manual_seed(2026)
    weight = torch.nn.Parameter(torch.randn(hidden, device="cuda", dtype=dtype))
    x = torch.randn(rows, hidden, device="cuda", dtype=dtype)
    residual = torch.randn_like(x) if with_residual else None
    base_x = x.clone()
    base_r = residual.clone() if residual is not None else None
    new_x = x.clone()
    new_r = residual.clone() if residual is not None else None

    with torch.inference_mode():
        monkeypatch.setenv("VLLM_FL_BIV150_RMS_V3", "0")
        expected = rms_norm_flaggems(NormLayer(weight), base_x, base_r)
        monkeypatch.setenv("VLLM_FL_BIV150_RMS_V3", "1")
        actual = rms_norm_flaggems(NormLayer(weight), new_x, new_r)
        torch.cuda.synchronize()

    if with_residual:
        base_y, base_res = expected
        new_y, new_res = actual
        # FlagGems residual-add is in-place: compare BOTH mutations and
        # identity, not just returned tensor values.
        assert base_y is base_x and base_res is base_r
        assert new_y is new_x and new_res is new_r
        torch.testing.assert_close(new_r, base_r, atol=0, rtol=0)
        torch.testing.assert_close(new_y, base_y, atol=3e-2, rtol=3e-2)
    else:
        assert actual is not new_x
        torch.testing.assert_close(new_x, x, atol=0, rtol=0)
        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


def test_vendor_opt_in_stays_out_of_place(monkeypatch):
    _require_biv150()
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.normalization import (
        rms_norm_iluvatar,
    )

    monkeypatch.setenv("VLLM_FL_BIV150_RMS_V3", "1")
    with torch.inference_mode():
        x = torch.randn(4, 2048, device="cuda", dtype=torch.bfloat16)
        r = torch.randn_like(x)
        orig_x, orig_r = x.clone(), r.clone()
        w = torch.nn.Parameter(torch.ones(2048, device="cuda", dtype=x.dtype))
        y, res_out = rms_norm_iluvatar(NormLayer(w), x, r)
        torch.cuda.synchronize()
    assert y is not x and res_out is not r
    torch.testing.assert_close(x, orig_x, atol=0, rtol=0)
    torch.testing.assert_close(r, orig_r, atol=0, rtol=0)


def test_real_oot_rmsnorm_calls_v3_via_flaggems(monkeypatch):
    """Regression for default BI-V150 FlagGems selection + Parameter guard."""
    _require_biv150()
    pytest.importorskip("flag_gems")

    from vllm_fl.dispatch import get_default_manager
    from vllm_fl.ops.layernorm import RMSNormFL
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl import normalization_biv150_v3

    # The production default should remain FlagGems, not the vendor reference.
    assert get_default_manager().get_selected_impl_id("rms_norm") == "default.flagos"

    launches = []
    orig = normalization_biv150_v3.fused_rms_norm

    def tracked(*args, **kwargs):
        launches.append(kwargs)
        return orig(*args, **kwargs)

    monkeypatch.setattr(normalization_biv150_v3, "fused_rms_norm", tracked)
    monkeypatch.setenv("VLLM_FL_BIV150_RMS_V3", "1")
    with torch.inference_mode():
        layer = RMSNormFL(2048, dtype=torch.bfloat16).cuda()
        assert isinstance(layer.weight, torch.nn.Parameter)
        assert layer.weight.requires_grad
        x = torch.randn(8, 2048, device="cuda", dtype=torch.bfloat16)
        r = torch.randn_like(x)
        y, new_r = layer.forward_oot(x, r)
        torch.cuda.synchronize()

    assert len(launches) == 1, "RMSNormFL did not dispatch into v3"
    assert launches[0]["inplace"] is True
    assert y is x and new_r is r
