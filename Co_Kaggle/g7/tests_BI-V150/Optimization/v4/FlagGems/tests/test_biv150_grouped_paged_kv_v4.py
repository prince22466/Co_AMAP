"""BI-V150 integration regression for v4 grouped paged-KV address lookups.

This test requires the real Iluvatar Triton compiler + device; CPU-only
validation is provided separately. The baseline and grouped kernels share the exact same input,
including randomized physical page IDs and an incomplete final KV page.
"""

import math

import pytest
import torch

import flag_gems
from flag_gems.config import use_c_extension
from flag_gems.ops.flash_api import mha_varlan_fwd, mha_varlan_fwd_opt
from flag_gems.ops.iluvatar_grouped_paged_kv import ENV_NAME


@pytest.mark.parametrize("query_len,kv_len", [(256, 8193), (2048, 16384)])
@pytest.mark.parametrize("entrypoint", ("regular", "optimized"))
def test_shuffled_paged_kv_grouped_matches_baseline(
    monkeypatch, query_len, kv_len, entrypoint
):
    if flag_gems.vendor_name != "iluvatar" or not torch.cuda.is_available():
        pytest.skip("Requires BI-V150 with Iluvatar FlagGems")
    if use_c_extension:
        pytest.skip("Requires the Python Triton attention path, not C extension")

    torch.manual_seed(2026)
    device = "cuda"
    q_heads, kv_heads, head_dim, page_size = 16, 2, 128, 16
    pages = math.ceil(kv_len / page_size)
    q = torch.randn(
        (query_len, q_heads, head_dim), device=device, dtype=torch.bfloat16
    )
    k = torch.randn(
        (pages, page_size, kv_heads, head_dim),
        device=device, dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)
    page_table = torch.randperm(pages, device=device).to(torch.int32)[None, :]
    cu_q = torch.tensor([0, query_len], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, kv_len], device=device, dtype=torch.int32)
    seqused_k = torch.tensor([kv_len], device=device, dtype=torch.int32)

    def launch(mode):
        monkeypatch.setenv(ENV_NAME, mode)
        # The v4 port has no tile sweep; the original tile policy is fixed.
        out = torch.empty_like(q)
        common = (
            cu_q, cu_k, seqused_k, None, page_table, None,
            query_len, kv_len, 0.0, head_dim ** -0.5,
            False, True, -1, -1, 0.0, False, None,
        )
        if entrypoint == "optimized":
            lse = torch.empty(
                (q_heads, query_len), device=device, dtype=torch.float32
            )
            return mha_varlan_fwd_opt(q, k, v, out, lse, *common)[0].clone()
        return mha_varlan_fwd(q, k, v, out, *common)[0].clone()

    baseline = launch("baseline")
    grouped = launch("grouped")
    torch.cuda.synchronize()
    assert torch.isfinite(baseline).all().item()
    assert torch.isfinite(grouped).all().item()
    torch.testing.assert_close(grouped, baseline, atol=0.03, rtol=0.03)
