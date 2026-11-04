# SPDX-License-Identifier: Apache-2.0
"""v8 v1+v5+v6 KV-cache integration and safe-fallback regression tests."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("tokens", [1, 255, 256, 259, 512])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_v6_nhd_layout_and_threshold(tokens, dtype):
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.kv_cache_v6 import (
        supports_v6_kv_write,
        try_write_kv_cache_v6,
    )

    num_heads, head_size, block_size = 2, 80, 16
    key = torch.empty(tokens * 2, num_heads, head_size, dtype=dtype)[::2]
    value = torch.empty_like(key)
    kv_cache = torch.zeros(
        ((tokens + block_size - 1) // block_size + 3, 2, block_size,
         num_heads, head_size),
        dtype=dtype,
    )
    slots = torch.arange(tokens, dtype=torch.int32)
    assert supports_v6_kv_write(key, value, kv_cache, slots, "auto") == (
        tokens >= 256
    )
    # A CPU-only environment cannot launch a GPU Triton kernel.
    assert not try_write_kv_cache_v6(key, value, kv_cache, slots, "auto")

    if tokens >= 256:
        # The same logical shape with an HND physical layout is unsupported.
        hnd_cache = torch.empty(
            kv_cache.shape[0], 2, num_heads, block_size, head_size,
            dtype=dtype,
        ).permute(0, 1, 3, 2, 4)
        assert not supports_v6_kv_write(key, value, hnd_cache, slots, "auto")
        assert not supports_v6_kv_write(key, value, kv_cache, slots, "fp8")
        assert not supports_v6_kv_write(
            key, value, kv_cache,
            torch.arange(tokens * 2, dtype=torch.int32)[::2], "auto"
        )


def test_v8_combined_backend_uses_v6_then_inherited_fallback(monkeypatch):
    from vllm.v1.attention.backend import AttentionType
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl import (
        attention_biv150_v1_v5 as combined,
    )
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.attention_biv150 import (
        BIV150TritonAttentionImpl,
    )

    impl = object.__new__(combined.BIV150V1V5AttentionImpl)
    impl.attn_type = AttentionType.DECODER
    impl._is_per_token_head_quant = False
    impl.kv_cache_dtype = "auto"
    calls = []

    def optimized(*args):
        calls.append("v6")
        return True

    def inherited(*args):
        calls.append("upstream")

    monkeypatch.setattr(combined, "try_write_kv_cache_v6", optimized)
    monkeypatch.setattr(
        BIV150TritonAttentionImpl, "do_kv_cache_update", inherited,
        raising=False,
    )
    impl.do_kv_cache_update(None, None, None, None, None)
    assert calls == ["v6"]

    calls.clear()
    monkeypatch.setattr(
        combined, "try_write_kv_cache_v6",
        lambda *args: calls.append("v6") or False,
    )
    impl.do_kv_cache_update(None, None, None, None, None)
    assert calls == ["v6", "upstream"]

    calls.clear()
    impl._is_per_token_head_quant = True
    impl.do_kv_cache_update(None, None, None, None, None)
    assert calls == ["upstream"]

    calls.clear()
    impl._is_per_token_head_quant = False
    impl.attn_type = AttentionType.ENCODER
    impl.do_kv_cache_update(None, None, None, None, None)
    assert calls == ["upstream"]


def test_v8_optimized_writer_matches_exact_kv_values_on_gpu(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("Requires BI-V150 or another CUDA-compatible GPU")

    from importlib import import_module

    from vllm.v1.attention.backend import AttentionType
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.attention_biv150_v1_v5 import (
        BIV150V1V5AttentionImpl,
    )

    impl = object.__new__(BIV150V1V5AttentionImpl)
    impl.attn_type = AttentionType.DECODER
    impl._is_per_token_head_quant = False
    impl.kv_cache_dtype = "auto"

    tokens, heads, head_size, block_size = 259, 2, 80, 16
    num_blocks = (tokens + block_size - 1) // block_size + 3
    key = torch.randn(tokens * 2, heads, head_size,
                      dtype=torch.bfloat16, device="cuda")[::2]
    value = torch.randn_like(key)
    slots = torch.arange(tokens, device="cuda", dtype=torch.int32)
    slots = (slots * 17) % (num_blocks * block_size)
    slots[::19] = -1
    cache = torch.zeros(
        (num_blocks, 2, block_size, heads, head_size),
        dtype=key.dtype, device="cuda",
    )
    expected = torch.zeros_like(cache)
    slot_list = slots.cpu().tolist()
    assert len({s for s in slot_list if s >= 0}) == sum(s >= 0 for s in slot_list)
    for i, s in enumerate(slot_list):
        if s >= 0:
            block, offset = divmod(s, block_size)
            expected[block, 0, offset] = key[i]
            expected[block, 1, offset] = value[i]

    cache_module = import_module("flag_gems.fused.reshape_and_cache_flash")
    original = cache_module.reshape_and_cache_flash_kernel
    launches = []

    class KernelSpy:
        def __getitem__(self, grid):
            launcher = original[grid]

            def call(*args, **kwargs):
                launches.append((grid, args[-1]))
                return launcher(*args, **kwargs)

            return call

    monkeypatch.setattr(cache_module, "reshape_and_cache_flash_kernel", KernelSpy())
    impl.do_kv_cache_update(
        SimpleNamespace(_k_scale=None, _v_scale=None),
        key, value, cache, slots,
    )
    torch.cuda.synchronize()
    assert launches == [((65,), 4)]
    torch.testing.assert_close(cache, expected, atol=0, rtol=0)
