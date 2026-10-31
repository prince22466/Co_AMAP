"""GPU correctness checks for the v6 paged KV-cache Triton write kernel."""

from importlib import import_module

import pytest
import torch


@pytest.mark.parametrize("num_tokens", [1, 7, 64, 255, 256, 259, 512])
@pytest.mark.parametrize("head_size", [64, 80])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
def test_tiled_reshape_and_cache_flash(
    num_tokens, head_size, dtype, slot_dtype, monkeypatch
):
    if not torch.cuda.is_available():
        pytest.skip("Triton KV-cache kernel requires a CUDA-compatible GPU")

    cache_module = import_module("flag_gems.fused.reshape_and_cache_flash")
    # Explicitly exercise the kernel under review, even when a C-extension
    # is enabled by default in the installed FlagGems configuration.
    monkeypatch.setattr(cache_module, "use_c_extension", False)

    heads, block_size = 2, 16
    num_blocks = (num_tokens + block_size - 1) // block_size + 3
    device = torch.device("cuda")
    # Sliced inputs have a nontrivial token stride; their innermost
    # head dimension remains contiguous as required by the kernel.
    key = torch.randn(num_tokens * 2, heads, head_size, device=device, dtype=dtype)[::2]
    value = torch.randn(num_tokens * 2, heads, head_size, device=device, dtype=dtype)[::2]
    slots = torch.arange(num_tokens, device=device, dtype=slot_dtype)
    slots = (slots * 17) % (num_blocks * block_size)
    slots[::19] = -1

    key_cache = torch.zeros(num_blocks, block_size, heads, head_size, device=device, dtype=dtype)
    value_cache = torch.zeros_like(key_cache)
    expected_key = torch.zeros_like(key_cache)
    expected_value = torch.zeros_like(value_cache)

    # Unique slots guarantee deterministic correctness checks without
    # concurrent writes to the same physical cache address.
    slot_list = slots.cpu().tolist()
    assert len({s for s in slot_list if s >= 0}) == sum(s >= 0 for s in slot_list)
    for t, slot in enumerate(slot_list):
        if slot >= 0:
            expected_key.view(-1, heads, head_size)[slot] = key[t]
            expected_value.view(-1, heads, head_size)[slot] = value[t]

    cache_module.reshape_and_cache_flash(
        key, value, key_cache, value_cache, slots, "auto", None, None
    )
    torch.testing.assert_close(key_cache, expected_key, rtol=0, atol=0)
    torch.testing.assert_close(value_cache, expected_value, rtol=0, atol=0)
