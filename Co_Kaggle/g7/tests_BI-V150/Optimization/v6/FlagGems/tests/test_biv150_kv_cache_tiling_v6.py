"""Correctness checks for v6 tiled paged-KV cache writes."""
import pytest
import torch


@pytest.mark.parametrize("num_tokens", [7, 64, 256, 259, 512])
@pytest.mark.parametrize("head_size", [64, 80])
def test_tiled_reshape_and_cache_flash(num_tokens, head_size):
    if not torch.cuda.is_available():
        pytest.skip("Triton KV-cache kernel requires an available GPU")

    from flag_gems.fused.reshape_and_cache_flash import reshape_and_cache_flash
    import flag_gems.fused.reshape_and_cache_flash as cache_module

    # Validate the Triton implementation rather than the C-extension path.
    if cache_module.use_c_extension:
        pytest.skip("Requires FlagGems Triton implementation")

    device = "cuda"
    heads, block_size = 2, 16
    num_blocks = (num_tokens + block_size - 1) // block_size + 3
    key = torch.randn(num_tokens, heads, head_size, device=device, dtype=torch.float16)
    value = torch.randn_like(key)
    # Nonmonotonic physical slots exercise the page-table / slot arithmetic.
    slots = torch.arange(num_tokens, device=device, dtype=torch.int64)
    slots = (slots * 17) % (num_blocks * block_size)
    slots[::19] = -1
    key_cache = torch.zeros(num_blocks, block_size, heads, head_size, device=device, dtype=key.dtype)
    value_cache = torch.zeros_like(key_cache)
    expected_key = torch.zeros_like(key_cache)
    expected_value = torch.zeros_like(value_cache)

    for t, slot in enumerate(slots.cpu().tolist()):
        if slot >= 0:
            expected_key.view(-1, heads, head_size)[slot] = key[t]
            expected_value.view(-1, heads, head_size)[slot] = value[t]

    reshape_and_cache_flash(
        key, value, key_cache, value_cache, slots, "auto", None, None
    )
    torch.testing.assert_close(key_cache, expected_key, rtol=0, atol=0)
    torch.testing.assert_close(value_cache, expected_value, rtol=0, atol=0)
