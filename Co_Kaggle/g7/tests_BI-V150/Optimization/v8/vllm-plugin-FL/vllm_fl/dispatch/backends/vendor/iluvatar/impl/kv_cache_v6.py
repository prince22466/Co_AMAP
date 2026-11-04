# SPDX-License-Identifier: Apache-2.0
"""v6 tiled paged KV-cache writes for the v8 v1+v5 Triton attention stack.

The v6 FlagGems kernel assumes NHD-contiguous, unquantized K/V storage.
Use the optimized kernel only when those assumptions hold; otherwise the
caller must retain vLLM's original KV-cache update implementation.
"""

import torch


def supports_v6_kv_write(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str | None,
) -> bool:
    """Check the exact layout and dtype contract of the original v6 kernel."""
    if slot_mapping.ndim != 1 or slot_mapping.numel() < 256:
        return False
    if kv_cache_dtype not in (None, "auto"):
        return False
    if key.ndim != 3 or value.ndim != 3 or kv_cache.ndim != 5:
        return False
    if kv_cache.size(1) != 2 or kv_cache.size(2) == 0:
        return False
    if key.shape != value.shape or key.size(0) != slot_mapping.numel():
        return False
    if tuple(kv_cache.shape[3:]) != tuple(key.shape[1:]):
        return False
    if key.dtype not in (torch.float16, torch.bfloat16):
        return False
    if key.dtype != value.dtype or key.dtype != kv_cache.dtype:
        return False
    if slot_mapping.dtype not in (torch.int32, torch.int64):
        return False
    if key.device != value.device or key.device != kv_cache.device:
        return False
    if key.device != slot_mapping.device or slot_mapping.stride(0) != 1:
        return False

    head_size = key.size(2)
    n = key.size(1) * head_size
    if not (
        key.stride(2) == value.stride(2) == 1
        and key.stride(1) == value.stride(1) == head_size
    ):
        return False

    # v6 writes each slot as n contiguous elements. HND or nonstandard
    # layouts must use the inherited vLLM writer instead.
    key_cache, value_cache = kv_cache.unbind(1)
    for cache in (key_cache, value_cache):
        if not (
            cache.stride(1) == n
            and cache.stride(2) == head_size
            and cache.stride(3) == 1
        ):
            return False
    return key_cache.stride(0) == value_cache.stride(0)


def try_write_kv_cache_v6(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str | None,
) -> bool:
    """Launch v6's 4-token/CTA Triton kernel; return False for fallback.

    Calling the JIT kernel directly is intentional: the FlagGems public
    wrapper can select its C-extension path, making v6's Triton tiling inert.
    This reuses the v6 kernel rather than substituting a different operator.
    """
    if not supports_v6_kv_write(
        key, value, kv_cache, slot_mapping, kv_cache_dtype
    ):
        return False
    if not (key.is_cuda and value.is_cuda and kv_cache.is_cuda and slot_mapping.is_cuda):
        return False

    import triton

    from flag_gems.fused.reshape_and_cache_flash import reshape_and_cache_flash_kernel
    from flag_gems.runtime import torch_device_fn

    key_cache, value_cache = kv_cache.unbind(1)
    num_tokens = slot_mapping.numel()
    num_heads, head_size = key.shape[1:]
    tokens_per_program = 4
    grid = (triton.cdiv(num_tokens, tokens_per_program),)

    with torch_device_fn.device(key.device):
        reshape_and_cache_flash_kernel[grid](
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            key_cache.stride(0),
            key.stride(0),
            value.stride(0),
            num_heads,
            head_size,
            kv_cache.size(2),
            None,
            None,
            num_heads * head_size,
            num_tokens,
            tokens_per_program,
        )
    return True
