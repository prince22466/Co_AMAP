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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.config import use_c_extension
from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

vendor_name = device.vendor_name


@libentry()
@triton.jit
def reshape_and_cache_flash_kernel(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    block_stride,
    key_stride,
    value_stride,
    num_heads,
    head_size,
    block_size,
    k_scale,
    v_scale,
    n: tl.constexpr,
    num_tokens: tl.constexpr,
    TOKENS_PER_PROGRAM: tl.constexpr,
):
    # Batch several cache insertions per CTA during prefill. The innermost
    # dimension remains contiguous, and each token retains its own page slot.
    token_idx = tl.program_id(0) * TOKENS_PER_PROGRAM + tl.arange(
        0, TOKENS_PER_PROGRAM
    )
    valid_token = token_idx < num_tokens
    slot_idx = tl.load(slot_mapping + token_idx, mask=valid_token, other=-1)
    valid_slot = valid_token & (slot_idx >= 0)

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    i = tl.arange(0, triton.next_power_of_2(n))
    mask = valid_slot[:, None] & (i[None, :] < n)

    src_key_idx = token_idx[:, None] * key_stride + i[None, :]
    src_value_idx = token_idx[:, None] * value_stride + i[None, :]
    # The NHD page layout places all head elements of one token together.
    # Avoid the per-element head division/modulo in the previous kernel.
    tgt_key_value_idx = (
        block_idx[:, None] * block_stride
        + block_offset[:, None] * n
        + i[None, :]
    )

    tgt_key = tl.load(key + src_key_idx, mask=mask, other=0)
    tgt_value = tl.load(value + src_value_idx, mask=mask, other=0)

    # TODO: support fp8 dtype
    tl.store(key_cache + tgt_key_value_idx, tgt_key, mask=mask)
    tl.store(value_cache + tgt_key_value_idx, tgt_value, mask=mask)


def reshape_and_cache_flash(
    key,  # [num_tokens, num_heads, head_size]
    value,  # [num_tokens, num_heads, head_size]
    key_cache,  # [num_blocks, block_size, num_heads, head_size]
    value_cache,  # [num_blocks, block_size, num_heads, head_size]
    slot_mapping,  # [num_tokens]
    kv_cache_dtype,
    k_scale,
    v_scale,
):
    if use_c_extension:
        logger.debug("GEMS RESHAPE_AND_CACHE_FLASH(C EXTENSION)")
        torch.ops.flag_gems.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
    else:
        logger.debug("GEMS RESHAPE_AND_CACHE_FLASH")
        # GCU300 rejects 64-bit IR; slot_mapping arrives as int64 and its
        # index arithmetic would emit 64-bit ops. Downcast to int32 on enflame
        # (mirrors the enflame concat_and_cache_mla override). Vendor+dtype
        # guarded so no other backend is affected.
        if vendor_name == "enflame" and slot_mapping.dtype == torch.int64:
            slot_mapping = slot_mapping.to(torch.int32)
        num_tokens = slot_mapping.size(0)
        num_heads = key.size(1)
        head_size = key.size(2)
        block_size = key_cache.size(1)

        key_stride = key.stride(0)
        value_stride = value.stride(0)
        block_stride = key_cache.stride(0)
        assert key_cache.stride(0) == value_cache.stride(0)

        # Decode writes are latency-sensitive: keep one token per program.
        # Larger prefill writes amortize CTA scheduling over four cache slots.
        tokens_per_program = 4 if num_tokens >= 256 else 1
        grid = (triton.cdiv(num_tokens, tokens_per_program),)
        with torch_device_fn.device(key.device):
            reshape_and_cache_flash_kernel[grid](
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                block_stride,
                key_stride,
                value_stride,
                num_heads,
                head_size,
                block_size,
                k_scale,
                v_scale,
                num_heads * head_size,
                num_tokens,
                tokens_per_program,
            )
