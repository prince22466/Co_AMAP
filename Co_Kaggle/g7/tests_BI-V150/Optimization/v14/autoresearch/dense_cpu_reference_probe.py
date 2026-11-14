# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: compare v12 dense output with CPU causal-attention references.

Run on BI-V150 with v12 plugin and FlagGems on PYTHONPATH. This reports a
numerical discrepancy; it does not identify its cause or establish model quality.
"""
import torch
from vllm_fl.dispatch.backends.vendor.iluvatar.impl import triton_unified_attention_biv150 as kernel

torch.manual_seed(12012026)
qlen, context, page, heads, kvheads, dim = 65, 33, 16, 16, 2, 128
length = qlen + context
used = (length + page - 1) // page
pages = used + 2
q = torch.randn((qlen, heads, dim), device="cuda", dtype=torch.bfloat16)
k = torch.randn((pages, page, kvheads, dim), device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)
order = torch.randperm(pages, device="cuda", dtype=torch.int64)
out = torch.empty_like(q)
kwargs = dict(
    q=q, k=k, v=v, out=out,
    cu_seqlens_q=torch.tensor([0, qlen], device="cuda", dtype=torch.int32),
    max_seqlen_q=qlen,
    seqused_k=torch.tensor([length], device="cuda", dtype=torch.int32),
    max_seqlen_k=length, softmax_scale=dim ** -0.5, causal=True,
    window_size=(-1, -1), block_table=order.to(torch.int32)[None, :],
    softcap=0.0, q_descale=None, k_descale=None, v_descale=None,
)
with torch.inference_mode():
    kernel.unified_attention(**kwargs)
    torch.cuda.synchronize()
    q_cpu = q.cpu().float()
    order_cpu = order.cpu()
    k_cpu = k.cpu()[order_cpu[:used]].reshape(-1, kvheads, dim)[:length].float()
    v_cpu = v.cpu()[order_cpu[:used]].reshape(-1, kvheads, dim)[:length].float()
    actual = out.cpu().float()
    torch.set_num_threads(4)
    head_map = torch.arange(heads) // (heads // kvheads)
    scores = torch.einsum("qhd,thd->qht", q_cpu, k_cpu[:, head_map]) * dim ** -0.5
    mask = torch.arange(length)[None, :] <= context + torch.arange(qlen)[:, None]
    scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
    for mode in ("dense", "extra_zero_logit", "extra_first_tile_max"):
        if mode == "dense":
            augmented = scores
        else:
            extra = (torch.zeros((qlen, heads, 1)) if mode == "extra_zero_logit"
                     else scores[:, :, :32].max(dim=-1, keepdim=True).values)
            augmented = torch.cat((scores, extra), dim=-1)
        weights = torch.softmax(augmented, dim=-1)[..., :length]
        reference = torch.einsum("qht,thd->qhd", weights, v_cpu[:, head_map])
        diff = (actual - reference).abs()
        failures = (diff > 0.05 + 0.05 * reference.abs()).sum().item()
        print(mode, "mean_abs", diff.mean().item(), "max_abs", diff.max().item(),
              "failures", failures, "elements", diff.numel())
