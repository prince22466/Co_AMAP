#!/usr/bin/env python3
"""Isolated BF16 probes of the existing v17 kernels; no tuning or A/B changes."""
import argparse
import itertools
import json
import math

# M,K,N and transposed weight layout from profiling/v7/v7_observed_gemm_shapes.csv.
GEMM = {
    "mm_qkv": (2048, 2048, 2560),
    "mm_o": (2048, 2048, 2048),
    "mm_gate_up": (2048, 2048, 12288),
    "mm_down": (2048, 6144, 2048),
    "linear_decode": (32, 2048, 12288),
}
ATTN = {
    # query lengths, total lengths, segmented decode buffers
    "attn_4k_prefill": ([2048], [4096], 0),
    "attn_16k_prefill": ([2048], [16384], 0),
    "attn_4k_decode": ([1] * 32, [4096] * 32, 0),
    "attn_16k_decode": ([1] * 24, [16384] * 24, 0),
    "attn_4k_decode_3d": ([1] * 16, [4096] * 16, 16),
    "attn_4k_mixed": ([2048] + [1] * 32, [4096] * 33, 0),
    "attn_16k_mixed": ([2048] + [1] * 24, [16384] * 25, 0),
}
CASES = tuple(GEMM) + tuple(ATTN)


def attention_payload(qlens, lengths, block_q):
    # Idealized causal query-block payload, before L2 reuse; not hardware bytes.
    return 1024 * sum(
        sum(s - q + min(i + block_q, q) for i in range(0, q, block_q))
        for q, s in zip(qlens, lengths)
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("case", choices=CASES)
    p.add_argument("--layout", choices=["shuffled", "contiguous"], default="shuffled")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--native", action="store_true")
    a = p.parse_args()
    if a.warmup < 1 or a.repeats < 1:
        p.error("warmup and repeats must be positive")

    import torch
    from vllm_fl.platform import PlatformFL
    PlatformFL.pre_register_and_update()
    torch.manual_seed(20261119)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-compatible BI-V150 device required")
    device = "cuda"
    dtype = torch.bfloat16

    if a.case in GEMM:
        m, k, n = GEMM[a.case]
        x = torch.randn((m, k), device=device, dtype=dtype)
        w = torch.randn((n, k), device=device, dtype=dtype)
        if a.case.startswith("linear"):
            from flag_gems.runtime.backend._iluvatar.ops.linear import linear
            fn = lambda: linear(x, w)
        else:
            from flag_gems.runtime.backend._iluvatar.ops.mm import mm
            fn = lambda: mm(x, w.T)
        unique = 2 * (m*k + k*n + m*n)
        info = dict(M=m, K=k, N=n, input_bytes=2*m*k,
                    weight_bytes=2*k*n, output_bytes=2*m*n,
                    compulsory_tensor_bytes=unique, flops=2*m*k*n,
                    ideal_flops_per_byte=2*m*k*n/unique)
    else:
        from vllm_fl.dispatch.backends.triton_unified_attention import unified_attention
        qlens, lengths, segments = ATTN[a.case]
        page, h, qh, kh = 16, 128, 16, 2
        counts = [math.ceil(s/page)+1 for s in lengths]  # padded page-table tail
        pages = sum(counts)
        q = torch.randn((sum(qlens), qh, h), device=device, dtype=dtype)
        k = torch.randn((pages, page, kh, h), device=device, dtype=dtype)
        v = torch.randn_like(k)
        order = (torch.randperm(pages, device=device) if a.layout == "shuffled"
                 else torch.arange(pages, device=device)).to(torch.int32)
        table = torch.zeros((len(counts), max(counts)), dtype=torch.int32, device=device)
        offset = 0
        for i, count in enumerate(counts):
            table[i, :count] = order[offset:offset+count]
            offset += count
        out = torch.empty_like(q)
        args = dict(q=q, k=k, v=v, out=out,
                    cu_seqlens_q=torch.tensor([0, *itertools.accumulate(qlens)],
                                              dtype=torch.int32, device=device),
                    max_seqlen_q=max(qlens),
                    seqused_k=torch.tensor(lengths, dtype=torch.int32, device=device),
                    max_seqlen_k=max(lengths), softmax_scale=h**-0.5,
                    causal=True, window_size=(-1, -1), block_table=table,
                    softcap=0., q_descale=None, k_descale=None, v_descale=None)
        if segments:
            shape = (sum(qlens), qh, segments)
            args.update(seq_threshold_3D=64, num_par_softmax_segments=segments,
                        softmax_segm_output=torch.empty((*shape, h), device=device,
                                                        dtype=torch.float32),
                        softmax_segm_max=torch.empty(shape, device=device,
                                                     dtype=torch.float32),
                        softmax_segm_expsum=torch.empty(shape, device=device,
                                                        dtype=torch.float32))
        fn = lambda: unified_attention(**args)
        bq = 8 if max(qlens)>1 else 2
        info = dict(query_lengths=qlens, seq_lengths=lengths, layout=a.layout,
                    segments_requested=segments, unique_kv_payload_bytes=1024*sum(lengths),
                    causal_query_block_payload_proxy_bytes=attention_payload(qlens, lengths, bq),
                    query_block_size_proxy=bq,
                    padded_query_positions_proxy=sum(math.ceil(qi/bq)*bq-qi for qi in qlens),
                    q_bytes=q.numel()*q.element_size(), output_bytes=out.numel()*out.element_size())
    with torch.inference_mode():
        for _ in range(a.warmup):
            fn()
        torch.cuda.synchronize()
        if a.native:
            # IXKN --profile-from-start off isolates the warmed call. Fail
            # explicitly if this runtime cannot provide the profiler API.
            torch.cuda.profiler.start()
            try:
                fn()
                torch.cuda.synchronize()
            finally:
                torch.cuda.profiler.stop()
            info["native_region_completed"] = True
        else:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(a.repeats):
                fn()
            end.record()
            end.synchronize()
            info["event_ms_per_call"] = start.elapsed_time(end)/a.repeats
    print("PROBE_JSON " + json.dumps(dict(case=a.case, **info)), flush=True)


if __name__ == "__main__":
    main()
