#!/usr/bin/env python3
"""Standalone data-oriented Triton A/B experiments; no vLLM or FlagGems changes.
Run --lab all for correctness and paired timing; --lab D02 --variant optimized
--native for one warmed profiler region. Optimized is a hypothesis, not a claim.
"""
import argparse
import csv
import json
import statistics
import torch
import triton
import triton.language as tl


@triton.jit
def copy_kernel(X, Y, N, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    """Purpose: copy with selectable stride and tile. Inputs: X,N,stride,block.
    Output: N contiguous values in Y."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i * STRIDE, i < N, other=0)
    tl.store(Y + i, x, i < N)


@triton.jit
def transpose_naive(X, Y, M: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    """Purpose: read flat contiguous chunks and scatter. Inputs: X[M,N],sizes.
    Output: contiguous Y[N,M]."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < M * N, other=0)
    tl.store(Y + (i % N) * M + i // N, x, i < M * N)


@triton.jit
def transpose_tiled(X, Y, M: tl.constexpr, N: tl.constexpr, T: tl.constexpr):
    """Purpose: express transpose in square tiles. Inputs: X[M,N], sizes, T.
    Output: Y[N,M]; compiler determines actual shared-memory layout."""
    r = tl.program_id(0) * T + tl.arange(0, T)
    c = tl.program_id(1) * T + tl.arange(0, T)
    mask = (r[:, None] < M) & (c[None, :] < N)
    x = tl.load(X + r[:, None] * N + c[None, :], mask, other=0)
    tl.store(Y + c[:, None] * M + r[None, :], tl.trans(x), tl.trans(mask))

@triton.jit
def silu_kernel(X, Y, N, BLOCK: tl.constexpr):
    """Purpose: first half of an unfused SiLU-and-multiply operation.
    Inputs: X, N elements, BLOCK elements/program. Output: writes SiLU(X) to Y.
    """
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0)
    tl.store(Y + i, x / (1.0 + tl.exp(-x)), i < N)


@triton.jit
def mul_kernel(X, G, Y, N, BLOCK: tl.constexpr):
    """Purpose: second half of the unfused operation.
    Inputs: X, G, N, BLOCK. Output: writes X * G to Y.
    """
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0)
    g = tl.load(G + i, i < N, other=0)
    tl.store(Y + i, x * g, i < N)


@triton.jit
def fused_kernel(X, G, Y, N, BLOCK: tl.constexpr):
    """Purpose: fuse SiLU and multiply, eliminating the intermediate buffer.
    Inputs: X, G, N, BLOCK. Output: writes SiLU(X) * G to Y.
    """
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0)
    g = tl.load(G + i, i < N, other=0)
    tl.store(Y + i, (x / (1.0 + tl.exp(-x))) * g, i < N)


@triton.jit
def gemm_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr,
                K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                BK: tl.constexpr, GROUP: tl.constexpr):
    """Purpose: tiled matrix multiply; compare row-major and grouped tile IDs.
    Inputs: contiguous A[M,K], B[K,N], sizes, tile sizes, GROUP (1=row-major).
    Output: writes C[M,N] with FP32 accumulation and output dtype conversion.
    """
    pid = tl.program_id(0)
    gm, gn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    if GROUP == 1:
        pm, pn = pid // gn, pid % gn
    else:
        width = GROUP * gn
        first = (pid // width) * GROUP
        size = tl.minimum(gm - first, GROUP)
        pm = first + (pid % width) % size
        pn = (pid % width) // size

    m = pm * BM + tl.arange(0, BM)
    n = pn * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for step in range(tl.cdiv(K, BK)):
        kk = step * BK + k
        a = tl.load(A + m[:, None] * K + kk[None, :],
                    (m[:, None] < M) & (kk[None, :] < K), other=0)
        b = tl.load(B + kk[:, None] * N + n[None, :],
                    (kk[:, None] < K) & (n[None, :] < N), other=0)
        acc = tl.dot(a, b, acc)
    tl.store(C + m[:, None] * N + n[None, :], acc,
             (m[:, None] < M) & (n[None, :] < N))


@triton.jit
def kv_tokenwise(K, V, TABLE, OK, OV, L: tl.constexpr, D: tl.constexpr,
                 P: tl.constexpr, NP: tl.constexpr):
    """Purpose: one token/program, repeat page lookup. Inputs: paged K,V, table.
    Output: dense K,V in logical token order."""
    t = tl.program_id(0)
    b = tl.program_id(1)
    d = tl.arange(0, D)
    p = tl.load(TABLE + b * NP + t // P)
    src = (p * P + t % P) * D + d
    dst = (b * L + t) * D + d
    tl.store(OK + dst, tl.load(K + src))
    tl.store(OV + dst, tl.load(V + src))


@triton.jit
def kv_pagewise(K, V, TABLE, OK, OV, L: tl.constexpr, D: tl.constexpr,
                P: tl.constexpr, NP: tl.constexpr):
    """Purpose: one page/program, reuse physical page lookup. Inputs: same as tokenwise.
    Output: identical dense K,V; masks the last logical page."""
    page = tl.program_id(0)
    b = tl.program_id(1)
    p = tl.load(TABLE + b * NP + page)
    i = tl.arange(0, P * D)
    t = page * P + i // D
    dst = (b * L + t) * D + i % D
    tl.store(OK + dst, tl.load(K + p * P * D + i, t < L, other=0), t < L)
    tl.store(OV + dst, tl.load(V + p * P * D + i, t < L, other=0), t < L)


@triton.jit
def attention_scores(Q, K, S, L: tl.constexpr, D: tl.constexpr, T: tl.constexpr):
    """Purpose: materialize scaled QK scores. Inputs: Q[B,D], K[B,L,D].
    Output: S[B,L]; noncausal single-query decode over valid keys."""
    b, p = tl.program_id(0), tl.program_id(1)
    t = p * T + tl.arange(0, T)
    d = tl.arange(0, D)
    q = tl.load(Q + b * D + d)
    k = tl.load(K + (b * L + t[:, None]) * D + d[None, :], t[:, None] < L, other=0)
    s = tl.sum(k * q[None, :], 1) * (D ** -0.5)
    tl.store(S + b * L + t, s, t < L)


@triton.jit
def attention_probs(S, W, MAX, SUM, L: tl.constexpr, NP: tl.constexpr, T: tl.constexpr):
    """Purpose: materialize segment-local softmax numerators. Inputs: scores S.
    Output: W[B,L], per-segment maxima and sums."""
    b, p = tl.program_id(0), tl.program_id(1)
    t = p * T + tl.arange(0, T)
    s = tl.load(S + b * L + t, t < L, other=-float('inf'))
    m = tl.max(s, 0)
    w = tl.exp(s - m)
    tl.store(W + b * L + t, w, t < L)
    tl.store(MAX + b * NP + p, m)
    tl.store(SUM + b * NP + p, tl.sum(w, 0))


@triton.jit
def attention_weighted(W, V, PART, L: tl.constexpr, D: tl.constexpr,
                       NP: tl.constexpr, T: tl.constexpr):
    """Purpose: apply materialized weights to V. Inputs: W[B,L], V[B,L,D].
    Output: unnormalized weighted vector PART per segment."""
    b, p = tl.program_id(0), tl.program_id(1)
    t = p * T + tl.arange(0, T)
    d = tl.arange(0, D)
    w = tl.load(W + b * L + t, t < L, other=0)
    v = tl.load(V + (b * L + t[:, None]) * D + d[None, :], t[:, None] < L, other=0)
    tl.store(PART + (b * NP + p) * D + d, tl.sum(w[:, None] * v, 0))


@triton.jit
def attention_online(Q, K, V, MAX, SUM, PART, L: tl.constexpr, D: tl.constexpr,
                     NP: tl.constexpr, T: tl.constexpr):
    """Purpose: fuse score, local softmax and weighted V; no full S or W.
    Inputs: Q,K,V. Output: per-segment maxima, sums and weighted vectors."""
    b, p = tl.program_id(0), tl.program_id(1)
    t = p * T + tl.arange(0, T)
    d = tl.arange(0, D)
    q = tl.load(Q + b * D + d)
    k = tl.load(K + (b * L + t[:, None]) * D + d[None, :], t[:, None] < L, other=0)
    s = tl.sum(k * q[None, :], 1) * (D ** -0.5)
    s = tl.where(t < L, s, -float('inf'))
    m = tl.max(s, 0)
    w = tl.exp(s - m)
    v = tl.load(V + (b * L + t[:, None]) * D + d[None, :], t[:, None] < L, other=0)
    tl.store(MAX + b * NP + p, m)
    tl.store(SUM + b * NP + p, tl.sum(w, 0))
    tl.store(PART + (b * NP + p) * D + d, tl.sum(w[:, None] * v, 0))


@triton.jit
def attention_merge(MAX, SUM, PART, O, D: tl.constexpr, NP: tl.constexpr,
                    BP: tl.constexpr, BD: tl.constexpr):
    """Purpose: stable global softmax merge of segment-local statistics.
    Inputs: maxima, sums, weighted vectors. Output: exact (up to FP error) O[B,D]."""
    b, col = tl.program_id(0), tl.program_id(1)
    p = tl.arange(0, BP)
    d = col * BD + tl.arange(0, BD)
    m = tl.load(MAX + b * NP + p, p < NP, other=-float('inf'))
    z = tl.load(SUM + b * NP + p, p < NP, other=0)
    a = tl.exp(m - tl.max(m, 0))
    v = tl.load(PART + (b * NP + p[:, None]) * D + d[None, :],
                (p[:, None] < NP) & (d[None, :] < D), other=0)
    y = tl.sum(a[:, None] * v, 0) / tl.sum(a * z, 0)
    tl.store(O + b * D + d, y, d < D)


def check(launch, outputs, refs, atol, rtol):
    """Purpose: compile and verify output coverage and values. Inputs: launch and tensors.
    Output: None; raises on incorrect results."""
    for out in outputs:
        out.fill_(float('nan'))
    launch()
    torch.cuda.synchronize()
    for out, ref in zip(outputs, refs):
        torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


def timed(launch, repeats):
    """Purpose: time launches with device events. Inputs: callable, repeats.
    Output: mean milliseconds per invocation for this sample."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        launch()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def pair(args, lab, names, launches, outputs, refs, work, unit, logical_bytes,
         atol=1e-5, rtol=1e-5):
    """Purpose: check and compare a pair with alternating timing order.
    Inputs: names, launches, tensors, common work, rate unit, logical byte estimates.
    Output: result rows, optional single native capture; raises on failed correctness."""
    chosen = [0, 1] if args.variant is None else [int(args.variant == 'optimized')]
    for i in chosen:
        check(launches[i], outputs, refs, atol, rtol)
        for _ in range(3):
            launches[i]()
        torch.cuda.synchronize()
        print(f'{lab} {names[i]} correctness=PASS', flush=True)
    if args.native:
        i = chosen[0]
        torch.cuda.profiler.start()
        try:
            launches[i]()
            torch.cuda.synchronize()
        finally:
            torch.cuda.profiler.stop()
        print(f'PROFILE_OK {lab} {args.variant}', flush=True)
        return []
    samples = {i: [] for i in chosen}
    for trial in range(args.trials):
        for i in (chosen if trial % 2 == 0 else chosen[::-1]):
            # Identical warm state policy for both variants; cache is not flushed.
            launches[i]()
            torch.cuda.synchronize()
            samples[i].append(timed(launches[i], args.repeats))
    med = {i: statistics.median(samples[i]) for i in chosen}
    speedup = med[0] / med[1] if len(chosen) == 2 else None
    rows = []
    for i in chosen:
        ms = med[i]
        rate = work / (ms * (1e6 if unit == 'GB/s' else 1e9))
        row = dict(lab=lab, variant=['vanilla', 'optimized'][i], function=names[i],
                   ms=ms, min_ms=min(samples[i]), max_ms=max(samples[i]),
                   rate=rate, rate_unit=unit, logical_bytes=logical_bytes[i],
                   pair_speedup=speedup, correctness='PASS')
        rows.append(row)
        print(f'RESULT {json.dumps(row)}', flush=True)
    if speedup is not None:
        print(f'{lab} speedup={speedup:.3f}x (vanilla/optimized)', flush=True)
    return rows


def copy_lab(args, lab):
    """Purpose: D01 program size or D02 physical stride. Inputs: CLI configuration.
    Output: equal-value copy A/B measurements; layout construction excluded."""
    n = args.elements
    x = torch.randn(n, device='cuda')
    y = torch.empty_like(x)
    if lab == 'D01':
        src = [x, x]; strides = [1, 1]; blocks = [128, 1024]
        names = ['copy_small_block', 'copy_large_block']
    else:
        backing = torch.empty(n * 4, device='cuda')
        backing[::4].copy_(x)
        src = [backing, x]; strides = [4, 1]; blocks = [1024, 1024]
        names = ['copy_strided', 'copy_contiguous']
    launches = [lambda i=i: copy_kernel[(triton.cdiv(n, blocks[i]),)](
        src[i], y, n, strides[i], blocks[i], num_warps=4) for i in range(2)]
    return pair(args, lab, names, launches, [y], [x], 8*n, 'GB/s', [8*n]*2, 0, 0)


def transpose_lab(args):
    """Purpose: D03 compare flat scatter and tile transpose. Inputs: CLI sizes.
    Output: A/B timings and exact correctness."""
    m, n = args.transpose_m, args.transpose_n
    x = torch.randn((m, n), device='cuda')
    y = torch.empty((n, m), device='cuda')
    launches = [lambda: transpose_naive[(triton.cdiv(m*n, 1024),)](x,y,m,n,1024,num_warps=4),
                lambda: transpose_tiled[(triton.cdiv(m,32),triton.cdiv(n,32))](x,y,m,n,32,num_warps=4)]
    return pair(args,'D03',['transpose_naive','transpose_tiled'],launches,[y],[x.T],8*m*n,'GB/s',[8*m*n]*2,0,0)


def gemm_lab(args, lab):
    """Purpose: D04 grouped order or D06 pipeline request. Inputs: GEMM shape/dtype.
    Output: equal math, identical tile sizes, paired timing; no stages claim."""
    m,n,k = args.m,args.n,args.k
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float16
    a = torch.randn((m,k),device='cuda',dtype=dtype) / k**0.5
    b = torch.randn((k,n),device='cuda',dtype=dtype)
    y = torch.empty((m,n),device='cuda',dtype=dtype)
    ref = (a.float() @ b.float()).to(dtype)
    bm,bn,bk = 32,64,32
    groups, stages = ([1,8],[1,1]) if lab == 'D04' else ([8,8],[1,3])
    names = ['gemm_rowmajor','gemm_grouped'] if lab == 'D04' else ['gemm_single_stage','gemm_pipelined']
    grid = (triton.cdiv(m,bm)*triton.cdiv(n,bn),)
    launches = [lambda i=i: gemm_kernel[grid](a,b,y,m,n,k,bm,bn,bk,groups[i],num_warps=4,num_stages=stages[i]) for i in range(2)]
    # Compulsory tensor payload only: not measured HBM or all tile loads.
    payload = 2*(m*k+k*n+m*n)
    return pair(args,lab,names,launches,[y],[ref],2*m*n*k,'TFLOP/s',[payload]*2,0.02,0.02)


def fusion_lab(args):
    """Purpose: D05 materialized SiLU vs fused. Inputs: CLI vector length.
    Output: paired timing with preallocated intermediate and output."""
    n=args.elements; block=1024; grid=(triton.cdiv(n,block),)
    x=torch.randn(n,device='cuda'); g=torch.randn_like(x)
    temp=torch.empty_like(x); y=torch.empty_like(x)
    def unfused():
        """Purpose: launch SiLU then multiply. Inputs: captured buffers. Output: writes y."""
        silu_kernel[grid](x,temp,n,block,num_warps=4)
        mul_kernel[grid](temp,g,y,n,block,num_warps=4)
    def fused():
        """Purpose: one fused launch. Inputs: captured buffers. Output: writes y."""
        fused_kernel[grid](x,g,y,n,block,num_warps=4)
    return pair(args,'D05',['silu_mul_unfused','silu_mul_fused'],[unfused,fused],[y],
                [torch.nn.functional.silu(x)*g],12*n,'GB/s',[20*n,12*n],2e-5,2e-5)


def kv_lab(args):
    """Purpose: D07 compare token vs page granularity with identical shuffled pages.
    Inputs: batch, context length; output: exact dense K and V, paired timing."""
    b,l,d,p=args.batch,args.context,128,16; np=triton.cdiv(l,p)
    shape=(b*np,p,d); k=torch.randn(shape,device='cuda'); v=torch.randn_like(k)
    table=torch.randperm(b*np,device='cuda',dtype=torch.int32).reshape(b,np)
    ok=torch.empty((b,l,d),device='cuda'); ov=torch.empty_like(ok)
    refs=[x[table.long()].reshape(b,np*p,d)[:,:l,:] for x in (k,v)]
    launches=[lambda: kv_tokenwise[(l,b)](k,v,table,ok,ov,l,d,p,np,num_warps=4),
              lambda: kv_pagewise[(np,b)](k,v,table,ok,ov,l,d,p,np,num_warps=4)]
    # Two inputs, two outputs; table access excluded from common rate.
    payload=16*b*l*d
    return pair(args,'D07',['kv_gather_tokenwise','kv_gather_pagewise'],launches,[ok,ov],refs,
                payload,'GB/s',[payload+4*b*l,payload+4*b*np],0,0)


def attention_lab(args):
    """Purpose: D08 segmented exact single-query attention with/without S,W buffers.
    Inputs: batch and context. Output: equal attention result and paired timing.
    This is a teaching proxy, not a production paged/GQA attention backend."""
    b,l,d,t=args.batch,args.context,128,32; np=triton.cdiv(l,t)
    q=torch.randn((b,d),device='cuda'); k=torch.randn((b,l,d),device='cuda'); v=torch.randn_like(k)
    s=torch.empty((b,l),device='cuda'); w=torch.empty_like(s)
    mx=torch.empty((b,np),device='cuda'); z=torch.empty_like(mx)
    part=torch.empty((b,np,d),device='cuda'); out=torch.empty_like(q)
    scores=(k*q[:,None,:]).sum(-1)*d**-0.5
    ref=(scores.softmax(-1)[:,:,None]*v).sum(1)
    grid=(b,np); mg=(b,triton.cdiv(d,16))
    def merge():
        """Purpose: launch common global softmax merge. Inputs: captured stats. Output: out."""
        attention_merge[mg](mx,z,part,out,d,np,triton.next_power_of_2(np),16,num_warps=4)
    def vanilla():
        """Purpose: materialize scores and local weights, then merge. Inputs: Q,K,V. Output: out."""
        attention_scores[grid](q,k,s,l,d,t,num_warps=4)
        attention_probs[grid](s,w,mx,z,l,np,t,num_warps=4)
        attention_weighted[grid](w,v,part,l,d,np,t,num_warps=4)
        merge()
    def optimized():
        """Purpose: fuse local attention, then merge. Inputs: Q,K,V. Output: out."""
        attention_online[grid](q,k,v,mx,z,part,l,d,np,t,num_warps=4)
        merge()
    # Common useful bytes: Q,K,V,O once. Partial buffers and rereads not included.
    payload=4*b*d*(2*l+2)
    # Estimate requested tensor payload by stages, including repeated Q and merge stats.
    shared=4*(2*b*l*d+b*np*d+2*b*np+b*np*d+2*b*np*(d//16)+b*d+b*np*d)
    return pair(args,'D08',['attention_materialized','attention_online'],[vanilla,optimized],[out],[ref],
                payload,'GB/s',[shared+16*b*l,shared],1e-4,1e-4)


def main():
    """Purpose: parse options and run independent A/B labs. Inputs: CLI.
    Output: JSON lines, optional CSV; exit nonzero if any selected experiment fails."""
    p=argparse.ArgumentParser(description=__doc__)
    labs={f'D{i:02d}':None for i in range(1,9)}
    p.add_argument('--lab',choices=['all',*labs],default='all')
    p.add_argument('--variant',choices=['vanilla','optimized'])
    p.add_argument('--native',action='store_true')
    p.add_argument('--elements',type=int,default=8*1024*1024+3)
    p.add_argument('--transpose-m',type=int,default=2049);p.add_argument('--transpose-n',type=int,default=4099)
    p.add_argument('--m',type=int,default=2048);p.add_argument('--n',type=int,default=4096);p.add_argument('--k',type=int,default=2048)
    p.add_argument('--dtype',choices=['bf16','fp16'],default='bf16')
    p.add_argument('--batch',type=int,default=16);p.add_argument('--context',type=int,default=4096)
    p.add_argument('--trials',type=int,default=5);p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--csv',help='Write successful timing rows to this path')
    args=p.parse_args()
    if min(args.elements,args.transpose_m,args.transpose_n,args.m,args.n,args.k,args.batch,args.context,args.trials,args.repeats)<1:
        p.error('all sizes and counts must be positive')
    if args.native and (args.lab=='all' or not args.variant): p.error('--native needs one lab and variant')
    if args.context>16384 and args.lab in ('all','D08'): p.error('D08 teaching implementation limited to context <= 16384')
    if not torch.cuda.is_available(): p.error('a compatible GPU runtime is required')
    torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False
    print(f'GPU={torch.cuda.get_device_name(0)} torch={torch.__version__} triton={triton.__version__}',flush=True)
    print(f'CONFIG {json.dumps(vars(args))}',flush=True)
    run={'D01':lambda:copy_lab(args,'D01'),'D02':lambda:copy_lab(args,'D02'),'D03':lambda:transpose_lab(args),
         'D04':lambda:gemm_lab(args,'D04'),'D05':lambda:fusion_lab(args),'D06':lambda:gemm_lab(args,'D06'),
         'D07':lambda:kv_lab(args),'D08':lambda:attention_lab(args)}
    rows=[]; failures=[]
    for lab in (run if args.lab=='all' else [args.lab]):
        torch.manual_seed(42)
        try:
            rows.extend(run[lab]())
        except Exception as e:
            failures.append(lab)
            print(f'FAILED {lab}: {type(e).__name__}: {e}',flush=True)
            if args.lab!='all': raise
    if args.csv and rows:
        with open(args.csv,'w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if failures: raise SystemExit('Failed experiments: '+','.join(failures))


if __name__=='__main__': main()
