#!/usr/bin/env python3
"""Group11: Triton compiler optimization A/B experiments.

K01 loop unroll; K02 automatic layout/tile; K03 source-level live ranges;
K04 loop invariant code motion; optional K02G Gluon explicit layouts.
The generated backend stages, not source appearances, decide whether a change
is a compiler optimization. No competition benchmark scripts are modified.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
from compiler_lab_common import (
    check_tensor, compiler_metadata, measure, parse_ints, require_runtime,
    save_compiler_artifacts, summarise, write_csv, write_environment,
)
from compiler_reference import (
    assert_gpu_warp_size, base_arrays, fma_reference, licm_reference,
    live_range_reference, parse_tile_pairs,
)

try:
    import triton
    import triton.language as tl
except ImportError:
    raise SystemExit('Triton is required on the BI-V150 test machine')


@triton.jit
def unroll_kernel(X,A,B,Y,N:tl.constexpr,BLOCK:tl.constexpr,ITERS:tl.constexpr,UNROLL:tl.constexpr):
    idx=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    m=idx<N
    acc=tl.load(X+idx,m,other=0)
    a=tl.load(A+idx,m,other=1)
    b=tl.load(B+idx,m,other=0)
    for _ in tl.range(0,ITERS,loop_unroll_factor=UNROLL):
        acc=tl.fma(acc,a,b)
    tl.store(Y+idx,acc,m)


@triton.jit
def transpose_layout_kernel(X,Y,M:tl.constexpr,N:tl.constexpr,BR:tl.constexpr,BC:tl.constexpr):
    rr=tl.program_id(0)*BR+tl.arange(0,BR)
    cc=tl.program_id(1)*BC+tl.arange(0,BC)
    mask=(rr[:,None]<M)&(cc[None,:]<N)
    src=tl.load(X+rr[:,None]*N+cc[None,:],mask,other=0)
    # Logical transpose; compiler chooses hardware lane/warp distribution.
    tl.store(Y+cc[None,:]*M+rr[:,None],src,mask)


@triton.jit
def live_range_kernel(X,A,B,Y,N:tl.constexpr,BLOCK:tl.constexpr,ITERS:tl.constexpr,EARLY:tl.constexpr):
    idx=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    mask=idx<N
    a=tl.load(A+idx,mask,other=1)
    b=tl.load(B+idx,mask,other=0)
    acc=tl.load(X+idx,mask,other=0)
    if EARLY:
        v0=tl.load(X+idx+0*N,mask,other=0)
        v1=tl.load(X+idx+1*N,mask,other=0)
        v2=tl.load(X+idx+2*N,mask,other=0)
        v3=tl.load(X+idx+3*N,mask,other=0)
        v4=tl.load(X+idx+4*N,mask,other=0)
        v5=tl.load(X+idx+5*N,mask,other=0)
        v6=tl.load(X+idx+6*N,mask,other=0)
        v7=tl.load(X+idx+7*N,mask,other=0)
    for _ in range(ITERS):
        acc=tl.fma(acc,a,b)
    if not EARLY:
        v0=tl.load(X+idx+0*N,mask,other=0)
        v1=tl.load(X+idx+1*N,mask,other=0)
        v2=tl.load(X+idx+2*N,mask,other=0)
        v3=tl.load(X+idx+3*N,mask,other=0)
        v4=tl.load(X+idx+4*N,mask,other=0)
        v5=tl.load(X+idx+5*N,mask,other=0)
        v6=tl.load(X+idx+6*N,mask,other=0)
        v7=tl.load(X+idx+7*N,mask,other=0)
    # Deliberately identical source order of FP32 additions across variants.
    out=acc+v0
    out=out+v1
    out=out+v2
    out=out+v3
    out=out+v4
    out=out+v5
    out=out+v6
    out=out+v7
    tl.store(Y+idx,out,mask)


@triton.jit
def licm_kernel(X,A,B,Y,N:tl.constexpr,BLOCK:tl.constexpr,ITERS:tl.constexpr,DISABLE_LICM:tl.constexpr):
    idx=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    mask=idx<N
    acc=tl.load(X+idx,mask,other=0)
    a=tl.load(A+idx,mask,other=0)
    b=tl.load(B+idx,mask,other=1)
    # Compiler can hoist tl.exp(a) because a is loop-invariant.
    for _ in tl.range(0,ITERS,disable_licm=DISABLE_LICM):
        e=tl.exp(a)
        acc=tl.fma(acc,b,e)
    tl.store(Y+idx,acc,mask)


# Optional explicit layout control: Gluon import and runtime compiler support
# are independent of base Triton support on the particular GPU backend.
try:
    import triton.experimental.gluon as g
    import triton.experimental.gluon.language as gl
except ImportError:
    g=None
    gl=None

if g is not None:
    @g.jit
    def gluon_layout_kernel(X,Y,N:gl.constexpr,BLOCK:gl.constexpr,
                            VEC:gl.constexpr,LANES:gl.constexpr,WARPS:gl.constexpr):
        layout:gl.constexpr=gl.BlockedLayout([VEC],[LANES],[WARPS],[0])
        idx=gl.program_id(0)*BLOCK+gl.arange(0,BLOCK,layout=layout)
        m=idx<N
        x=gl.load(X+idx,m,other=0)
        gl.store(Y+idx,x*1.001+0.25,m)
else:
    gluon_layout_kernel=None


def cpu_data(torch, n, seed):
    x,a,b=base_arrays(n,seed)
    return (x,a,b),(torch.from_numpy(x).to('cuda'),torch.from_numpy(a).to('cuda'),torch.from_numpy(b).to('cuda'))


def make_record(lab, variant, extra=None):
    return {'lab':lab, 'variant':variant, **(extra or {})}


def run_one(lab, variant, args, torch, triton_mod, launch, output, expected,
            records, config, atol=.002, rtol=.001):
    row=make_record(lab,variant,config)
    try:
        output.fill_(float('nan'))
        compiled=launch()
        row['max_abs_error']=check_tensor(torch,output,expected,atol=atol,rtol=rtol)
        row.update(compiler_metadata(compiled))
        safe=''.join(c if c.isalnum() or c in '._-' else '_' for c in variant)
        row.update(save_compiler_artifacts(compiled,args.dump_asm,f'{lab}_{safe}'))
        row.update(measure(triton_mod,launch,warmup_ms=args.warmup_ms,
                           rep_ms=args.rep_ms,trials=args.trials,single_run=args.single_run))
        row['status']='ok'
        print(f'PASS {lab:4s} {variant:22s} time={row["latency_us"]}us '
              f'regs={row["registers_per_thread"]} spills={row["spills"]} '
              f'stage={row["asm_stage"]}',flush=True)
    except Exception as exc:
        # Unsupported compiler knobs must be visible; do not silently swap variants.
        row.update(status='error',error=f'{type(exc).__name__}: {exc}')
        print(f'FAIL {lab} {variant}: {row["error"]}',flush=True)
        if args.strict:
            records.append(row); write_csv(args.out,records)
            raise
    records.append(row)


def should_run(args,variant):
    return args.variant is None or args.variant==variant


def run_k01(args,torch,triton_mod,records):
    host,(x,a,b)=cpu_data(torch,args.n,args.seed)
    expect=torch.from_numpy(fma_reference(*host,args.iters))
    y=torch.empty_like(x)
    for u in parse_ints(args.unrolls,label='unroll factors',powers_of_two=True):
        variant=f'unroll{u}'
        if not should_run(args,variant): continue
        launch=lambda u=u: unroll_kernel[(triton_mod.cdiv(args.n,args.block),)](
            x,a,b,y,args.n,args.block,args.iters,u,num_warps=args.warps)
        run_one('K01',variant,args,torch,triton_mod,launch,y,expect,records,
                {'n':args.n,'block':args.block,'warps':args.warps,'iters':args.iters,'unroll':u})


def run_k02(args,torch,triton_mod,records):
    m,n=args.rows,args.cols
    xhost=np.arange(m*n,dtype=np.float32).reshape(m,n)*0.0001
    x=torch.from_numpy(xhost).to('cuda')
    y=torch.empty((n,m),dtype=torch.float32,device='cuda')
    expect=torch.from_numpy(xhost.T.copy())
    for br,bc in parse_tile_pairs(args.tiles):
        variant=f'tile{br}x{bc}'
        if not should_run(args,variant): continue
        launch=lambda br=br,bc=bc: transpose_layout_kernel[(triton_mod.cdiv(m,br),triton_mod.cdiv(n,bc))](
            x,y,m,n,br,bc,num_warps=args.warps)
        run_one('K02',variant,args,torch,triton_mod,launch,y,expect,records,
                {'rows':m,'cols':n,'block_r':br,'block_c':bc,'warps':args.warps},atol=0,rtol=0)


def run_k03(args,torch,triton_mod,records):
    rng=np.random.default_rng(args.seed)
    n=args.n
    xhost=rng.uniform(-.05,.05,size=(8,n)).astype(np.float32)
    ahost=rng.uniform(.998,1.001,size=n).astype(np.float32)
    bhost=rng.uniform(.0001,.0003,size=n).astype(np.float32)
    expect=torch.from_numpy(live_range_reference(xhost,ahost,bhost,args.live_iters))
    x=torch.from_numpy(xhost.reshape(-1)).to('cuda')
    a=torch.from_numpy(ahost).to('cuda')
    b=torch.from_numpy(bhost).to('cuda')
    y=torch.empty_like(a)
    for variant,early in [('early_loads',True),('late_loads',False)]:
        if not should_run(args,variant): continue
        launch=lambda early=early: live_range_kernel[(triton_mod.cdiv(n,args.block),)](
            x,a,b,y,n,args.block,args.live_iters,early,num_warps=args.warps)
        run_one('K03',variant,args,torch,triton_mod,launch,y,expect,records,
                {'n':n,'block':args.block,'warps':args.warps,'live_iters':args.live_iters,'early':early})


def run_k04(args,torch,triton_mod,records):
    host,(x,a,b)=cpu_data(torch,args.n,args.seed)
    # exp input should be modest and FMA multiplier close to 1; custom A and B.
    rng=np.random.default_rng(args.seed+5)
    ahost=rng.uniform(.10,.30,size=args.n).astype(np.float32)
    bhost=rng.uniform(.995,.999,size=args.n).astype(np.float32)
    a=torch.from_numpy(ahost).to('cuda')
    b=torch.from_numpy(bhost).to('cuda')
    expect=torch.from_numpy(licm_reference(host[0],ahost,bhost,args.licm_iters))
    y=torch.empty_like(x)
    for variant,disable in [('licm_enabled',False),('licm_disabled',True)]:
        if not should_run(args,variant): continue
        launch=lambda disable=disable: licm_kernel[(triton_mod.cdiv(args.n,args.block),)](
            x,a,b,y,args.n,args.block,args.licm_iters,disable,num_warps=args.warps)
        run_one('K04',variant,args,torch,triton_mod,launch,y,expect,records,
                {'n':args.n,'block':args.block,'warps':args.warps,'licm_iters':args.licm_iters,'disable_licm':disable},
                atol=.02,rtol=.002)


def run_k02g(args,torch,triton_mod,records):
    if gluon_layout_kernel is None:
        print('SKIP K02G: installed Triton lacks experimental Gluon')
        records.append({'lab':'K02G','variant':'all','status':'unsupported','error':'Gluon not importable'})
        return
    props=torch.cuda.get_device_properties(0)
    lane_value=getattr(props,'warp_size',None)
    if lane_value is None:
        # The first compiled base kernel provides vendor-specific warp_size metadata.
        h,(x0,a,b)=cpu_data(torch,256,args.seed)
        y0=torch.empty_like(x0)
        compiled=unroll_kernel[(1,)](x0,a,b,y0,256,256,2,1,num_warps=4)
        lane_value=getattr(getattr(compiled,'metadata',None),'warp_size',None)
    if lane_value is None:
        print('SKIP K02G: cannot determine device warp size safely')
        records.append({'lab':'K02G','variant':'all','status':'unsupported','error':'warp size unavailable'})
        return
    lanes=assert_gpu_warp_size(int(lane_value))
    n=args.n
    rng=np.random.default_rng(args.seed)
    host=rng.uniform(.1,.9,size=n).astype(np.float32)
    x=torch.from_numpy(host).to('cuda')
    y=torch.empty_like(x)
    expect=torch.from_numpy((host.astype(np.float64)*1.001+.25).astype(np.float32))
    for v in parse_ints(args.vector_widths,label='vector widths',powers_of_two=True):
        variant=f'vec{v}'
        if not should_run(args,variant): continue
        if v*lanes*args.warps > args.gluon_block:
            records.append({'lab':'K02G','variant':variant,'status':'unsupported',
                            'error':f'block={args.gluon_block} < vector*lanes*warps={v*lanes*args.warps}'})
            continue
        launch=lambda v=v: gluon_layout_kernel[(triton_mod.cdiv(n,args.gluon_block),)](
            x,y,n,args.gluon_block,v,lanes,args.warps,num_warps=args.warps)
        run_one('K02G',variant,args,torch,triton_mod,launch,y,expect,records,
                {'n':n,'block':args.gluon_block,'warps':args.warps,'lanes':lanes,'vec':v},atol=1e-5,rtol=1e-5)


def make_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lab',default='all',choices=('all','K01','K02','K03','K04','K02G'))
    p.add_argument('--variant',help='Exact variant name; useful for targeted profiler captures')
    p.add_argument('--n',type=int,default=262144)
    p.add_argument('--rows',type=int,default=1025)
    p.add_argument('--cols',type=int,default=1537)
    p.add_argument('--block',type=int,default=256)
    p.add_argument('--warps',type=int,default=4)
    p.add_argument('--iters',type=int,default=256)
    p.add_argument('--unrolls',default='1,2,4,8')
    p.add_argument('--tiles',default='16x64,32x32,64x16')
    p.add_argument('--live-iters',type=int,default=128)
    p.add_argument('--licm-iters',type=int,default=64)
    p.add_argument('--vector-widths',default='1,2,4')
    p.add_argument('--gluon-block',type=int,default=1024)
    p.add_argument('--warmup-ms',type=int,default=50)
    p.add_argument('--rep-ms',type=int,default=150)
    p.add_argument('--trials',type=int,default=3)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--single-run',action='store_true',help='Check compilation and correctness, without timed benchmarking')
    p.add_argument('--strict',action='store_true',help='Stop at the first failed configuration')
    p.add_argument('--dump-asm',default='results/asm',metavar='DIR',
                   help='Save all textual and binary compiler stages exposed by the backend')
    p.add_argument('--out',default='results/compiler_results.csv')
    return p


def main():
    p=make_parser(); args=p.parse_args()
    for key in ('n','rows','cols','block','warps','iters','live_iters','licm_iters','gluon_block','trials','warmup_ms','rep_ms'):
        if getattr(args,key)<=0: p.error(f'--{key.replace("_","-")} must be > 0')
    if any(v & (v-1) for v in (args.block,args.warps,args.gluon_block)):
        p.error('block, warps and gluon-block must be powers of two')
    if args.lab=='all' and args.variant:
        p.error('--variant requires --lab with one specific experiment')
    try:
        parse_tile_pairs(args.tiles)
        parse_ints(args.unrolls,label='unrolls',powers_of_two=True)
        parse_ints(args.vector_widths,label='vector-widths',powers_of_two=True)
    except ValueError as exc:
        p.error(str(exc))
    torch,triton_mod=require_runtime()
    print(f'GPU={torch.cuda.get_device_name(0)} torch={torch.__version__} '
          f'triton={triton_mod.__version__}',flush=True)
    write_environment(args.out,torch,triton_mod,args)
    records=[]
    jobs={'K01':run_k01,'K02':run_k02,'K03':run_k03,'K04':run_k04,'K02G':run_k02g}
    targets=['K01','K02','K03','K04'] if args.lab=='all' else [args.lab]
    for name in targets:
        print(f'== {name} ==',flush=True)
        jobs[name](args,torch,triton_mod,records)
    if args.variant and not any(r['variant']==args.variant for r in records):
        raise SystemExit(f'Unknown variant: {args.variant!r}')
    write_csv(args.out,records)
    summarise(records)
    if not any(r.get('status')=='ok' for r in records):
        raise SystemExit('No successful configurations; inspect errors in CSV')
    if any(r.get('status')=='error' for r in records):
        raise SystemExit('Some configurations failed; inspect errors in CSV')


if __name__=='__main__': main()
