"""Experimental exact FP32 divide/argmax reduction for sampler-owned buffers."""
import torch
import triton
import triton.language as tl

@triton.jit
def part(P,Q,V,I, N:tl.constexpr, C:tl.constexpr, BLOCK:tl.constexpr):
 row=tl.program_id(0); ch=tl.program_id(1)
 idx=ch*BLOCK+tl.arange(0,BLOCK); valid=idx<N
 p=tl.load(P+row*N+idx,valid,0); q=tl.load(Q+row*N+idx,valid,1)
 val=tl.div_rn(p,q)
 nan=valid & (val != val)
 has=tl.sum(nan.to(tl.int32),0)>0
 vmax=tl.max(tl.where(valid & ~nan,val,float('-inf')),0)
 chosen=tl.min(tl.where(valid & tl.where(has,nan,val==vmax),idx,2147483647),0)
 tl.store(V+row*C+ch,tl.where(has,float('nan'),vmax))
 tl.store(I+row*C+ch,chosen)
@triton.jit
def finish(V,I,O,C:tl.constexpr,BLOCK:tl.constexpr):
 row=tl.program_id(0);j=tl.arange(0,BLOCK); valid=j<C
 v=tl.load(V+row*C+j,valid,float('-inf'));i=tl.load(I+row*C+j,valid,2147483647)
 nan=valid & (v!=v);has=tl.sum(nan.to(tl.int32),0)>0
 vmax=tl.max(tl.where(valid & ~nan,v,float('-inf')),0)
 ind=tl.min(tl.where(valid & tl.where(has,nan,v==vmax),i,2147483647),0)
 tl.store(O+row,ind.to(tl.int64))

def sample(probs, noise):
    """Return argmax(probs / noise) without modifying either input."""
    rows, vocab = probs.shape
    chunks = triton.cdiv(vocab, 2048)
    values = torch.empty((rows, chunks), dtype=torch.float32, device=probs.device)
    indices = torch.empty((rows, chunks), dtype=torch.int32, device=probs.device)
    result = torch.empty((rows,), dtype=torch.int64, device=probs.device)
    part[(rows, chunks)](probs, noise, values, indices, vocab, chunks, 2048, num_warps=4)
    finish[(rows,)](values, indices, result, chunks, triton.next_power_of_2(chunks), num_warps=4)
    return result
