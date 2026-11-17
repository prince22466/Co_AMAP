import torch,triton,triton.language as tl,statistics,json
@triton.jit
def partial(X,Q,PM,PI,N:tl.constexpr,T:tl.constexpr,B:tl.constexpr):
 row=tl.program_id(0); part=tl.program_id(1)
 x=part*B+tl.arange(0,B); valid=x<N
 v=tl.load(X+row*N+x,mask=valid,other=-float('inf'))
 q=tl.load(Q+row*N+x,mask=valid,other=1.)
 s=v-tl.log(q); z=tl.max(s,0)
 ix=tl.min(tl.where(valid & (s==z),x,2147483647),0)
 tl.store(PM+row*T+part,z); tl.store(PI+row*T+part,ix)
@triton.jit
def finish(PM,PI,OUT,T:tl.constexpr,B:tl.constexpr):
 row=tl.program_id(0); i=tl.arange(0,B)
 s=tl.load(PM+row*T+i,i<T,other=-float('inf'))
 ix=tl.load(PI+row*T+i,i<T,other=2147483647); z=tl.max(s,0)
 out=tl.min(tl.where(s==z,ix,2147483647),0); tl.store(OUT+row,out)
def fused(x,q):
 r,n=x.shape; t=triton.cdiv(n,4096)
 pm=torch.empty((r,t),device=x.device,dtype=torch.float32)
 pi=torch.empty((r,t),device=x.device,dtype=torch.int32)
 out=torch.empty((r,),device=x.device,dtype=torch.int64)
 partial[(r,t)](x,q,pm,pi,n,t,4096,num_warps=4)
 finish[(r,)](pm,pi,out,t,triton.next_power_of_2(t),num_warps=4)
 return out
def base(x,q): return x.softmax(dim=-1,dtype=torch.float32).div_(q).argmax(dim=-1)
if __name__ == '__main__':
    torch.manual_seed(32)
    for rows,n in [(8,10003),(256,130560)]:
     x=torch.randn((rows,n),device='cuda',dtype=torch.float32)
     q=torch.empty_like(x).exponential_()
     a=base(x,q); b=fused(x,q); torch.cuda.synchronize()
     print(json.dumps({'shape':[rows,n],'mismatches':int((a!=b).sum())}),flush=True)
     for name,fn in [('baseline',base),('candidate',fused)]:
      fn(x,q); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
      start=torch.cuda.memory_allocated(); times=[]
      for j in range(8):
       e0=torch.cuda.Event(enable_timing=True); e1=torch.cuda.Event(enable_timing=True)
       e0.record(); out=fn(x,q); e1.record(); e1.synchronize(); times.append(e0.elapsed_time(e1))
      print(json.dumps({'path':name,'shape':[rows,n],'peak_increment':torch.cuda.max_memory_allocated()-start,'median_ms':statistics.median(times),'samples_ms':times}),flush=True)
