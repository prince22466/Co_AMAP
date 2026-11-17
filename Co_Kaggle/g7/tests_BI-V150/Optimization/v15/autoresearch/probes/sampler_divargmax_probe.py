import torch,triton,triton.language as tl,json,statistics
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
torch.manual_seed(42)
for rows,n in [(8,10003),(256,130560)]:
 p=torch.rand((rows,n),device='cuda');q=torch.empty_like(p).exponential_()
 for i in range(min(rows,6)):
  if i==0: p[i,1]=float('nan');p[i,5]=float('nan')
  if i==1: p[i,2]=float('inf');p[i,4]=float('inf')
  if i==2: p[i,3]=0;q[i,3]=0
  if i==3: p[i,:]=1;q[i,:]=1
  if i==4: p[i,:]=0;q[i,:]=float('inf')
 C=triton.cdiv(n,2048);v=torch.empty((rows,C),device='cuda');idx=torch.empty((rows,C),device='cuda',dtype=torch.int32)
 out=torch.empty(rows,device='cuda',dtype=torch.int64);tmp=torch.empty_like(p)
 def baseline():
  torch.div(p,q,out=tmp);return tmp.argmax(dim=-1)
 def fused():
  part[(rows,C)](p,q,v,idx,n,C,2048,num_warps=4)
  finish[(rows,)](v,idx,out,C,triton.next_power_of_2(C),num_warps=4)
  return out
 a=baseline();b=fused();torch.cuda.synchronize()
 print(json.dumps({'shape':[rows,n],'mismatch':(a!=b).sum().item(),'expected':a[:6].tolist(),'actual':b[:6].tolist()}),flush=True)
 if not torch.equal(a,b): raise SystemExit(1)
 timings={'baseline':[],'fused':[]}
 for j in range(16):
  for name,fn in ([('baseline',baseline),('fused',fused)] if j%2==0 else [('fused',fused),('baseline',baseline)]):
   s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True);s.record();fn();e.record();e.synchronize();timings[name].append(s.elapsed_time(e))
 print(json.dumps({'shape':[rows,n],'median_ms':{k:statistics.median(x) for k,x in timings.items()},'samples':timings}),flush=True)
