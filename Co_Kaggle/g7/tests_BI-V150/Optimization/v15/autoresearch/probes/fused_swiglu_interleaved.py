import torch,triton,triton.language as tl,importlib,statistics,json
m=importlib.import_module('flag_gems.runtime.backend._iluvatar.ops.mm')
silu=importlib.import_module('flag_gems.fused.silu_and_mul')
@triton.jit
def fused(A,B,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
          BAM:tl.constexpr,BAK:tl.constexpr,BBK:tl.constexpr,BBN:tl.constexpr,
          BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
 pm=tl.program_id(0);pn=tl.program_id(1)
 rm=pm*BM+tl.arange(0,BM);rn=pn*BN+tl.arange(0,BN)
 rp=tl.arange(0,2*BN);col=pn*BN+rp//2;which=rp%2
 rk=tl.arange(0,BK)
 acc=tl.zeros((BM,2*BN),tl.float32)
 for k in range(tl.cdiv(K,BK)):
  r=k*BK+rk
  a=tl.load(A+rm[:,None]*BAM+r[None,:]*BAK,(rm[:,None]<M)&(r[None,:]<K),0)
  b=tl.load(B+r[:,None]*BBK+(col[None,:]+which[None,:]*N)*BBN,(r[:,None]<K)&(col[None,:]<N),0)
  acc=tl.dot(a,b,acc)
 pair=tl.reshape(acc.to(tl.bfloat16).to(tl.float32),(BM,BN,2))
 g,u=tl.split(pair)
 out=(g/(1.+tl.exp(-g)))*u
 tl.store(C+rm[:,None]*N+rn[None,:],out,(rm[:,None]<M)&(rn[None,:]<N))
@triton.jit
def pointwise(X,Y,N:tl.constexpr,TOTAL:tl.constexpr,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
 mask=i<TOTAL
 row=i//N;col=i%N
 g=tl.load(X+row*(2*N)+col,mask,0).to(tl.float32)
 u=tl.load(X+row*(2*N)+N+col,mask,0).to(tl.float32)
 tl.store(Y+i,g/(1.+tl.exp(-g))*u,mask)
torch.manual_seed(123)
for M,K,N in [(33,96,257),(2048,2048,6144)]:
 a=torch.randn((M,K),device='cuda',dtype=torch.bfloat16)
 b=torch.randn((N*2,K),device='cuda',dtype=torch.bfloat16).T
 tmp=torch.empty((M,2*N),device='cuda',dtype=torch.bfloat16)
 out=torch.empty((M,N),device='cuda',dtype=torch.bfloat16)
 baseline_out=torch.empty_like(out)
 def baseline():
  m._launch_mm(a,b,tmp,M,2*N,K)
  pointwise[(triton.cdiv(M*N,1024),)](tmp,baseline_out,N,M*N,1024,num_warps=4)
  return baseline_out
 def candidate():
  fused[(triton.cdiv(M,256),triton.cdiv(N,128))](a,b,out,M,N,K,a.stride(0),a.stride(1),b.stride(0),b.stride(1),256,128,32,num_warps=16,num_stages=2)
  return out
 ref=baseline();res=candidate();torch.cuda.synchronize()
 delta=(res.float()-ref.float()).abs()
 print(json.dumps({'shape':[M,K,N],'correct':bool(torch.allclose(res.float(),ref.float(),rtol=.02,atol=.05)),'max_abs':delta.max().item(),'mean_abs':delta.mean().item()}),flush=True)
 if not torch.allclose(res.float(),ref.float(),rtol=.02,atol=.05): raise SystemExit(1)
 times={'baseline':[],'fused':[]}
 for i in range(12):
  for name,fn in ([('baseline',baseline),('fused',candidate)] if i%2==0 else [('fused',candidate),('baseline',baseline)]):
   e0=torch.cuda.Event(enable_timing=True);e1=torch.cuda.Event(enable_timing=True)
   e0.record();z=fn();e1.record();e1.synchronize();times[name].append(e0.elapsed_time(e1))
 print(json.dumps({'shape':[M,K,N],'median_ms':{n:statistics.median(v) for n,v in times.items()},'samples':times}),flush=True)
