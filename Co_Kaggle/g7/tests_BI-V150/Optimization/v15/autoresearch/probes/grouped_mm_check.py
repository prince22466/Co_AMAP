import importlib,torch,json,statistics
m=importlib.import_module('flag_gems.runtime.backend._iluvatar.ops.mm')
torch.manual_seed(42)
results=[]
for M,K,N in [(2048,2048,12288),(2048,2048,2560),(2048,2048,2048),(2048,6144,2048),(33,65,97)]:
 a=torch.randn((M,K),device='cuda',dtype=torch.bfloat16)
 b=torch.randn((N,K),device='cuda',dtype=torch.bfloat16).T
 c=torch.empty((M,N),device='cuda',dtype=torch.bfloat16)
 m._launch_mm(a,b,c,M,N,K); torch.cuda.synchronize()
 ref=(a.float() @ b.float()).to(torch.bfloat16)
 ok=torch.allclose(c.float(),ref.float(),rtol=0.02,atol=0.25)
 err=(c.float()-ref.float()).abs()
 timings=[]
 for iteration in range(15):
  start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
  start.record(); m._launch_mm(a,b,c,M,N,K); end.record(); end.synchronize()
  timings.append(start.elapsed_time(end))
 results.append({'M':M,'K':K,'N':N,'correct':ok,'max_abs':err.max().item(),'median_ms':statistics.median(timings),'samples_ms':timings})
 print(json.dumps(results[-1]),flush=True)
 if not ok: raise SystemExit(1)
print('success',flush=True)
