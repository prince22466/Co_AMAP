import importlib.util,torch,triton,json
from pathlib import Path
spec=importlib.util.spec_from_file_location('probe',Path(__file__).with_name('sampler_divargmax_probe.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.manual_seed(318)
for rows,n in [(65536,16),(2048,2047),(2048,2048),(2048,2049),(256,130560)]:
 q=torch.empty((rows,n),device='cuda').exponential_()
 if n==16:
  p=q*torch.rand((rows,1),device='cuda')
 else:
  p=torch.rand((rows,n),device='cuda')
 p[0,:]=0;q[0,:]=0
 p[1,:]=1;q[1,:]=1
 p[2,0]=float('nan');p[2,-1]=float('nan')
 p[3,0]=float('inf');p[3,-1]=float('inf')
 C=triton.cdiv(n,2048);v=torch.empty((rows,C),device='cuda');i=torch.empty((rows,C),dtype=torch.int32,device='cuda');o=torch.empty(rows,dtype=torch.int64,device='cuda')
 def calc():
  m.part[(rows,C)](p,q,v,i,n,C,2048,num_warps=4)
  m.finish[(rows,)](v,i,o,C,triton.next_power_of_2(C),num_warps=4)
 ref=p.clone().div_(q).argmax(-1)
 calc();torch.cuda.synchronize()
 mism=(ref!=o).sum().item()
 print(json.dumps({'shape':[rows,n],'mismatch':mism}),flush=True)
 if mism: raise SystemExit(2)
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):calc()
 for _ in range(3):
  g.replay();torch.cuda.synchronize()
  if not torch.equal(ref,o):raise SystemExit(3)
 print(json.dumps({'shape':[rows,n],'graph_replays':3,'passed':True}),flush=True)
