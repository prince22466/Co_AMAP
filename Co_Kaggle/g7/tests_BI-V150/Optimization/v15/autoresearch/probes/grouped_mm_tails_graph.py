import importlib,torch
m=importlib.import_module('flag_gems.runtime.backend._iluvatar.ops.mm')
for dtype in (torch.float16,torch.bfloat16):
 torch.manual_seed(7)
 M,K,N=1025,128,8193
 a=torch.randn((M,K),device='cuda',dtype=dtype)
 b=torch.randn((N,K),device='cuda',dtype=dtype).T
 c=torch.empty((M,N),device='cuda',dtype=dtype)
 m._launch_mm(a,b,c,M,N,K)
 torch.cuda.synchronize()
 ref=(a.float() @ b.float()).to(dtype)
 assert torch.allclose(c.float(),ref.float(),atol=.25,rtol=.02)
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph): m._launch_mm(a,b,c,M,N,K)
 for i in range(3):
  graph.replay()
  torch.cuda.synchronize()
  assert torch.allclose(c.float(),ref.float(),atol=.25,rtol=.02)
 print(dtype,'tail and 3 graph replays passed',flush=True)
