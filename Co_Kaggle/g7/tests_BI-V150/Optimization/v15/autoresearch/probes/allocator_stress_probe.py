import torch,hashlib,json,gc
torch.manual_seed(2026)
device='cuda'
torch.cuda.init()
base=torch.randn((2048,2048),device=device)
expect=base*1.25+0.5
stream=torch.cuda.Stream()
for i in range(80):
 with torch.cuda.stream(stream):
  scratch=[torch.full((100000+i*127,),float(i),device=device) for _ in range(8)]
  check=sum(t.sum() for t in scratch)
  target=sum(t.numel()*float(i) for t in scratch)
  assert abs(check.item()-target)<max(1,target*1e-6)
  del scratch
stream.synchronize()
x=base.clone()
out=torch.empty_like(x)
torch.cuda.synchronize()
graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
 torch.mul(x,1.25,out=out)
 out.add_(0.5)
for i in range(10):
 x.copy_(base)
 graph.replay()
 torch.cuda.synchronize()
 assert torch.equal(out,expect)
digest=hashlib.sha256(out.cpu().numpy().tobytes()).hexdigest()
del x,out,graph,base,expect
gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
print(json.dumps({'status':'PASS','graph_replays':10,'reuse_iterations':80,'digest':digest,'allocated_after':torch.cuda.memory_allocated(),'reserved_after':torch.cuda.memory_reserved()}),flush=True)
