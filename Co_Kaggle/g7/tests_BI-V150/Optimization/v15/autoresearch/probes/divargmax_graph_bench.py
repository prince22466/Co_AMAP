import torch,triton,json,statistics
from types import SimpleNamespace
import vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar
import vllm.v1.sample.ops.topk_topp_sampler as t
from vllm_fl.ops.divargmax import sample
torch.manual_seed(89)
p=torch.rand((8,4097),device='cuda');q=torch.empty_like(p).exponential_()
sample(p,q);torch.cuda.synchronize()
g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g):o=sample(p,q)
for i in range(6):
 p.copy_(torch.rand_like(p));q.exponential_()
 if i==1:p[0,:]=1;q[0,:]=1
 if i==2:p[1,2]=float('nan')
 if i==3:p[2,3]=float('inf')
 if i==4:p[3,0]=0;q[3,0]=0
 ref=p.clone().div_(q).argmax(-1);g.replay();torch.cuda.synchronize()
 assert torch.equal(ref,o),(i,ref,o)
print(json.dumps({'graph_changed_inputs':6,'match':True}),flush=True)
for rows in (8,64,256):
 x=torch.randn((rows,130560),device='cuda')
 s=SimpleNamespace(logprobs_mode='raw_logprobs',use_fp64_gumbel=False)
 def baseline():return t.random_sample(x.softmax(-1,dtype=torch.float32),{},False)
 def candidate():return t.TopKTopPSampler.forward_native(s,x,{},None,None)[0]
 for _ in range(3):baseline();candidate()
 torch.cuda.synchronize()
 ts={'baseline':[],'candidate':[]}
 for i in range(14):
  for name,fn in ([('baseline',baseline),('candidate',candidate)] if i%2==0 else [('candidate',candidate),('baseline',baseline)]):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();fn();b.record();b.synchronize();ts[name].append(a.elapsed_time(b))
 print(json.dumps({'rows':rows,'vocab':130560,'median_ms':{k:statistics.median(v) for k,v in ts.items()},'samples':ts}),flush=True)
