import torch,json
from types import SimpleNamespace
import vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar
import vllm.v1.sample.ops.topk_topp_sampler as t
def reference(logits,generators,k,p):
 logits=t.apply_top_k_top_p(logits,k,p)
 return t.random_sample(logits.softmax(dim=-1,dtype=torch.float32),generators,False)
torch.manual_seed(42)
for rows,n,mode,per_gen in [(1,17,'none',False),(8,10003,'none',False),(16,10003,'topk',False),(16,10003,'topp',False),(64,130560,'none',True)]:
 x=torch.randn(rows,n,device='cuda',dtype=torch.float32)
 k=torch.full((rows,),min(20,n),device='cuda',dtype=torch.int32) if mode=='topk' else None
 p=torch.full((rows,),.9,device='cuda') if mode=='topp' else None
 s=SimpleNamespace(logprobs_mode='raw_logprobs',use_fp64_gumbel=False)
 def gens(): return {0:torch.Generator(device='cuda').manual_seed(95)} if per_gen else {}
 torch.manual_seed(831); a=reference(x.clone(),gens(),k,p)
 torch.manual_seed(831); b,other=t.TopKTopPSampler.forward_native(s,x.clone(),gens(),k,p)
 assert other is None and torch.equal(a,b)
 print(json.dumps({'rows':rows,'n':n,'mode':mode,'per_generator':per_gen,'exact_tokens':True}),flush=True)
x=torch.tensor([[0.,float('inf'),1.],[0.,float('nan'),1.],[-float('inf')]*3],device='cuda')
s=SimpleNamespace(logprobs_mode='raw_logprobs',use_fp64_gumbel=False)
torch.manual_seed(99);a=reference(x.clone(),{},None,None)
torch.manual_seed(99);b,_=t.TopKTopPSampler.forward_native(s,x.clone(),{},None,None)
assert torch.equal(a,b)
x=torch.randn(8,10003,device='cuda'); x2=x.clone()
s=SimpleNamespace(logprobs_mode='processed_logits',use_fp64_gumbel=False)
torch.manual_seed(10);o,logits=t.TopKTopPSampler.forward_native(s,x2,{},None,None)
assert torch.equal(x,x2) and logits.data_ptr()==x2.data_ptr()
print('nonfinite and processed_logits fallback passed',flush=True)
s=SimpleNamespace(logprobs_mode='raw_logprobs',use_fp64_gumbel=False)
x=torch.randn(8,4097,device='cuda')
fixed=x.clone()
torch.manual_seed(102)
t.TopKTopPSampler.forward_native(s,x,{},None,None)
x.copy_(fixed)
torch.cuda.synchronize()
g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
 result,_=t.TopKTopPSampler.forward_native(s,x,{},None,None)
for i in range(3):
 x.copy_(fixed)
 g.replay()
 torch.cuda.synchronize()
 assert result.shape==(8,) and bool(((result>=0)&(result<4097)).all())
print('three sampler graph replays passed',flush=True)
