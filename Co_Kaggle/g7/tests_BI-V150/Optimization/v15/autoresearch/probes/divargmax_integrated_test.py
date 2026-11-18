import json,torch
from types import SimpleNamespace
import vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar
import vllm.v1.sample.ops.topk_topp_sampler as t
def ref(s,logits,gens,k,p):
 logits=t.apply_top_k_top_p(logits,k,p)
 extra=None
 if s.logprobs_mode=='processed_logits':extra=logits
 elif s.logprobs_mode=='processed_logprobs':extra=logits.log_softmax(dim=-1,dtype=torch.float32)
 return t.random_sample(logits.softmax(dim=-1,dtype=torch.float32),gens,s.use_fp64_gumbel),extra
def gens(mode,rows):
 indices=[] if mode=='none' else ([0,rows-1] if mode=='mixed' else list(range(rows)))
 return {i:torch.Generator(device='cuda').manual_seed(1000+i) for i in indices}
cases=[(1,17,'none','none',False,False,'raw_logprobs'),(8,10003,'none','none',False,False,'raw_logits'),(16,10003,'topk','mixed',False,False,'raw_logprobs'),(16,10003,'topp','all',False,False,'raw_logprobs'),(64,130560,'none','mixed',False,False,'raw_logprobs'),(8,4097,'none','none',True,False,'raw_logprobs'),(8,4097,'none','none',False,True,'raw_logprobs'),(8,4097,'none','none',False,False,'processed_logits'),(8,4097,'none','none',False,False,'processed_logprobs')]
for rows,n,filter_mode,gen_mode,fp64,noncontig,mode in cases:
 x=torch.randn((rows,n*2 if noncontig else n),device='cuda')
 if noncontig:x=x[:,::2]
 if rows>1:
  x[0,0]=float('nan');x[1,1]=float('inf')
 k=torch.full((rows,),min(20,n),device='cuda',dtype=torch.int32) if filter_mode=='topk' else None
 p=torch.full((rows,),.9,device='cuda') if filter_mode=='topp' else None
 s=SimpleNamespace(logprobs_mode=mode,use_fp64_gumbel=fp64)
 ga,gb=gens(gen_mode,rows),gens(gen_mode,rows)
 xa=x.clone(); xb=x.clone()
 if noncontig:
  xa=torch.empty((rows,n*2),device='cuda')[:,::2];xa.copy_(x)
  xb=torch.empty((rows,n*2),device='cuda')[:,::2];xb.copy_(x)
  assert not xa.is_contiguous() and not xb.is_contiguous()
 torch.manual_seed(831);a,la=ref(s,xa,ga,k,p);state_a=torch.cuda.get_rng_state()
 torch.manual_seed(831);b,lb=t.TopKTopPSampler.forward_native(s,xb,gb,k,p);state_b=torch.cuda.get_rng_state()
 torch.cuda.synchronize()
 assert torch.equal(a,b),(cases, a,b)
 assert torch.equal(state_a,state_b)
 assert all(torch.equal(ga[i].get_state(),gb[i].get_state()) for i in ga)
 if la is not None:assert torch.allclose(la,lb,equal_nan=True)
 print(json.dumps({'shape':[rows,n],'filter':filter_mode,'generators':gen_mode,'fp64':fp64,'noncontiguous':noncontig,'mode':mode,'tokens_and_rng_match':True}),flush=True)
