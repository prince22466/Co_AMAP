import torch
from vllm_fl.dispatch.backends.vendor.iluvatar.impl import triton_unified_attention_biv150 as m
torch.manual_seed(22)
qlen,context,page,head=65,33,16,128
total=qlen+context; pages=(total+15)//16+1
order=torch.randperm(pages,device='cuda').int()
q=torch.zeros((qlen,16,head),device='cuda',dtype=torch.bfloat16)
k=torch.zeros((pages,page,2,head),device='cuda',dtype=torch.bfloat16)
v=torch.zeros_like(k)
vals=torch.arange(pages*page,device='cuda',dtype=torch.float32).reshape(pages,page,1,1).expand_as(v).to(v.dtype)
v[order.long()]=torch.zeros_like(vals)
v[order[2].item(),1,:,:]=1
out=torch.empty_like(q)
args=dict(q=q,k=k,v=v,out=out,cu_seqlens_q=torch.tensor([0,qlen],device='cuda',dtype=torch.int32),max_seqlen_q=qlen,seqused_k=torch.tensor([total],device='cuda',dtype=torch.int32),max_seqlen_k=total,softmax_scale=head**-.5,causal=True,window_size=(-1,-1),block_table=order[None,:],softcap=0.,q_descale=None,k_descale=None,v_descale=None)
with torch.inference_mode():m.unified_attention(**args);torch.cuda.synchronize()
print('single_value_q0',out[0,0,0].item(),'q1',out[1,0,0].item(),'expected_inclusive_q0',1/34, 'expected_inclusive_q1',1/35,flush=True)

