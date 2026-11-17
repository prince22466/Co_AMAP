import runpy,torch
from pathlib import Path
d=runpy.run_path(str(Path(__file__).with_name('sampler_fusion_probe.py')))
base,fused=d['base'],d['fused']
tests=[
 ('near_tie', [[0.,1e-8]], [[1.,1.]]),
 ('positive_inf', [[0.,float('inf'),-1.]], [[1.,1.,1.]]),
 ('all_masked', [[-float('inf'),-float('inf')]], [[1.,1.]]),
 ('exact_tie', [[1.,1.,0.]], [[1.,1.,1.]]),
 ('nan', [[0.,float('nan'),1.]], [[1.,1.,1.]]),
]
for name,xx,qq in tests:
 x=torch.tensor(xx,device='cuda'); q=torch.tensor(qq,device='cuda')
 a=base(x,q); b=fused(x,q)
 print(name,'baseline',a.tolist(),'candidate',b.tolist(),'match',bool(torch.equal(a,b)),flush=True)
