import runpy,torch
d=runpy.run_path('.autoresearch-work/sampler-fusion/probe.py')
base,fused=d['base'],d['fused']
torch.manual_seed(345)
r=65536;n=17
x=torch.full((r,n),-60.,device='cuda')
x[:,0]=torch.randn(r,device='cuda')*4
x[:,1]=x[:,0]+torch.randn(r,device='cuda')
q=torch.ones_like(x)
q[:,0]=torch.empty(r,device='cuda').exponential_()
q[:,1]=q[:,0]*torch.exp(x[:,1]-x[:,0])*(1+torch.randn(r,device='cuda')*0.0000002)
a=base(x,q);b=fused(x,q)
print('finite near tie mismatches',int((a!=b).sum()),'of',r,flush=True)
