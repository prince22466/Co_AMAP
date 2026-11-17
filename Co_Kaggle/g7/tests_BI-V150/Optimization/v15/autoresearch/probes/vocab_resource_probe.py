import torch,importlib,sys
m=importlib.import_module('flag_gems.runtime.backend._iluvatar.ops.linear')
M=int(sys.argv[1]);K=2048;N=130560
torch.manual_seed(12)
a=torch.randn((M,K),device='cuda',dtype=torch.bfloat16)
w=torch.randn((N,K),device='cuda',dtype=torch.bfloat16)
out=m.linear(a,w);torch.cuda.synchronize()
print('shape',M,K,N,'finite',bool(torch.isfinite(out).all().item()),'best',m.linear_kernel.fn.best_config,flush=True)
