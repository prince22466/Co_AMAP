import torch,sys,json
torch.cuda.init()
def snap(label):
 torch.cuda.synchronize()
 free,total=torch.cuda.mem_get_info()
 reserved=torch.cuda.memory_reserved()
 print(json.dumps({'phase':label,'free':free,'reserved':reserved,'allocated':torch.cuda.memory_allocated(),'non_torch':total-free-reserved}),flush=True)
snap('before')
count=int(sys.argv[1]);size=5_000_000_000//count
data=[torch.empty(size,device='cuda',dtype=torch.uint8) for _ in range(count)]
snap('allocated')
del data
torch.cuda.empty_cache()
snap('released')
