"""One warmed unchanged top-p launch for native counter attribution."""
import hashlib
from pathlib import Path
import torch
from vllm.v1.sample.ops import topk_topp_triton as original

torch.manual_seed(8123)
logits = torch.randn((64, 130560), device='cuda') * 3
p = torch.full((64,), .95, device='cuda')
print('source_sha256', hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(), flush=True)
for i in range(3):
    x = logits.clone()
    original.apply_top_k_top_p_triton(x, None, p)
    torch.cuda.synchronize()
    print('launch_done', i, flush=True)

