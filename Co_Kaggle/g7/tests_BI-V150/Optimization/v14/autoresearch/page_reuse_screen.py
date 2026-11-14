import importlib.util, pathlib, torch, time, statistics, json, hashlib
root=pathlib.Path('/root/Optimization/v12')
source=root/'vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/iluvatar/impl/triton_unified_attention_biv150.py'
work=root/'.autoresearch-work/screen'
orig=source.read_text()
needle='max_seqlen_q == 1\n        and num_query_heads == 16'
assert orig.count(needle)==1
variant=orig.replace(needle,'(max_seqlen_q == 1 or optimize_prefill_gqa)\n        and num_query_heads == 16')
for name,contents in [('screen_baseline',orig),('screen_page_reuse',variant)]:
    (work/(name+'.py')).write_text(contents)
def load(name):
    spec=importlib.util.spec_from_file_location(name,work/(name+'.py'))
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
mods=[load('screen_baseline'),load('screen_page_reuse')]
torch.manual_seed(261111)
results=[]
for qlen,context in [(63,4097),(2048,2048),(2048,14336),(1,16377)]:
    length=qlen+context; page=16; pages=(length+page-1)//page+1
    q=torch.randn((qlen,16,128),device='cuda',dtype=torch.bfloat16)
    k=torch.randn((pages,page,2,128),device='cuda',dtype=torch.bfloat16)
    v=torch.randn_like(k)
    table=torch.randperm(pages,device='cuda',dtype=torch.int64).to(torch.int32)[None,:]
    common=dict(q=q,k=k,v=v,cu_seqlens_q=torch.tensor([0,qlen],device='cuda',dtype=torch.int32),
        max_seqlen_q=qlen,seqused_k=torch.tensor([length],device='cuda',dtype=torch.int32),
        max_seqlen_k=length,softmax_scale=128**-.5,causal=True,window_size=(-1,-1),
        block_table=table,softcap=0.,q_descale=None,k_descale=None,v_descale=None)
    outs=[torch.empty_like(q),torch.empty_like(q)]
    def call(i): mods[i].unified_attention(**common,out=outs[i])
    with torch.inference_mode():
        for i in [0,1,1,0]: call(i)
        torch.cuda.synchronize()
        try: torch.testing.assert_close(outs[1],outs[0],rtol=3e-2,atol=3e-2); correct=True
        except AssertionError as e: correct=False; print(str(e),flush=True)
        times=[[],[]]
        for rep in range(12):
            for i in ([0,1] if rep%2==0 else [1,0]):
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record(); call(i); b.record(); b.synchronize(); times[i].append(a.elapsed_time(b))
        r=dict(qlen=qlen,context=context,correct=correct,max_abs=(outs[1].float()-outs[0].float()).abs().max().item(),
               baseline_ms=times[0],candidate_ms=times[1],
               baseline_median_ms=statistics.median(times[0]),candidate_median_ms=statistics.median(times[1]))
        results.append(r); print(json.dumps(r),flush=True)
print('baseline_sha256',hashlib.sha256(orig.encode()).hexdigest(),'candidate_sha256',hashlib.sha256(variant.encode()).hexdigest(),flush=True)
(work/'page_reuse_results.json').write_text(json.dumps(results,indent=2))

