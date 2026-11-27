import json, statistics
from pilot import OUT, load_runs
def combine(r,which="all"):
    bs=[v for k,v in r["bands"].items() if which=="all" or (which in ("early","later") and k.startswith(which+"_")) or k.endswith("_"+which)]
    keys=("loss","comfort","energy","carbon")
    n=sum(b["n"] for b in bs)
    if not n:return None
    return {**{k:sum(b.get(k,0.) for b in bs)/n for k in keys},"n":n,"changed":sum(b["changed"] for b in bs),
            "actions":{k:sum(b.get(k,0) for b in bs) for k in ("base","sat_down","sat_up","fan_down","fan_up")}}
def main():
    full={}
    for group in ["train","val"]:
        runs={}
        for r in load_runs(group):
            runs[r["scenario"],r["kind"]]=r
        methods=sorted({k[1] for k in runs if k[1]!="base"})
        result={}
        for m in methods:
            ss=sorted(s for (s,k) in runs if k==m and (s,"base") in runs)
            comparisons={}
            for part in ("all","early","later","occupied","pre","empty"):
                paired={s:{"base":combine(runs[s,"base"],part),"candidate":combine(runs[s,m],part)} for s in ss}
                paired={s:v for s,v in paired.items() if v["base"] and v["candidate"]}
                for p in paired.values():
                    p["delta"]={k:p["candidate"][k]-p["base"][k] for k in ("loss","comfort","energy","carbon")}
                if not paired:continue
                mean=statistics.mean(p["candidate"]["loss"] for p in paired.values())
                baseline=statistics.mean(p["base"]["loss"] for p in paired.values())
                d={k:statistics.mean(p["delta"][k] for p in paired.values()) for k in ("loss","comfort","energy","carbon")}
                comparisons[part]={"baseline":baseline,"loss":mean,"delta_pct":100*(mean/baseline-1) if baseline else None,
                   "delta_components":d,"compared":paired}
                if part=="all":print(group,m,len(paired),f"{baseline:.8f} -> {mean:.8f}",f"{comparisons[part]['delta_pct']:+.2f}%")
            result[m]=comparisons
        full[group]=result
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"summary.json").write_text(json.dumps(full,indent=2,allow_nan=False))
if __name__=="__main__": main()
