"""Assert split, score, phase and baseline parity for the generated runs."""
import json, math
from pilot import REPO, TRAIN, VAL, ACTIONS, load_model, load_runs
def near(a,b,atol=1e-8):
    assert math.isclose(a,b,rel_tol=1e-7,abs_tol=atol),(a,b)

def verify():
    model=load_model()
    expected={"train":(TRAIN,tuple(ACTIONS),1152),"val":(VAL,tuple(dict.fromkeys(("base","rule","learned",model["best_static_action"]))),1728)}
    prev=json.loads((REPO/"Co_Kaggle/g8/rl/p3/results/run1/history.json").read_text())["history"][0]["validation"]["baseline"]
    for phase,(ss,choices,n) in expected.items():
        results={}
        for r in load_runs(phase):
            s=r["scenario"]; a=r["kind"]
            assert (s,a) not in results and r["steps"]==n
            assert sum(b["n"] for b in r["bands"].values())==n
            assert sum(b["n"] for b in r["blocks"])==n
            near(r["loss"],sum(b["loss"] for b in r["bands"].values())/n)
            near(r["loss"],sum(b["sum"] for b in r["blocks"])/n)
            for b in r["bands"].values():
                assert 0<=b["changed"]<=b["n"] and b["n"]==sum(b.get(a,0) for a in ACTIONS)
                near(sum(b[key] for key in ("comfort","energy","carbon")),b["loss"],2e-6)
            if a=="base":
                assert sum(b["changed"] for b in r["bands"].values())==0
                if phase=="val":
                    loss=sum(b["loss"] for k,b in r["bands"].items() if k.startswith("early_"))/864
                    target=next(q["mean_loss"] for q in prev if q["scenario"]==s)
                    near(loss,target,1e-10)
            results[s,a]=r
        assert set(results)==set((s,a) for s in ss for a in choices)
        for s in ss:
            b=results[s,"base"]
            for a in choices:
                q=results[s,a]
                assert q["seed"]==b["seed"]
                assert {k:v["n"] for k,v in q["bands"].items()}=={k:v["n"] for k,v in b["bands"].items()}
        print(phase,len(results),"runs verified")
if __name__=="__main__":verify()
