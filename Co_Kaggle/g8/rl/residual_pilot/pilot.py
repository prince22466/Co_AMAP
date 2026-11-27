"""Local V23 residual pilot. No deployment path or repository modifications."""
import argparse, json, os, random, sys, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from collections import Counter, defaultdict
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
os.environ.setdefault("MPLCONFIGDIR","/tmp/g8-mpl")
os.environ.setdefault("XDG_CACHE_HOME","/tmp/g8-xdg")
os.environ.setdefault("NUMBA_CACHE_DIR","/tmp/g8-numba-audit")
os.environ.setdefault("MPLBACKEND","Agg")
import numpy as np
REPO=Path(os.environ.get("G8_REPO",Path(__file__).resolve().parents[4])).resolve()
ROOT=REPO/"Co_Kaggle/g8/competition_material/Data"
sys.path[:0]=[str(REPO),str(ROOT)]
from Co_Kaggle.g8.rl.p0.controllers import load_controller, validate_action
from Co_Kaggle.g8.rl.p0.runner import simulator_working_directory
from Co_Kaggle.g8.rl.p0.metrics import weighted_losses
HERE=Path(__file__).resolve().parent
OUT=Path(os.environ.get("G8_PILOT_OUT", HERE/"runs")).resolve()
FROZEN=HERE/"results"
ACTIONS={"base":(0.,0.),"sat_down":(-.10,0.),"sat_up":(.10,0.),"fan_down":(0.,-.12),"fan_up":(0.,.12)}
TRAIN=["building_train_1/denver_7","building_train_2/phoenix_6","building_train_4/atlanta_5","building_train_7/singapore_6","building_train_4/fairbanks_4"]
VAL=["building_train_5/atlanta_6","building_train_5/denver_3","building_train_5/fairbanks_5","building_train_5/phoenix_1","building_train_5/singapore_4"]
BLOCK=36  # 3h training targets; policy chooses every 15 minutes from causal observations.
CADENCE=3
MIN_GAIN=.00007

def get_context_features(v,obs):
    fallback=v._schedule_fallback(obs)
    occ=v._value(obs,v.fields.get("num_occupants"))
    now=v._value(obs,v.fields.get("comfort_mode_now"))
    soon=v._value(obs,v.fields.get("comfort_mode_soon"))
    temps=[]; heat=[]; cool=[]; outs=[]
    for ahu in v.ahus:
        outs.append(v._outdoor_temperature(obs,ahu["outdoor"]))
        for z in ahu["zones"]:
            s=v._read_zone(obs,z,fallback)
            if s:
                temps.append(s["temperature_k"]); heat.append(s["heating_error_k"]); cool.append(s["cooling_error_k"])
    outdoor=[o for o in outs if np.isfinite(o)]
    avg=lambda x: float(np.mean(x)) if len(x) else 0.
    top=lambda x: float(np.quantile(x,.90)) if len(x) else 0.
    values=[now,soon,float(occ>0),np.log1p(max(0.,occ))/5.,
            (avg(outdoor)-295.)/20. if outdoor else 0.,
            (avg(temps)-295.)/10. if temps else 0.,
            avg(heat),avg(cool),top(heat),top(cool),
            v._value(obs,v.fields.get("hod_sin_000")),
            v._value(obs,v.fields.get("hod_cos_000"))]
    return np.clip(np.nan_to_num(values,nan=0.,posinf=4.,neginf=-4.),-4.,4.)

def rule(x):
    if (x[0]>.5 or x[1]>.5) and x[9] > .7 and x[9]>x[8]+.3:
        return "sat_down"
    return "base"

def learned(x, params):
    x=np.clip((x-np.array(params["mean"]))/np.array(params["scale"]),-4,4)
    xx=np.r_[1.,x]
    effects=np.asarray(params["coef"])@xx
    j=int(np.argmin(effects))
    return params["actions"][j] if effects[j]<-params["margin"] else "base"

def apply(v,base,act):
    sat,fan=ACTIONS[act]
    if not sat and not fan: return base
    cont=base["continuous_action"].copy()
    for unit in v.ahus:
        if not base["discrete_action"][unit["run"]]: continue
        for idx,offset in ((unit["supply"],sat),(unit["pressure"],fan)):
            if idx is not None and offset:
                cont[idx]=np.clip(float(cont[idx])+offset,-1.,1.)
    return {"continuous_action":cont,"discrete_action":base["discrete_action"]}

def roll(inp):
    scenario,kind,steps,seed,subdir,params=inp
    from smart_buildings.evaluate import create_building_meta_data,load_hybrid_environment
    np.random.seed(seed);random.seed(seed)
    cfg=ROOT/"smart_buildings/training_scenarios"/scenario/"config.gin"
    dest=OUT/subdir/(scenario.replace("/","__")+"__"+kind+".json")
    if dest.exists():
        old=json.loads(dest.read_text())
        if old["steps"]==steps and old["seed"]==seed and old.get("params")==params:
            return old
    with simulator_working_directory(ROOT):
        env=load_hybrid_environment(cfg)
        env.metrics_path=None; env._occupancy_normalization_constant=125.
        spec=env.action_spec()
        v=load_controller(23)(create_building_meta_data(env,cfg.parent),spec)
        obs,_=env.reset()
        bands=defaultdict(Counter); blocks=[]; act="base"
        for t in range(steps):
            f=get_context_features(v,obs)
            if t % BLOCK ==0: blocks.append({"x":f.tolist(),"sum":0.,"n":0,"changed":0})
            if t % CADENCE ==0:
                act=(rule(f) if kind=="rule" else learned(f,params) if kind=="learned" else kind)
            mode="occupied" if f[2] else "pre" if f[1] else "empty"
            # Classify from the pre-step observation, like the action.
            b=bands[("early" if t<864 else "later")+"_"+mode]
            base=v.get_action(obs); cmd=apply(v,base,act)
            changed=not np.array_equal(base["continuous_action"],cmd["continuous_action"])
            obs,r,term,trunc,_=env.step(validate_action(cmd,spec))
            _,res=env.get_reward_info_and_response()
            parts=weighted_losses(res.normalized_productivity_regret,res.normalized_energy_cost,
                                  res.normalized_carbon_emission,env.reward_function.weights)
            if not np.isclose(sum(parts.values()),-r,atol=1e-8,rtol=1e-7): raise ValueError("reward parity")
            b.update(parts); b["loss"]-=float(r); b["n"]+=1; b["changed"]+=int(changed); b[act]+=1
            blocks[-1]["sum"]-=float(r); blocks[-1]["n"]+=1; blocks[-1]["changed"]+=int(changed)
            if term or trunc: raise RuntimeError("Episode unexpectedly finished")
        env.close()
    row={"scenario":scenario,"kind":kind,"steps":steps,"seed":seed,"params":params,
         "bands":dict(bands),"loss":sum(b["loss"] for b in bands.values())/steps,"blocks":blocks}
    tmp=dest.with_suffix(".tmp");tmp.write_text(json.dumps(row,allow_nan=False));tmp.replace(dest)
    return row

def load_runs(phase):
    """Prefer a local run if it exists; do not mix it with archived results."""
    directory=OUT/phase
    if directory.exists():
        return [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))]
    return [json.loads(line) for line in (FROZEN/f"{phase}.jsonl").read_text().splitlines()]


def load_model():
    source=OUT/"model.json"
    if not source.exists():
        source=FROZEN/"model.json"
    return json.loads(source.read_text())


def fit(rows):
    from sklearn.linear_model import Ridge
    rows={(r["scenario"],r["kind"]):r for r in rows}
    acts=[a for a in ACTIONS if a!="base"]
    x=[];ys=[];groups=[]
    for s in TRAIN:
        for k,b in enumerate(rows[s,"base"]["blocks"]):
            x.append(b["x"]); groups.append(s.split("/")[0])
            ys.append([rows[s,a]["blocks"][k]["sum"]/b["n"]-b["sum"]/b["n"] for a in acts])
    x=np.asarray(x); ys=np.asarray(ys); groups=np.asarray(groups)
    def train(mask,alpha):
        mean=x[mask].mean(0);scale=np.maximum(x[mask].std(0),.15)
        m=Ridge(alpha=alpha).fit(np.clip((x[mask]-mean)/scale,-4,4),ys[mask])
        return {"actions":acts,"coef":np.c_[m.intercept_,m.coef_].tolist(),"mean":mean.tolist(),"scale":scale.tolist(),"alpha":alpha,"margin":MIN_GAIN}
    grid=[]
    for alpha in (10.,40.,160.):
        for margin in (MIN_GAIN,.0002,.0005):
            values=[]; count=0
            for s in sorted(set(groups)):
                mask=groups!=s; p=train(mask,alpha); p["margin"]=margin
                pairs=[(learned(z,p),y) for z,y in zip(x[~mask],ys[~mask])]
                values.append(float(np.mean([0. if a=="base" else y[acts.index(a)] for a,y in pairs])))
                count+=sum(a!="base" for a,_ in pairs)
            grid.append((float(np.mean(values)),alpha,margin,values,count))
    _,alpha,margin,_,_=min(grid)
    p=train(np.ones(len(x),dtype=bool),alpha);p["margin"]=margin
    p["train_cv"]=grid;p["blocks"]=len(x)
    p["static_train_scores"]={a:float(np.mean([rows[s,a]["loss"] for s in TRAIN])) for a in ACTIONS}
    p["best_static_action"]=min(p["static_train_scores"],key=p["static_train_scores"].get)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"model.json").write_text(json.dumps(p,indent=2))
    print("FIT",json.dumps(p),flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("stage",choices=["train","fit","val","val-controls","val-candidates"]);ap.add_argument("--workers",type=int,default=8)
    args=ap.parse_args()
    if args.stage=="fit":
        fit(load_runs("train"));return
    jobs_dir=OUT/("val" if args.stage.startswith("val") else args.stage);jobs_dir.mkdir(exist_ok=True)
    if args.stage=="train":
        jobs=[(s,k,1152,42+i,"train",None) for i,s in enumerate(TRAIN) for k in ACTIONS]
    elif args.stage=="val-controls":
        jobs=[(s,k,1728,42+i,"val",None) for i,s in enumerate(VAL) for k in ("base","rule")]
    else:
        model=load_model()
        choices=tuple(dict.fromkeys(("base","rule","learned",model["best_static_action"])))
        if args.stage=="val-candidates": choices=tuple(k for k in choices if k not in ("base","rule"))
        jobs=[(s,k,1728,42+i,"val",model if k=="learned" else None) for k in choices for i,s in enumerate(VAL)]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context("spawn")) as pool:
        futs=[pool.submit(roll,j) for j in jobs]
        for f in as_completed(futs):
            r=f.result()
            print(r["scenario"],r["kind"],round(r["loss"],9),sum(x["changed"] for x in r["bands"].values()),flush=True)
if __name__=="__main__":
    main()
