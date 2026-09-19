#!/usr/bin/env python3
"""v20: PPO over v19-feasible worker/task choices."""
from __future__ import annotations
import argparse, json, math, random, re, tempfile, time, types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

HERE=Path(__file__).resolve().parent
LOCAL_ARENA=HERE.parent
G5_ROOT=LOCAL_ARENA.parent
SUBMISSION_DIR=G5_ROOT/"submission_nb"
DEFAULT_EXECUTOR=SUBMISSION_DIR/"kaggriculture-sub_v19.ipynb"
DEFAULT_OPPONENTS=[SUBMISSION_DIR/"kaggriculture-sub_v16.ipynb",SUBMISSION_DIR/"kaggriculture-sub_v17.ipynb",DEFAULT_EXECUTOR]

TASK_FEATURE_NAMES=("day","hour","distance","task_weight","op_index","kind_index","age","yield_units","consecutive_unwatered","fed_today","cared_today","fertilized_days_left","worker_wheat","worker_fertilizer","worker_inventory","task_shed_distance","task_x","task_y","worker_x","worker_y","workers")
TASK_SCALES=np.asarray([29,23,20,500,8,7,30,20,5,1,1,30,20,20,50,10,9,9,9,9,16],dtype=np.float32)
GLOBAL_FEATURE_NAMES=("day","hour","money","opp_money","money_margin","hands","opp_hands","lands","opp_lands","wheat","fertilizer","price_wheat","price_milk","price_wool","price_egg","candidate_count","free_workers")

OLD_BLOCK="""    score_cache={}
    while free and tasks:
        best=None
        for i in free:
            for k,(p,op,weight,resource) in enumerate(tasks):
                if resource=='NO_WHEAT':
                    if invs[i].get('WHEAT',0):continue
                elif resource=='NO_FERTILIZER':
                    if invs[i].get('FERTILIZER',0):continue
                elif resource and not invs[i].get(resource,0):continue
                distance=dist(positions[i],p)
                if distance>=24-hour:continue
                key=(i,p,tuple(op))
                if key not in score_cache:
                    score_cache[key]=learned_task_score(task_features(obs,i,(p,op,weight,resource)))
                score=score_cache[key]
                choice=(score,-distance,-i,-k)
                if best is None or choice>best:best=choice
        if best is None:break
        _,_,ni,nk=best;i=-ni;k=-nk;p,op,_,_=tasks[k]
        actions[i]=op if tuple(positions[i])==p else move(positions[i],p)
        free.remove(i);tasks=[t for t in tasks if t[0]!=p]
"""
NEW_BLOCK="""    while free and tasks:
        selected=RL_SELECTOR.choose(obs,free,tasks,positions,invs)
        if selected is None:break
        i,k=selected;p,op,_,_=tasks[k]
        actions[i]=op if tuple(positions[i])==p else move(positions[i],p)
        free.remove(i);tasks=[t for t in tasks if t[0]!=p]
"""

def _field(obj:Any,key:str,default=None):
    return obj.get(key,default) if isinstance(obj,dict) else getattr(obj,key,default)

def _extract_notebook_main(path:Path)->str:
    nb=json.loads(path.read_text(encoding="utf-8")); found=[]
    for cell in nb.get("cells",[]):
        if cell.get("cell_type")!="code":continue
        lines="".join(cell.get("source",[])).splitlines()
        if not lines:continue
        m=re.match(r"^\\s*%%writefile\\s+(.+?)\\s*$",lines[0])
        if m and Path(m.group(1).strip("'\\\"")).name=="main.py":found.append("\\n".join(lines[1:])+"\\n")
    if len(found)!=1:raise ValueError(f"{path}: expected one main.py cell, found {len(found)}")
    return found[0]

def _load_executor(path:Path,selector):
    path=path.expanduser().resolve()
    source=_extract_notebook_main(path) if path.suffix==".ipynb" else path.read_text(encoding="utf-8")
    if OLD_BLOCK not in source:raise RuntimeError("v19 unit_actions selection block not found; source changed")
    source=source.replace(OLD_BLOCK,NEW_BLOCK,1)
    module=types.ModuleType(f"v20_executor_{random.randrange(1<<60)}")
    module.__file__=str(path);module.RL_SELECTOR=selector
    exec(compile(source,str(path),"exec"),module.__dict__)
    return module

def _extract_plain(path:Path)->str:
    return _extract_notebook_main(path) if path.suffix==".ipynb" else path.read_text(encoding="utf-8")

def prepare_opponents(raw:list[str],directory:Path):
    out=[]
    for idx,value in enumerate(raw):
        if value in {"pass","random","starter"}:out.append((value,value));continue
        p=Path(value).expanduser().resolve()
        if not p.exists():raise FileNotFoundError(p)
        if p.suffix==".py":out.append((p.stem,str(p)));continue
        q=directory/f"opponent_{idx}_{p.stem}.py";q.write_text(_extract_plain(p),encoding="utf-8");out.append((p.stem,str(q)))
    return out

def norm_task(x):
    return np.clip(np.asarray(x,dtype=np.float32)/TASK_SCALES,-5,5)

def global_state(m,obs,candidates,free_workers):
    p=int(obs["player"]);o=1-p;own=obs["farms"][p];opp=obs["farms"][o];private=obs["private"];total=m.totals(private);prices=obs["market"]["prices"]
    x=np.asarray([
        obs["day"]/29,obs["hour"]/23,own["money"]/100000,opp["money"]/100000,(own["money"]-opp["money"])/100000,
        len(own["hands"])/12,len(opp["hands"])/12,len(own["unlocked_quadrants"])/4,len(opp["unlocked_quadrants"])/4,
        total.get("WHEAT",0)/300,total.get("FERTILIZER",0)/200,
        prices["WHEAT"]/25,prices["MILK"]/160,prices["WOOL"]/200,prices["EGG"]/50,
        candidates/128,free_workers/16],dtype=np.float32)
    return np.clip(x,-5,5)

@dataclass
class Step:
    state:np.ndarray;candidates:np.ndarray;baseline:np.ndarray;action:int;old_log_prob:float;old_value:float;turn:int

@dataclass
class EpisodeResult:
    ok:bool;seed:int;opponent:str;seat:int;our_money:float|None;opponent_money:float|None;margin:float|None;terminal_reward:float|None;status_ours:str;status_opponent:str;decisions:int;error:str=""

class ActorCritic(nn.Module):
    def __init__(self,task_dim,state_dim,hidden=64):
        super().__init__()
        self.actor=nn.Sequential(nn.Linear(task_dim,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.critic=nn.Sequential(nn.Linear(state_dim,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
        for x in self.modules():
            if isinstance(x,nn.Linear):nn.init.orthogonal_(x.weight,gain=math.sqrt(2));nn.init.zeros_(x.bias)
        nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias);nn.init.orthogonal_(self.critic[-1].weight,gain=1.0)
    def logits(self,candidates,baseline,scale):
        return scale*(baseline-baseline.max())+self.actor(candidates).squeeze(-1)
    def value(self,state):return self.critic(state).squeeze(-1)

class Selector:
    def __init__(self,model,device,scale=1.0,deterministic=False,forced_baseline=False,collect=True):
        self.model=model;self.device=device;self.scale=scale;self.deterministic=deterministic;self.forced_baseline=forced_baseline;self.collect=collect;self.steps=[];self.module=None
    def choose(self,obs,free,tasks,positions,invs):
        choices=[]
        for i in free:
            for k,(p,op,weight,resource) in enumerate(tasks):
                if resource=="NO_WHEAT":
                    if invs[i].get("WHEAT",0):continue
                elif resource=="NO_FERTILIZER":
                    if invs[i].get("FERTILIZER",0):continue
                elif resource and not invs[i].get(resource,0):continue
                d=self.module.dist(positions[i],p)
                if d>=24-obs["hour"]:continue
                f=self.module.task_features(obs,i,(p,op,weight,resource));b=float(self.module.learned_task_score(f))
                choices.append((i,k,d,f,b))
        if not choices:return None
        if self.forced_baseline:
            j=max(range(len(choices)),key=lambda z:(choices[z][4],-choices[z][2],-choices[z][0],-choices[z][1]))
            return choices[j][0],choices[j][1]
        cand=np.stack([norm_task(c[3]) for c in choices]).astype(np.float32);base=np.asarray([c[4] for c in choices],dtype=np.float32)
        ct=torch.as_tensor(cand,dtype=torch.float32,device=self.device);bt=torch.as_tensor(base,dtype=torch.float32,device=self.device)
        state=global_state(self.module,obs,len(choices),len(free));st=torch.as_tensor(state,dtype=torch.float32,device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits=self.model.logits(ct,bt,self.scale);dist=Categorical(logits=logits);a=logits.argmax() if self.deterministic else dist.sample();lp=dist.log_prob(a);v=self.model.value(st)[0]
        j=int(a.item())
        if self.collect and len(choices)>1:self.steps.append(Step(state,cand,base,j,float(lp.item()),float(v.item()),int(obs["day"])*24+int(obs["hour"])))
        return choices[j][0],choices[j][1]

class Controller:
    def __init__(self,path,model,device,scale=1.0,deterministic=False,forced_baseline=False,collect=True):
        self.selector=Selector(model,device,scale,deterministic,forced_baseline,collect);self.executor=_load_executor(path,self.selector);self.selector.module=self.executor
    @property
    def steps(self):return self.selector.steps
    def __call__(self,obs):return self.executor.agent(obs)

def terminal_reward(margin):
    return (1.0 if margin>0 else -1.0 if margin<0 else 0.0)+0.05*math.tanh(margin/10000)

def run_episode(model,device,executor_path,opponent,seed,seat,episode_steps,baseline_scale,deterministic=False,forced_baseline=False,collect_steps=True):
    from kaggle_environments import make
    label,runner=opponent;ctrl=Controller(executor_path,model,device,baseline_scale,deterministic,forced_baseline,collect_steps)
    def candidate(obs):return ctrl(obs)
    players=[None,None];players[seat]=candidate;players[1-seat]=runner
    try:
        env=make("kaggriculture",configuration={"episodeSteps":episode_steps,"seed":seed},debug=False);env.run(players);final=env.steps[-1];ours,theirs=final[seat],final[1-seat]
        so,st=str(_field(ours,"status","")),str(_field(theirs,"status",""));ro,rt=_field(ours,"reward"),_field(theirs,"reward")
        if ro is None or rt is None:raise RuntimeError("missing terminal reward")
        ro,rt=float(ro),float(rt);margin=ro-rt;ok=so=="DONE" and st=="DONE"
        return EpisodeResult(ok,seed,label,seat,ro,rt,margin,terminal_reward(margin) if ok else None,so,st,len(ctrl.steps),"" if ok else "non-DONE status"),ctrl.steps
    except Exception as e:
        return EpisodeResult(False,seed,label,seat,None,None,None,None,"ERROR","ERROR",len(ctrl.steps),f"{type(e).__name__}: {e}"),ctrl.steps

def discounted_returns(steps,reward,gamma,episode_steps):
    return np.asarray([reward*gamma**max(0,episode_steps-1-s.turn) for s in steps],dtype=np.float32)

def ppo_update(model,opt,device,steps,returns,scale,epochs,minibatch,clip_ratio,value_coef,entropy_coef,max_grad_norm):
    oldv=np.asarray([s.old_value for s in steps],dtype=np.float32);adv=returns-oldv
    if len(adv)>1:adv=(adv-adv.mean())/(adv.std()+1e-8)
    stats=[]
    for _ in range(epochs):
        order=np.random.permutation(len(steps))
        for start in range(0,len(order),minibatch):
            ids=order[start:start+minibatch];nlog=[];ent=[];vals=[]
            for q in ids:
                s=steps[int(q)];ct=torch.as_tensor(s.candidates,dtype=torch.float32,device=device);bt=torch.as_tensor(s.baseline,dtype=torch.float32,device=device);dist=Categorical(logits=model.logits(ct,bt,scale));nlog.append(dist.log_prob(torch.tensor(s.action,device=device)));ent.append(dist.entropy());vals.append(model.value(torch.as_tensor(s.state,dtype=torch.float32,device=device).unsqueeze(0))[0])
            nlog=torch.stack(nlog);ent=torch.stack(ent).mean();vals=torch.stack(vals);oldlp=torch.as_tensor([steps[int(q)].old_log_prob for q in ids],dtype=torch.float32,device=device);a=torch.as_tensor(adv[ids],dtype=torch.float32,device=device);y=torch.as_tensor(returns[ids],dtype=torch.float32,device=device);ratio=torch.exp(nlog-oldlp);pl=-torch.minimum(ratio*a,torch.clamp(ratio,1-clip_ratio,1+clip_ratio)*a).mean();vl=0.5*((vals-y)**2).mean();loss=pl+value_coef*vl-entropy_coef*ent;opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm);opt.step();stats.append((pl.item(),vl.item(),ent.item(),loss.item()))
    a=np.asarray(stats);return {"policy_loss":float(a[:,0].mean()),"value_loss":float(a[:,1].mean()),"entropy":float(a[:,2].mean()),"loss":float(a[:,3].mean()),"return_mean":float(returns.mean())}

def write_jsonl(path,row):
    with path.open("a",encoding="utf-8") as f:f.write(json.dumps(row,sort_keys=True,default=str)+"\n")

def choose_device(v):
    if v!="auto":return torch.device(v)
    if torch.cuda.is_available():return torch.device("cuda")
    return torch.device("mps") if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available() else torch.device("cpu")

def save_checkpoint(path,model,opt,update,args):
    torch.save({"algorithm":"v20_constrained_task_ppo","update":update,"model_state_dict":model.state_dict(),"optimizer_state_dict":opt.state_dict(),"task_feature_names":TASK_FEATURE_NAMES,"global_feature_names":GLOBAL_FEATURE_NAMES,"baseline_scale":args.baseline_scale,"args":vars(args)},path)

def load_checkpoint(path,model,opt,device):
    p=torch.load(path,map_location=device,weights_only=False)
    if p.get("algorithm")!="v20_constrained_task_ppo":raise ValueError("checkpoint algorithm mismatch")
    model.load_state_dict(p["model_state_dict"])
    if opt is not None and p.get("optimizer_state_dict"):opt.load_state_dict(p["optimizer_state_dict"])
    return int(p.get("update",-1))+1,p

def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--executor",type=Path,default=DEFAULT_EXECUTOR);p.add_argument("--opponents",nargs="+",default=[str(x) for x in DEFAULT_OPPONENTS]);p.add_argument("--output-dir",type=Path,default=HERE/"runs"/"task_ppo");p.add_argument("--updates",type=int,default=100);p.add_argument("--episodes-per-update",type=int,default=8);p.add_argument("--episode-steps",type=int,default=720);p.add_argument("--seed",type=int,default=32020);p.add_argument("--device",default="auto");p.add_argument("--hidden",type=int,default=64);p.add_argument("--learning-rate",type=float,default=3e-4);p.add_argument("--gamma",type=float,default=0.999);p.add_argument("--ppo-epochs",type=int,default=4);p.add_argument("--minibatch-size",type=int,default=64);p.add_argument("--clip-ratio",type=float,default=0.2);p.add_argument("--value-coef",type=float,default=0.5);p.add_argument("--entropy-coef",type=float,default=0.01);p.add_argument("--max-grad-norm",type=float,default=0.5);p.add_argument("--baseline-scale",type=float,default=1.0);p.add_argument("--resume",type=Path);p.add_argument("--smoke-only",action="store_true");return p

def main():
    args=parser().parse_args();device=choose_device(args.device);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);model=ActorCritic(len(TASK_FEATURE_NAMES),len(GLOBAL_FEATURE_NAMES),args.hidden).to(device);opt=torch.optim.Adam(model.parameters(),lr=args.learning_rate);start=0
    if args.resume:start,_=load_checkpoint(args.resume.expanduser().resolve(),model,opt,device)
    out=args.output_dir.expanduser().resolve();ck=out/"checkpoints";ck.mkdir(parents=True,exist_ok=True);(out/"config.json").write_text(json.dumps({**vars(args),"algorithm":"v20_constrained_task_ppo","task_feature_names":TASK_FEATURE_NAMES,"global_feature_names":GLOBAL_FEATURE_NAMES},indent=2,default=str)+"\n",encoding="utf-8")
    rng=random.Random(args.seed)
    with tempfile.TemporaryDirectory(prefix="v20_rl_") as tmp:
        opponents=prepare_opponents(args.opponents,Path(tmp))
        if args.smoke_only:
            r,_=run_episode(model,device,args.executor,opponents[0],args.seed,0,args.episode_steps,args.baseline_scale,True,True,False);print(json.dumps(asdict(r),indent=2));raise SystemExit(0 if r.ok else 1)
        for u in range(start,args.updates):
            results=[];steps=[];rets=[];attempts=0;t0=time.perf_counter()
            while len(results)<args.episodes_per_update:
                attempts+=1
                if attempts>args.episodes_per_update*3:raise RuntimeError("too many failed episodes")
                opp=rng.choice(opponents);seat=rng.randrange(2);seed=rng.randrange(1,2_147_483_647);r,s=run_episode(model,device,args.executor,opp,seed,seat,args.episode_steps,args.baseline_scale);write_jsonl(out/"episodes.jsonl",{"update":u,**asdict(r)})
                if not r.ok or r.terminal_reward is None or not s:continue
                steps.extend(s);rets.append(discounted_returns(s,r.terminal_reward,args.gamma,args.episode_steps));results.append(r)
            returns=np.concatenate(rets);stat=ppo_update(model,opt,device,steps,returns,args.baseline_scale,args.ppo_epochs,args.minibatch_size,args.clip_ratio,args.value_coef,args.entropy_coef,args.max_grad_norm);m=np.asarray([r.margin for r in results]);metric={"update":u,"episodes":len(results),"decisions":len(steps),"decisions_per_episode":len(steps)/len(results),"wins":int((m>0).sum()),"ties":int((m==0).sum()),"losses":int((m<0).sum()),"win_rate":float((m>0).mean()),"mean_margin":float(m.mean()),"elapsed_seconds":round(time.perf_counter()-t0,3),**stat};write_jsonl(out/"metrics.jsonl",metric);print(json.dumps(metric,sort_keys=True));save_checkpoint(ck/f"update_{u:04d}.pt",model,opt,u,args);save_checkpoint(ck/"latest.pt",model,opt,u,args)

if __name__=="__main__":main()
