"""RL worker policy that replaces v25_rl.unit_actions completely."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from worker_reward import ANIMAL_PRODUCTS, ITEMS, OPS, PRODUCTS, RewardBreakdown, _positions

@dataclass(frozen=True)
class Task:
    target: tuple[int,int] | None
    op: str
    item: str=""
    amount: int=1
    critical: float=0.0
    slot: int=0
    @property
    def key(self): return (self.target,self.op,self.item,self.slot)

GLOBAL_FEATURE_NAMES=("day","hour","workers","shed_fill","wheat","fertilizer","products","seeds","animals","crops","weeds","unfed","critical_unfed","unwatered","critical_unwatered","yield_units","carried_products","crop_plan","animal_plan","free_workers")
BASE_FEATURE_NAMES=("day","hour","workers","free_workers","worker_x","worker_y","target_x","target_y","distance","at_target","worker_inv","worker_wheat","worker_fert","worker_products","worker_animals","shed_fill","shed_wheat","shed_fert","seed_count","age","yield_units","consecutive_unwatered","consecutive_unfed","watered","fed","cared","fert_days","care_bonus","fert_available","critical","amount","shed_distance")
CANDIDATE_FEATURE_NAMES=BASE_FEATURE_NAMES+tuple("op_"+x for x in OPS)+tuple("item_"+(x or "NONE") for x in ITEMS)

def totals(private):
    out=defaultdict(int)
    for k,v in private["shed"].items(): out[k]+=int(v)
    for inv in private["inventories"]:
        for k,v in inv.items(): out[k]+=int(v)
    return out

def global_features(e,obs,animal_plan,crop_plan,free_workers):
    farm=obs["farms"][obs["player"]]; p=obs["private"]; t=totals(p)
    animals=crops=weeds=unfed=cunfed=unwatered=cunwatered=0; held=0.0
    for row in farm["tiles"]:
        for x in row:
            if not isinstance(x,dict): continue
            if x.get("animal"):
                animals+=1; held+=float(x.get("yield_units",0) or 0)
                if not x.get("fed_today"):
                    unfed+=1; cunfed+=int(int(x.get("consecutive_unfed",0) or 0)>=1)
            elif x.get("kind")=="PLANT":
                crops+=1; held+=float(x.get("yield_units",0) or 0)
                if not x.get("watered_today"):
                    unwatered+=1; cunwatered+=int(int(x.get("consecutive_unwatered",0) or 0)>=1)
            elif x.get("kind")=="WEED": weeds+=1
    carried=sum(int(inv.get(q,0) or 0) for inv in p["inventories"] for q in PRODUCTS)
    v=[obs["day"]/29,obs["hour"]/23,len(p["inventories"])/16,sum(p["shed"].values())/100,t.get("WHEAT",0)/40,t.get("FERTILIZER",0)/40,sum(t.get(q,0) for q in PRODUCTS)/150,sum(p["seeds"].values())/50,animals/20,crops/50,weeds/30,unfed/20,cunfed/20,unwatered/50,cunwatered/50,held/100,carried/100,len(crop_plan)/100,len(animal_plan)/20,free_workers/16]
    return np.clip(np.asarray(v,dtype=np.float32),-5,5)

def candidate_features(e,obs,worker,task,free_workers):
    farm=obs["farms"][obs["player"]]; p=obs["private"]; pos=_positions(obs)[worker]; inv=p["inventories"][worker]
    target=task.target if task.target is not None else tuple(e.nearest_shed(pos))
    tile=e.tile(farm,task.target) if task.target is not None else {}
    tile=tile if isinstance(tile,dict) else {}
    age=obs["day"]-int(tile.get("planted_day",tile.get("placed_day",obs["day"])))
    base=[obs["day"]/29,obs["hour"]/23,len(p["inventories"])/16,free_workers/16,pos[0]/9,pos[1]/9,target[0]/9,target[1]/9,e.dist(pos,target)/18,float(pos==target),sum(inv.values())/30,int(inv.get("WHEAT",0))/10,int(inv.get("FERTILIZER",0))/10,sum(int(inv.get(q,0)) for q in PRODUCTS)/30,sum(int(inv.get(a,0)) for a in e.ANIMALS)/5,sum(p["shed"].values())/100,int(p["shed"].get("WHEAT",0))/40,int(p["shed"].get("FERTILIZER",0))/40,int(p["seeds"].get(task.item,0))/20,age/30,float(tile.get("yield_units",0) or 0)/12,float(tile.get("consecutive_unwatered",0) or 0)/2,float(tile.get("consecutive_unfed",0) or 0)/2,float(bool(tile.get("watered_today"))),float(bool(tile.get("fed_today"))),float(bool(tile.get("cared_today"))),(int(tile.get("fertilized_until_day",-1))-obs["day"])/10,float(tile.get("pending_care_bonus",0) or 0)/8,float(bool(tile.get("fertilizer_available"))),task.critical,task.amount/10,e.dist(target,e.nearest_shed(target))/18]
    oh=[0.0]*len(OPS); oh[OPS.index(task.op)]=1.0
    ih=[0.0]*len(ITEMS); ih[ITEMS.index(task.item) if task.item in ITEMS else 0]=1.0
    return np.clip(np.asarray(base+oh+ih,dtype=np.float32),-5,5)

class ActorCritic(nn.Module):
    def __init__(self,candidate_dim,state_dim,hidden=128):
        super().__init__()
        self.actor=nn.Sequential(nn.Linear(candidate_dim,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.critic=nn.Sequential(nn.Linear(state_dim,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
    def logits(self,x): return self.actor(x).squeeze(-1)
    def value(self,x): return self.critic(x).squeeze(-1)

@dataclass
class SubDecision:
    candidates: np.ndarray
    action_index: int
    old_log_prob: float

@dataclass
class TurnRecord:
    state: np.ndarray
    subdecisions: list[SubDecision]
    old_value: float
    turn: int
    reward: float=0.0
    reward_breakdown: dict=field(default_factory=dict)
    advantage: float=0.0
    return_target: float=0.0

class WorkerPolicy:
    def __init__(self,executor,model,device,deterministic=False,collect=True,rollout_temperature=0.2):
        if rollout_temperature<=0:
            raise ValueError("rollout_temperature must be > 0")
        self.e=executor; self.model=model; self.device=device; self.deterministic=deterministic; self.collect=collect
        self.rollout_temperature=float(rollout_temperature)
        self.records=[]; self.pending=None; self.candidate_counts=[]

    def tasks(self,obs,animal_plan,crop_plan):
        e=self.e; farm=obs["farms"][obs["player"]]; p=obs["private"]; day=obs["day"]; tasks=[]; need=defaultdict(int)
        for pt,planned in crop_plan.items():
            pt=tuple(pt); t=e.tile(farm,pt)
            if t is None:
                if int(p["seeds"].get(planned,0)): tasks.append(Task(pt,"PLANT",planned))
                continue
            if not isinstance(t,dict): continue
            if t.get("kind")=="WEED": tasks.append(Task(pt,"DIG")); continue
            if t.get("kind")!="PLANT": continue
            crop=t.get("crop",planned); age=day-int(t.get("planted_day",day)); spec=e.CROPS.get(crop)
            if float(t.get("yield_units",0) or 0)>0: tasks.append(Task(pt,"HARVEST",crop))
            if day<29 and not t.get("watered_today"): tasks.append(Task(pt,"WATER",crop,critical=float(int(t.get("consecutive_unwatered",0) or 0)>=1)))
            useful=max(a for a,_ in spec[2]) if spec and crop in ("TOMATO","STRAWBERRY") else (int(spec[3]) if spec else 0)
            if day<29 and age<=useful and int(t.get("fertilized_until_day",-1))<day: tasks.append(Task(pt,"FERTILIZE","FERTILIZER"))
            if age>useful and float(t.get("yield_units",0) or 0)<=0: tasks.append(Task(pt,"DIG"))
        for pt,a in animal_plan.items():
            pt=tuple(pt); t=e.tile(farm,pt); structure="COOP" if a=="GOOSE" else "PASTURE"
            if t is None: tasks.append(Task(pt,"BUILD_COOP" if a=="GOOSE" else "BUILD_PASTURE",a))
            elif isinstance(t,dict) and t.get("kind")=="WEED": tasks.append(Task(pt,"DIG"))
            elif isinstance(t,dict) and t.get("kind") in ("COOP","PASTURE") and not t.get("animal"):
                if t.get("kind")==structure: tasks.append(Task(pt,"PLACE_ANIMAL",a)); need[a]+=1
                else: tasks.append(Task(pt,"DIG"))
        unfed=0
        for y,row in enumerate(farm["tiles"]):
            for x,t in enumerate(row):
                if not isinstance(t,dict) or not t.get("animal"): continue
                pt=(x,y); a=t["animal"]
                if day<29 and not t.get("fed_today"): tasks.append(Task(pt,"FEED","WHEAT",critical=float(int(t.get("consecutive_unfed",0) or 0)>=1))); unfed+=1
                if day<29 and not t.get("cared_today"): tasks.append(Task(pt,"CARE",a))
                if float(t.get("yield_units",0) or 0)>0: tasks.append(Task(pt,"HARVEST",ANIMAL_PRODUCTS.get(a,"")))
                if t.get("fertilizer_available"): tasks.append(Task(pt,"COLLECT_FERTILIZER","FERTILIZER"))
        def pickups(item,short,cap):
            left=min(short,int(p["shed"].get(item,0))); slot=0
            while left>0:
                n=min(cap,left); tasks.append(Task(None,"PICKUP",item,n,slot=slot)); left-=n; slot+=1
        pickups("WHEAT",max(0,unfed-sum(int(i.get("WHEAT",0)) for i in p["inventories"])),4)
        pickups("FERTILIZER",max(0,sum(t.op=="FERTILIZE" for t in tasks)-sum(int(i.get("FERTILIZER",0)) for i in p["inventories"])),3)
        for a,n in need.items():
            shortage=max(0,n-sum(int(i.get(a,0)) for i in p["inventories"]))
            for slot in range(min(shortage,int(p["shed"].get(a,0)))): tasks.append(Task(None,"PICKUP",a,1,slot=slot))
        return tasks

    def extras(self,obs,w):
        pos=_positions(obs)[w]; inv=obs["private"]["inventories"][w]; shed=tuple(self.e.nearest_shed(pos)); out=[]
        for q in PRODUCTS:
            n=int(inv.get(q,0) or 0)
            if n: out.append(Task(shed,"DELIVER",q,n))
        out.append(Task(pos,"PASS"))
        return out

    def feasible(self,obs,w,t,seeds,shed):
        e=self.e; inv=obs["private"]["inventories"][w]; pos=_positions(obs)[w]; target=t.target if t.target is not None else tuple(e.nearest_shed(pos))
        if e.dist(pos,target)>max(0,23-int(obs["hour"])): return False
        if t.op=="PLANT" and int(seeds.get(t.item,0))<=0: return False
        if t.op=="FEED" and int(inv.get("WHEAT",0))<=0: return False
        if t.op=="FERTILIZE" and int(inv.get("FERTILIZER",0))<=0: return False
        if t.op=="PLACE_ANIMAL" and int(inv.get(t.item,0))<=0: return False
        if t.op=="PICKUP" and int(shed.get(t.item,0))<=0: return False
        if t.op=="DELIVER" and int(inv.get(t.item,0))<=0: return False
        return True

    def emit(self,obs,w,t):
        e=self.e; pos=_positions(obs)[w]; target=t.target if t.target is not None else tuple(e.nearest_shed(pos))
        if pos!=target: return e.move(pos,target)
        if t.op=="PLANT": return ["PLANT",t.item]
        if t.op=="PLACE_ANIMAL": return ["PLACE",t.item]
        if t.op=="PICKUP": return ["PICKUP",t.item,t.amount]
        if t.op=="DELIVER": return ["PLACE",t.item,t.amount]
        if t.op=="PASS": return ["PASS"]
        return [t.op]

    def unit_actions(self,obs,animal_plan,crop_plan):
        workers=list(range(len(_positions(obs)))); actions=[None]*len(workers); tasks=self.tasks(obs,animal_plan,crop_plan); reserved=set(); seeds=dict(obs["private"]["seeds"]); shed=dict(obs["private"]["shed"])
        state=global_features(self.e,obs,animal_plan,crop_plan,len(workers)); st=torch.as_tensor(state,dtype=torch.float32,device=self.device)
        with torch.no_grad(): old_value=float(self.model.value(st).item())
        subs=[]
        while workers:
            choices=[]; feats=[]
            for w in workers:
                for t in tasks+self.extras(obs,w):
                    if t.op!="PASS" and t.key in reserved: continue
                    if not self.feasible(obs,w,t,seeds,shed): continue
                    choices.append((w,t)); feats.append(candidate_features(self.e,obs,w,t,len(workers)))
            if not choices:
                for w in workers: actions[w]=["PASS"]
                break
            mat=np.stack(feats).astype(np.float32); ct=torch.as_tensor(mat,dtype=torch.float32,device=self.device)
            with torch.no_grad():
                raw_logits=self.model.logits(ct)
                behavior_logits=raw_logits if self.deterministic else raw_logits/self.rollout_temperature
                dist=Categorical(logits=behavior_logits)
                a=torch.argmax(raw_logits) if self.deterministic else dist.sample()
                lp=dist.log_prob(a)
            j=int(a.item()); w,t=choices[j]; actions[w]=self.emit(obs,w,t); self.candidate_counts.append(len(choices))
            if self.collect: subs.append(SubDecision(mat.astype(np.float16),j,float(lp.item())))
            workers.remove(w)
            if t.op!="PASS": reserved.add(t.key)
            if t.op=="PLANT": seeds[t.item]=max(0,int(seeds.get(t.item,0))-1)
            elif t.op=="PICKUP": shed[t.item]=max(0,int(shed.get(t.item,0))-t.amount)
        if self.collect: self.pending=TurnRecord(state,subs,old_value,int(obs["day"])*24+int(obs["hour"]))
        return [a or ["PASS"] for a in actions]

    def finish_turn(self,reward: RewardBreakdown):
        if not self.collect: return
        if self.pending is None: raise RuntimeError("missing pending turn")
        self.pending.reward=float(reward.reward); self.pending.reward_breakdown=reward.as_dict(); self.records.append(self.pending); self.pending=None
