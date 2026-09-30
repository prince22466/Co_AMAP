"""RL worker policy that replaces v25_rl.unit_actions completely."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from worker_reward import ANIMAL_PRODUCTS, ITEMS, MOVE_ACTIONS, OPS, PRODUCTS, RewardBreakdown, _positions, plant_can_deliver_before_end, _crop_decay_start_step as crop_decay_start_step

@dataclass(frozen=True)
class Task:
    target: tuple[int,int] | None
    op: str
    item: str=""
    amount: int=1
    critical: float=0.0
    slot: int=0
    planned: bool=False
    deadline_step: int | None=None
    release_step: int=0
    @property
    def key(self): return (self.target,self.op,self.item,self.slot)

GLOBAL_FEATURE_NAMES=("day","hour","workers","shed_fill","wheat","fertilizer","products","seeds","animals","crops","weeds","unfed","critical_unfed","unwatered","critical_unwatered","yield_units","carried_products","crop_plan","animal_plan","free_workers")
BASE_FEATURE_NAMES=("day","hour","workers","free_workers","worker_x","worker_y","target_x","target_y","distance","at_target","worker_inv","worker_wheat","worker_fert","worker_products","worker_animals","shed_fill","shed_wheat","shed_fert","seed_count","age","yield_units","consecutive_unwatered","consecutive_unfed","watered","fed","cared","fert_days","care_bonus","fert_available","critical","amount","shed_distance")
CANDIDATE_FEATURE_NAMES=BASE_FEATURE_NAMES+tuple("op_"+x for x in OPS)+tuple("item_"+(x or "NONE") for x in ITEMS)

# PASS/defer and non-plant planner shaping below are measured directly in
# normalized-advantage units.  Planned PLANT is intentionally different: its
# completion credit is expressed in raw reward-equivalent units so it can be
# calibrated against an ordinary crop HARVEST before PPO normalization.
PLANNED_PLANT_REWARD_EQUIV = 16.0
PLANNED_BUILD_ACTOR_BONUS = 1.0
PLANNED_ANIMAL_PICKUP_ACTOR_BONUS = 1.0
PLANNED_PLACE_ANIMAL_ACTOR_BONUS = 1.5
DEFER_PLANNED_PLANT_ACTOR_PENALTY = -0.75
AVOIDABLE_PASS_ACTOR_PENALTY = -1.0

def _weed_prevention_task(task):
    return task.op=="WATER" and float(task.critical)>=1.0

def _animal_survival_task(task):
    return (
        (task.op=="FEED" and float(task.critical)>=1.0)
        or (
            task.op in ("PICKUP","HARVEST")
            and task.item=="WHEAT"
            and float(task.critical)>=1.0
        )
    )

def _animal_setup_task(task):
    return task.planned and (
        task.op in ("BUILD_COOP","BUILD_PASTURE","PLACE_ANIMAL")
        or (task.op in ("PICKUP","DIG") and task.item in ANIMAL_PRODUCTS)
    )

def _crop_decay_start_step(e,tile):
    return crop_decay_start_step(e,tile)

def _crop_decay_deadline_step(e,obs,tile):
    """Harvest/retire before the first decay, independent of held quantity.

    The engine applies worker actions before decay at the same absolute step,
    so HARVEST at decay_start itself is still lossless.
    """
    start=_crop_decay_start_step(e,tile)
    if start is None:
        return None
    # Ongoing crops remain on the tile after HARVEST. Leave an action for DIG
    # before the engine turns an exhausted zero-yield plant into a weed.
    if tile.get("crop") in ("TOMATO","STRAWBERRY"):
        # Hands disappear and the farmer returns to the shed at midnight.
        # Finish both HARVEST and DIG using today's workforce, before reset.
        return start-(2 if float(tile.get("yield_units",0) or 0)>0 else 1)
    return start


def _crop_salvage_deadline_step(e,obs,tile):
    """Last action step that can still collect positive yield after spoilage."""
    start=_crop_decay_start_step(e,tile)
    units=int(np.ceil(float(tile.get("yield_units",0) or 0)))
    if start is None or units<=0:
        return None
    now=int(obs["day"])*24+int(obs["hour"])
    next_decay=start if now<=start else now+(now-start)%2
    return next_decay+2*(units-1)

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
    actor_bonus: float = 0.0
    reward_equiv_bonus: float = 0.0

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
        # A policy decision selects a job, not a one-turn direction. Workers keep
        # moving toward that job until it completes or becomes invalid.
        self.active_tasks: dict[int, Task] = {}
        # Originating PPO sample for a remote planned route.  Positive planner
        # shaping is credited to this original choice only when the route
        # reaches its execution point; interrupted/invalid routes earn nothing.
        self.active_origins: dict[int, SubDecision] = {}
        self.active_day: int | None = None
        # Track WHEAT that came from the shed so returning the same resource to
        # the shed cannot masquerade as newly completed production logistics.
        self.shed_sourced_wheat: dict[int, int] = defaultdict(int)
        self.turn_delivery_credit: list[dict[str, int]] = []
        # Reward-only provenance for planner execution, useful route movement,
        # and PASS choices that skipped feasible work.
        self.turn_plan_credit: list[dict[str, str]] = []
        self.turn_planned_plant_origins: list[SubDecision] = []
        self.turn_route_progress: list[bool] = []
        self.turn_avoidable_pass: list[bool] = []

    def tasks(self,obs,animal_plan,crop_plan):
        e=self.e; farm=obs["farms"][obs["player"]]; p=obs["private"]; day=obs["day"]; tasks=[]; need=defaultdict(int)
        for pt,planned in crop_plan.items():
            pt=tuple(pt); t=e.tile(farm,pt)
            if t is None:
                if int(p["seeds"].get(planned,0)): tasks.append(Task(pt,"PLANT",planned,planned=True))
                continue
            if not isinstance(t,dict): continue
            if t.get("kind")=="WEED": tasks.append(Task(pt,"DIG")); continue
        # Planner changes must not orphan assets already on the farm.
        live_plants=[((x,y),t) for y,row in enumerate(farm["tiles"]) for x,t in enumerate(row)]
        for pt,t in live_plants:
            if not isinstance(t,dict) or t.get("kind")!="PLANT": continue
            planned=t.get("crop","")
            crop=t.get("crop",planned); age=day-int(t.get("planted_day",day)); spec=e.CROPS.get(crop)
            first_yield_age=int(getattr(e,"CROP_FIRST_YIELD_DAY",{}).get(crop,10**9))
            # HARVEST legality comes from the engine's first_yield_day, not
            # from the planner's yield schedule. WHEAT/CARROT are legal at age
            # 2 even though the planner schedules their nominal yield later.
            if float(t.get("yield_units",0) or 0)>0 and age>=first_yield_age:
                tasks.append(Task(
                    pt,"HARVEST",crop,
                    deadline_step=_crop_decay_deadline_step(e,obs,t),
                ))
            if day<29 and not t.get("watered_today"): tasks.append(Task(pt,"WATER",crop,critical=float(int(t.get("consecutive_unwatered",0) or 0)>=1)))
            useful=max(a for a,_ in spec[2]) if spec and crop in ("TOMATO","STRAWBERRY") else (int(spec[3]) if spec else 0)
            if day<29 and age<=useful and int(t.get("fertilized_until_day",-1))<day: tasks.append(Task(pt,"FERTILIZE","FERTILIZER"))
            exhausted=(age>=useful if crop in ("TOMATO","STRAWBERRY") else age>useful)
            if exhausted and float(t.get("yield_units",0) or 0)<=0:
                tasks.append(Task(pt,"DIG",deadline_step=_crop_decay_deadline_step(e,obs,t)))
        for pt,a in animal_plan.items():
            pt=tuple(pt); t=e.tile(farm,pt); structure="COOP" if a=="GOOSE" else "PASTURE"
            if t is None: tasks.append(Task(pt,"BUILD_COOP" if a=="GOOSE" else "BUILD_PASTURE",a,planned=True))
            elif isinstance(t,dict) and t.get("kind")=="WEED": tasks.append(Task(pt,"DIG",a,planned=True))
            elif isinstance(t,dict) and t.get("kind") in ("COOP","PASTURE") and not t.get("animal"):
                if t.get("kind")==structure: tasks.append(Task(pt,"PLACE_ANIMAL",a,planned=True)); need[a]+=1
                else: tasks.append(Task(pt,"DIG",a,planned=True))
        unfed=0; critical_unfed=0; critical_feed_targets=[]
        for y,row in enumerate(farm["tiles"]):
            for x,t in enumerate(row):
                if not isinstance(t,dict) or not t.get("animal"): continue
                pt=(x,y); a=t["animal"]
                if day<29 and not t.get("fed_today"):
                    is_critical=int(t.get("consecutive_unfed",0) or 0)>=1
                    tasks.append(Task(
                        pt,"FEED","WHEAT",critical=float(is_critical)
                    ))
                    unfed+=1
                    critical_unfed+=int(is_critical)
                    if is_critical:
                        critical_feed_targets.append(pt)
                if day<29 and not t.get("cared_today"): tasks.append(Task(pt,"CARE",a))
                if float(t.get("yield_units",0) or 0)>0: tasks.append(Task(pt,"HARVEST",ANIMAL_PRODUCTS.get(a,"")))
                if t.get("fertilizer_available"): tasks.append(Task(pt,"COLLECT_FERTILIZER","FERTILIZER"))

        def pickups(item,short,cap,critical_units=0):
            left=min(short,int(p["shed"].get(item,0)))
            critical_left=min(max(0,int(critical_units)),left)
            slot=0
            while left>0:
                n=min(cap,left)
                tasks.append(Task(
                    None,"PICKUP",item,n,
                    critical=float(critical_left>0),
                    slot=slot,
                ))
                left-=n
                critical_left=max(0,critical_left-n)
                slot+=1

        carried_wheat=sum(
            int(inv.get("WHEAT",0) or 0) for inv in p["inventories"]
        )
        remaining=max(0,23-int(obs["hour"]))
        positions=_positions(obs)
        reachable_critical_wheat=0
        for w,pos in enumerate(positions):
            units=int(p["inventories"][w].get("WHEAT",0) or 0)
            if units<=0:
                continue
            if any(
                e.dist(pos,target)<=remaining
                for target in critical_feed_targets
            ):
                reachable_critical_wheat+=units
        ordinary_wheat_short=max(0,unfed-carried_wheat)
        critical_wheat_short=max(
            0,critical_unfed-reachable_critical_wheat
        )
        # Frozen market orders may leave the shed empty. Mature field WHEAT
        # is also a feed prerequisite; reserve its HARVEST before ordinary work.
        if critical_wheat_short>int(p["shed"].get("WHEAT",0)):
            tasks=[replace(t,critical=1.0) if t.op=="HARVEST" and t.item=="WHEAT" else t for t in tasks]
        # Stranded carried WHEAT must not suppress an emergency shed pickup.
        # Generate at least enough pickup capacity to cover the critical
        # reachable-supply deficit, even when total carried WHEAT would make
        # the ordinary aggregate shortage appear to be zero.
        wheat_short=max(ordinary_wheat_short,critical_wheat_short)
        pickups(
            "WHEAT",wheat_short,4,
            critical_units=critical_wheat_short,
        )
        pickups("FERTILIZER",max(0,sum(t.op=="FERTILIZE" for t in tasks)-sum(int(i.get("FERTILIZER",0)) for i in p["inventories"])),3)
        for a,n in need.items():
            shortage=max(0,n-sum(int(i.get(a,0)) for i in p["inventories"]))
            for slot in range(min(shortage,int(p["shed"].get(a,0)))):
                tasks.append(Task(None,"PICKUP",a,1,slot=slot,planned=True))
        return tasks

    def _sync_day(self,obs):
        day=int(obs["day"])
        if self.active_day != day:
            self.active_day=day
            self.active_tasks.clear()
            self.active_origins.clear()
            self.shed_sourced_wheat.clear()

    def _delivery_credit_for_worker(self,obs,w):
        inv=obs["private"]["inventories"][w]
        out={}
        for q in PRODUCTS:
            n=max(0,int(inv.get(q,0) or 0))
            if q=="WHEAT":
                n=max(0,n-int(self.shed_sourced_wheat.get(w,0)))
            out[q]=n
        return out

    def extras(self,obs,w):
        pos=_positions(obs)[w]; inv=obs["private"]["inventories"][w]; shed=tuple(self.e.nearest_shed(pos)); out=[]
        credit=self._delivery_credit_for_worker(obs,w)
        for q in PRODUCTS:
            n=int(credit.get(q,0) or 0)
            if n: out.append(Task(shed,"DELIVER",q,n))
        out.append(Task(pos,"PASS"))
        return out

    def _track_resource_action(self,w,action):
        if not action:
            return
        op=action[0]
        if op=="PICKUP" and len(action)>=2 and action[1]=="WHEAT":
            n=int(action[2]) if len(action)>=3 else 1
            self.shed_sourced_wheat[w]+=max(0,n)
        elif op=="FEED" and self.shed_sourced_wheat.get(w,0)>0:
            self.shed_sourced_wheat[w]-=1

    def feasible(self,obs,w,t,seeds,shed):
        e=self.e; inv=obs["private"]["inventories"][w]; pos=_positions(obs)[w]; target=t.target if t.target is not None else tuple(e.nearest_shed(pos))
        distance=e.dist(pos,target)
        remaining=max(0,23-int(obs["hour"]))
        # A newly planted seed starts with consecutive_unwatered=1 and becomes
        # WEED if it cannot be watered before the day refresh. Reserve one full
        # later turn for WATER after movement + PLANT.
        if t.op=="PLANT":
            if distance+1>remaining: return False
        elif (
            t.op in ("PICKUP","HARVEST")
            and t.item=="WHEAT"
            and float(t.critical)>=1.0
        ):
            # Emergency WHEAT supply is useful only if this worker can reach
            # the source, PICKUP/HARVEST, then reach at least one critical
            # animal and FEED it before the end-of-day escape refresh.
            farm=obs["farms"][obs["player"]]
            critical_animals=[]
            for y,row in enumerate(farm["tiles"]):
                for x,tile in enumerate(row):
                    if (
                        isinstance(tile,dict)
                        and tile.get("animal")
                        and not tile.get("fed_today")
                        and int(tile.get("consecutive_unfed",0) or 0)>=1
                    ):
                        critical_animals.append((x,y))
            if not critical_animals:
                return False
            followup=min(e.dist(target,pt) for pt in critical_animals)
            if distance+1+followup>remaining: return False
        elif distance>remaining:
            return False
        if t.op=="HARVEST" and t.deadline_step is not None:
            # Missing the first-loss deadline makes salvage urgent, not illegal.
            tile=e.tile(obs["farms"][obs["player"]],target)
            salvage=_crop_salvage_deadline_step(e,obs,tile)
            if salvage is not None and int(obs["day"])*24+int(obs["hour"])+distance>salvage:
                return False
        if t.op=="PLANT" and int(seeds.get(t.item,0))<=0: return False
        if t.op=="PLANT" and not self._plant_has_capacity(obs,w,t,seeds,shed): return False
        if t.op=="FEED" and int(inv.get("WHEAT",0))<=0: return False
        if t.op=="FERTILIZE" and int(inv.get("FERTILIZER",0))<=0: return False
        if t.op=="PLACE_ANIMAL" and int(inv.get(t.item,0))<=0: return False
        if t.op=="PLACE_ANIMAL" and not self._animal_has_capacity(obs,w,t,shed): return False
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

    def planned_completion_bonus(self,t):
        """Normalized actor-only completion bonus for non-plant plan steps."""
        if not t.planned:
            return 0.0
        if t.op in ("BUILD_COOP","BUILD_PASTURE"):
            return PLANNED_BUILD_ACTOR_BONUS
        if t.op=="PICKUP" and t.item in self.e.ANIMALS:
            return PLANNED_ANIMAL_PICKUP_ACTOR_BONUS
        if t.op=="PLACE_ANIMAL":
            return PLANNED_PLACE_ANIMAL_ACTOR_BONUS
        return 0.0

    def planned_completion_reward_equiv(self,t):
        """Raw reward-equivalent credit, normalized later with rollout GAE."""
        if t.planned and t.op=="PLANT":
            return PLANNED_PLANT_REWARD_EQUIV
        return 0.0

    def avoidable_plant_delay(self,w,t,choices):
        # Survival, maintenance, harvest/delivery and the forced animal pipeline
        # are legitimate reasons to defer planting. Alternatives here already
        # passed seed, travel and shared maintenance admission. Only useful
        # planting alternatives receive a deferral preference.
        if t.op in ("PLANT","WATER","FEED","CARE","HARVEST","DELIVER"):
            return False
        if _animal_setup_task(t) or (t.op=="DIG" and t.deadline_step is not None):
            return False
        if _animal_survival_task(t):
            return False
        obs=getattr(self,"_choice_obs",None)
        return any(cw==w and ct.planned and ct.op=="PLANT" and (
            obs is None or plant_can_deliver_before_end(
                self.e,obs,ct.item,ct.target,self.e.dist(_positions(obs)[w],ct.target)
            )
        ) for cw,ct,_ in choices)

    def actor_bonus_for_choice(self,w,t,choices,completed=False):
        same_worker=[ct for cw,ct,_ in choices if cw==w]
        bonus=0.0
        if t.op=="PASS" and any(ct.op!="PASS" for ct in same_worker):
            bonus+=AVOIDABLE_PASS_ACTOR_PENALTY
        if self.avoidable_plant_delay(w,t,choices):
            bonus+=DEFER_PLANNED_PLANT_ACTOR_PENALTY
        if completed:
            bonus+=self.planned_completion_bonus(t)
        return float(bonus)

    def _capacity_states(self,obs):
        now=int(obs["day"])*24+int(obs["hour"])
        return {
            w:(pos,now,int(obs["private"]["inventories"][w].get("WHEAT",0)))
            for w,pos in enumerate(_positions(obs))
        }

    def _advance_capacity_state(self,obs,t,state,shed_wheat,allow_pickup=False):
        """Project travel + action, including the FEED wheat prerequisite."""
        pos,ready,wheat=state
        target=t.target if t.target is not None else tuple(self.e.nearest_shed(pos))
        used=0
        if t.op=="FEED" and wheat<=0:
            if not allow_pickup or shed_wheat<=0:
                return None
            shed_pos=tuple(self.e.nearest_shed(pos))
            ready+=self.e.dist(pos,shed_pos)+1
            pos=shed_pos
            wheat+=1
            used=1
        action_step=max(ready+self.e.dist(pos,target),int(t.release_step))
        if t.op=="FEED":
            wheat-=1
        elif t.op=="PICKUP" and t.item=="WHEAT":
            if t.amount>shed_wheat:
                return None
            used=t.amount
            wheat+=t.amount
        elif t.op=="HARVEST" and t.item=="WHEAT":
            tile=self.e.tile(obs["farms"][obs["player"]],target)
            if isinstance(tile,dict):
                wheat+=int(tile.get("yield_units",0) or 0)
        ready=action_step+1
        if t.op=="HARVEST" and t.item in ("TOMATO","STRAWBERRY"):
            tile=self.e.tile(obs["farms"][obs["player"]],target)
            start=_crop_decay_start_step(self.e,tile) if isinstance(tile,dict) else None
            if start is not None and action_step>=start-24:
                ready+=1  # final harvest must be followed by exhausted cleanup
        return (target,ready,wheat),used,action_step

    def _deadline_schedule(self,obs,tasks,states=None,shed_wheat=None,delay=0):
        """Bounded EDF route estimate; a worker may finish several jobs.

        This is a conservative capacity guard, not an optimal-routing proof.
        It accounts for movement, each action, shared wheat, and release times.
        Unlike one-task-per-worker matching it exposes queued harvest pressure.
        """
        states=dict(self._capacity_states(obs) if states is None else states)
        states={w:(p,ready+delay,wheat) for w,(p,ready,wheat) in states.items()}
        stock=int(obs["private"]["shed"].get("WHEAT",0)) if shed_wheat is None else shed_wheat
        end=int(obs["day"])*24+23
        routes={w:[] for w in states}
        completed=set()
        slack={w:float("inf") for w in states}
        # Stable order, with survival first on equal deadlines. Planning each
        # action from its predecessor reserves cumulative worker time.
        def compatible_workers(t):
            deadline=end if t.deadline_step is None else min(end,t.deadline_step)
            return sum(
                (p:=self._advance_capacity_state(obs,t,state,stock,allow_pickup=True)) is not None
                and p[2]<=deadline for state in states.values()
            )
        def first_completion(t):
            finishes=[p[2] for state in states.values() if
                (p:=self._advance_capacity_state(obs,t,state,stock,allow_pickup=True)) is not None]
            return min(finishes,default=float("inf"))
        ordered=sorted(tasks,key=lambda t:(
            end if t.deadline_step is None else min(end,t.deadline_step),
            compatible_workers(t),
            0 if _weed_prevention_task(t) or _animal_survival_task(t) else 1,
            first_completion(t),
            repr(t.key),
        ))
        for t in ordered:
            deadline=end if t.deadline_step is None else min(end,int(t.deadline_step))
            candidates=[]
            for w,state in states.items():
                projected=self._advance_capacity_state(obs,t,state,stock,allow_pickup=True)
                if projected is not None and projected[2]<=deadline:
                    candidates.append((projected[2],w,projected))
            if not candidates:
                continue
            _,w,(new_state,used,action_step)=min(candidates,key=lambda c:(c[0],c[1]))
            stock-=used
            states[w]=new_state
            routes[w].append(t)
            slack[w]=min(slack[w],deadline-action_step)
            completed.add(t.key)
        return completed,routes,slack

    def _maintenance_tasks(self,obs,tasks):
        end=int(obs["day"])*24+23
        return [t for t in tasks if (
            t.op in ("WATER","FEED")
            or (t.op in ("HARVEST","DIG") and t.deadline_step is not None and t.deadline_step<=end+1)
        )]

    def _plant_has_capacity(self,obs,w,t,seeds,shed):
        now=int(obs["day"])*24+int(obs["hour"])
        in_turn=getattr(self,"_capacity_step",None)==now
        states=dict(self.turn_capacity_states if in_turn else self._capacity_states(obs))
        tasks=self.turn_capacity_tasks if in_turn else self.tasks(obs,{}, {})
        reserved=self.turn_capacity_reserved if in_turn else set()
        proposed=list(self.turn_capacity_plants if in_turn else [])
        projected=self._advance_capacity_state(obs,t,states[w],int(shed.get("WHEAT",0)))
        if projected is None:
            return False
        states[w]=projected[0]
        water=Task(t.target,"WATER",t.item,critical=1.0,release_step=projected[0][1])
        proposed.append(water)
        jobs=[job for job in self._maintenance_tasks(obs,tasks) if job.key not in reserved]+proposed
        stock=int(shed.get("WHEAT",0))
        if t.item=="WHEAT":
            # Do not deadlock the feed-production pipeline by forbidding its
            # renewal when today's wheat is depleted. Reserve realistic pickup
            # time; actual FEED still requires observed wheat in feasible().
            stock=max(stock,sum(job.op=="FEED" for job in jobs))
        covered,_,_=self._deadline_schedule(obs,jobs,states,stock)
        if len(covered)<len({job.key for job in jobs}):
            return False
        animals=list(self.turn_capacity_animals if in_turn else [])
        return self._next_day_maintenance_fits(obs,states,proposed,animals)

    def _animal_has_capacity(self,obs,w,t,shed):
        now=int(obs["day"])*24+int(obs["hour"])
        in_turn=getattr(self,"_capacity_step",None)==now
        states=dict(self.turn_capacity_states if in_turn else self._capacity_states(obs))
        tasks=self.turn_capacity_tasks if in_turn else self.tasks(obs,{}, {})
        reserved=self.turn_capacity_reserved if in_turn else set()
        plants=list(self.turn_capacity_plants if in_turn else [])
        animals=list(self.turn_capacity_animals if in_turn else [])
        projected=self._advance_capacity_state(obs,t,states[w],int(shed.get("WHEAT",0)))
        if projected is None:
            return False
        states[w]=projected[0]
        jobs=[job for job in self._maintenance_tasks(obs,tasks) if job.key not in reserved]+plants
        covered,_,_=self._deadline_schedule(obs,jobs,states,int(shed.get("WHEAT",0)))
        if len(covered)<len({job.key for job in jobs}):
            return False
        # New animals start consecutive_unfed=0: unlike PLANT, PLACE does not
        # require feeding on its placement day to survive. Reserve tomorrow's
        # FEED/CARE workload, including other placements assigned this turn.
        animals.append(t)
        return self._next_day_maintenance_fits(obs,states,plants,animals)

    def _next_day_maintenance_fits(self,obs,states,plants,animals):
        # Reserve a full day's WATER, FEED and CARE for the enlarged farm.
        # Use the current workforce; future market hires are not guaranteed.
        if int(obs["day"])>=29:
            return True
        tomorrow={**obs,"day":int(obs["day"])+1,"hour":0}
        daily=[]
        for y,row in enumerate(obs["farms"][obs["player"]]["tiles"]):
            for x,tile in enumerate(row):
                if not isinstance(tile,dict):
                    continue
                if tile.get("kind")=="PLANT":
                    daily.append(Task((x,y),"WATER",tile.get("crop","")))
                elif tile.get("animal"):
                    daily.extend((Task((x,y),"FEED","WHEAT"),Task((x,y),"CARE",tile["animal"])))
        daily.extend(Task(job.target,"WATER",job.item) for job in plants)
        for job in animals:
            daily.extend((Task(job.target,"FEED","WHEAT"),Task(job.target,"CARE",job.item)))
        # Reserve pickup time per future FEED. Future supply is unknown and may
        # include today's planting/harvest; using today's stock as a hard future
        # limit would forbid WHEAT planting precisely when feed needs renewal.
        future_states={q:(p,int(tomorrow["day"])*24,0) for q,(p,_,_) in states.items()}
        feed_jobs=sum(job.op=="FEED" for job in daily)
        covered,_,_=self._deadline_schedule(tomorrow,daily,future_states,feed_jobs)
        return len(covered)==len({job.key for job in daily})

    def _urgent_harvest_tasks(self,obs,worker_ids,tasks):
        """Reserve cumulative travel/action time before the first crop loss."""
        if not worker_ids:
            return []
        now=int(obs["day"])*24+int(obs["hour"])
        positions=_positions(obs)
        urgent=[]
        end=int(obs["day"])*24+23
        due=[t for t in tasks if t.op in ("HARVEST","DIG") and t.deadline_step is not None and t.deadline_step<=end+1]
        critical=[t for t in tasks if _weed_prevention_task(t) or (t.op=="FEED" and _animal_survival_task(t))]
        states={w:self._capacity_states(obs)[w] for w in worker_ids}
        covered,routes,slack=self._deadline_schedule(obs,critical+due,states)
        delayed,_,_=self._deadline_schedule(obs,critical+due,states,delay=1)
        pressured={t.key for w,route in routes.items() if slack[w]<=1 for t in route if t.op in ("HARVEST","DIG")}
        if len(delayed)<len(covered):
            pressured.update(t.key for route in routes.values() for t in route if t.op in ("HARVEST","DIG"))
        for t in tasks:
            if t.op not in ("HARVEST","DIG") or t.deadline_step is None or t.target is None:
                continue
            distances=[self.e.dist(positions[w],t.target) for w in worker_ids]
            if not distances:
                continue
            # One-turn reserve: when waiting one more turn would consume the
            # last safe start opportunity, reserve a worker now.
            # An exhausted ongoing plant has no remaining production benefit.
            # Retire it promptly instead of creating another midnight rescue.
            final_harvest=self._retirement_after_harvest(obs,t) is not None
            if t.op=="DIG" or final_harvest or t.key in pressured or now+min(distances)+1>=int(t.deadline_step):
                urgent.append(t)
        return urgent

    def _retirement_after_harvest(self,obs,t):
        """Commit cleanup once the ongoing crop has produced its final yield."""
        if t.op!="HARVEST" or t.item not in ("TOMATO","STRAWBERRY") or t.target is None:
            return None
        tile=self.e.tile(obs["farms"][obs["player"]],t.target)
        start=_crop_decay_start_step(self.e,tile) if isinstance(tile,dict) else None
        now=int(obs["day"])*24+int(obs["hour"])
        if start is None or now<start-24:
            return None
        return Task(t.target,"DIG",deadline_step=start-1)

    def _max_critical_task_matching(self,obs,worker_ids,tasks,seeds,shed):
        """Maximum reachable assignments for hard-deadline scheduler tasks."""
        if not tasks or not worker_ids:
            return 0
        feasible_by_task=[
            [
                w for w in worker_ids
                if self.feasible(obs,w,t,seeds,shed)
            ]
            for t in tasks
        ]
        order=sorted(range(len(tasks)),key=lambda i:len(feasible_by_task[i]))
        worker_match={}

        def augment(task_i,seen):
            for w in feasible_by_task[task_i]:
                if w in seen:
                    continue
                seen.add(w)
                prev=worker_match.get(w)
                if prev is None or augment(prev,seen):
                    worker_match[w]=task_i
                    return True
            return False

        matched=0
        for task_i in order:
            if augment(task_i,set()):
                matched+=1
        return matched

    def unit_actions(self,obs,animal_plan,crop_plan):
        self._sync_day(obs)
        self._choice_obs=obs
        positions=_positions(obs)
        workers=list(range(len(positions)))
        actions=[None]*len(workers)
        tasks=self.tasks(obs,animal_plan,crop_plan)
        seeds=dict(obs["private"]["seeds"])
        shed=dict(obs["private"]["shed"])
        critical_water_tasks=[t for t in tasks if _weed_prevention_task(t)]
        critical_feed_tasks=[
            t for t in tasks
            if t.op=="FEED" and float(t.critical)>=1.0
        ]
        critical_wheat_supply_tasks=[
            t for t in tasks
            if (
                t.op in ("PICKUP","HARVEST")
                and t.item=="WHEAT"
                and float(t.critical)>=1.0
            )
        ]
        direct_survival_tasks=critical_water_tasks+critical_feed_tasks
        survival_tasks=direct_survival_tasks+critical_wheat_supply_tasks
        deadline_harvest_tasks=self._urgent_harvest_tasks(
            obs,workers,tasks
        )
        critical_tasks=survival_tasks+deadline_harvest_tasks
        direct_survival_keys={t.key for t in direct_survival_tasks}
        survival_supply_keys={t.key for t in critical_wheat_supply_tasks}
        deadline_harvest_keys={t.key for t in deadline_harvest_tasks}
        critical_keys={t.key for t in critical_tasks}
        reserved=set()
        self._capacity_step=int(obs["day"])*24+int(obs["hour"])
        self.turn_capacity_states=self._capacity_states(obs)
        self.turn_capacity_tasks=tasks
        self.turn_capacity_reserved=reserved
        self.turn_capacity_plants=[]
        self.turn_capacity_animals=[]
        # Snapshot provenance before this turn's actions. Reward attribution uses
        # this to cap delivery credit to goods that did not originate in the shed.
        self.turn_delivery_credit=[
            self._delivery_credit_for_worker(obs,w) for w in workers
        ]
        self.turn_plan_credit=[{} for _ in workers]
        self.turn_planned_plant_origins=[]
        self.turn_route_progress=[False for _ in workers]
        self.turn_avoidable_pass=[False for _ in workers]
        self.turn_avoidable_plant_delay=[False for _ in workers]

        state=global_features(self.e,obs,animal_plan,crop_plan,len(workers))
        st=torch.as_tensor(state,dtype=torch.float32,device=self.device)
        with torch.no_grad():
            old_value=float(self.model.value(st).item())
        subs=[]

        def reserve_resources(t):
            if t.op=="PLANT":
                seeds[t.item]=max(0,int(seeds.get(t.item,0))-1)
            elif t.op=="PICKUP":
                shed[t.item]=max(0,int(shed.get(t.item,0))-t.amount)

        def reserve_capacity(w,t):
            projected=self._advance_capacity_state(obs,t,self.turn_capacity_states[w],int(shed.get("WHEAT",0)))
            if projected is not None:
                self.turn_capacity_states[w]=projected[0]
                if t.op=="PLANT":
                    self.turn_capacity_plants.append(Task(
                        t.target,"WATER",t.item,critical=1.0,release_step=projected[0][1]
                    ))
                elif t.op=="PLACE_ANIMAL":
                    self.turn_capacity_animals.append(t)

        def record_reward_provenance(w,t,pos,target,action,avoidable_pass=False):
            if t.planned and pos==target and t.op in ("PLANT","PLACE_ANIMAL"):
                self.turn_plan_credit[w]={"op":t.op,"item":t.item}
            useful_route=t.op!="PLANT" or plant_can_deliver_before_end(
                self.e,obs,t.item,target,self.e.dist(pos,target)
            )
            if useful_route and t.op!="PASS" and pos!=target and action and action[0] in MOVE_ACTIONS:
                self.turn_route_progress[w]=True
            if t.op=="PASS" and avoidable_pass:
                self.turn_avoidable_pass[w]=True

        def assign_committed(w,t,reservation_key):
            pos=positions[w]
            target=t.target if t.target is not None else tuple(self.e.nearest_shed(pos))
            action=self.emit(obs,w,t)
            actions[w]=action
            record_reward_provenance(w,t,pos,target,action)
            reserved.add(reservation_key)
            reserve_capacity(w,t)
            reserve_resources(t)
            # Only actual resource operations change provenance; movement does not.
            if pos==target:
                origin=self.active_origins.pop(w,None)
                if origin is not None:
                    origin.actor_bonus+=self.planned_completion_bonus(t)
                    if t.planned and t.op=="PLANT":
                        self.turn_planned_plant_origins.append(origin)
                    else:
                        origin.reward_equiv_bonus+=self.planned_completion_reward_equiv(t)
                self._track_resource_action(w,action)
                cleanup=self._retirement_after_harvest(obs,t)
                if cleanup is None:
                    self.active_tasks.pop(w,None)
                else:
                    self.active_tasks[w]=cleanup
            else:
                self.active_tasks[w]=t

        # First honor valid routes chosen on earlier turns. A committed route does
        # not create a new PPO actor sample: the actor already chose this job when
        # the route started.
        for w in list(workers):
            active=self.active_tasks.get(w)
            if active is None:
                continue
            # Preserve a non-critical committed route unless this worker is
            # required for lexicographic hard-deadline coverage:
            #   1) immediate survival (critical WATER / FEED)
            #   2) survival resource prerequisite (critical WHEAT PICKUP)
            #   3) imminent crop-decay HARVEST
            # This prevents harvest protection from sacrificing animals and
            # still preempts only the minimum number of unrelated routes.
            if active.key not in critical_keys:
                remaining_direct=[
                    t for t in direct_survival_tasks if t.key not in reserved
                ]
                remaining_supply=[
                    t for t in critical_wheat_supply_tasks if t.key not in reserved
                ]
                remaining_harvest=[
                    t for t in deadline_harvest_tasks if t.key not in reserved
                ]
                other_workers=[q for q in workers if q!=w]
                if remaining_direct or remaining_supply or remaining_harvest:
                    with_direct=self._max_critical_task_matching(
                        obs,workers,remaining_direct,seeds,shed
                    )
                    without_direct=self._max_critical_task_matching(
                        obs,other_workers,remaining_direct,seeds,shed
                    )
                    with_supply=self._max_critical_task_matching(
                        obs,workers,remaining_direct+remaining_supply,seeds,shed
                    )
                    without_supply=self._max_critical_task_matching(
                        obs,other_workers,remaining_direct+remaining_supply,seeds,shed
                    )
                    with_total=self._max_critical_task_matching(
                        obs,workers,
                        remaining_direct+remaining_supply+remaining_harvest,
                        seeds,shed
                    )
                    without_total=self._max_critical_task_matching(
                        obs,other_workers,
                        remaining_direct+remaining_supply+remaining_harvest,
                        seeds,shed
                    )
                    if (
                        without_direct < with_direct
                        or (
                            without_direct == with_direct
                            and without_supply < with_supply
                        )
                        or (
                            without_direct == with_direct
                            and without_supply == with_supply
                            and without_total < with_total
                        )
                    ):
                        self.active_tasks.pop(w,None)
                        self.active_origins.pop(w,None)
                        continue
            candidates=tasks+self.extras(obs,w)
            current=next((t for t in candidates if t.key==active.key),None)
            if current is None or not self.feasible(obs,w,current,seeds,shed):
                self.active_tasks.pop(w,None)
                self.active_origins.pop(w,None)
                continue
            reservation_key=(w,current.key) if current.op=="DELIVER" else current.key
            if reservation_key in reserved:
                self.active_tasks.pop(w,None)
                self.active_origins.pop(w,None)
                continue
            assign_committed(w,current,reservation_key)
            workers.remove(w)

        # Uncommitted workers receive new PPO task assignments.
        while workers:
            choices=[]; feats=[]
            for w in workers:
                for t in tasks+self.extras(obs,w):
                    reservation_key=(w,t.key) if t.op=="DELIVER" else t.key
                    if t.op!="PASS" and reservation_key in reserved:
                        continue
                    if not self.feasible(obs,w,t,seeds,shed):
                        continue
                    choices.append((w,t,reservation_key))
                    feats.append(candidate_features(self.e,obs,w,t,len(workers)))

            # Survival comes before production preservation. First force
            # critical WATER/FEED, then any WHEAT pickup needed to make
            # emergency FEED possible, and only then imminent decay-HARVEST.
            urgent=[
                i for i,(_w,t,_key) in enumerate(choices)
                if t.key in direct_survival_keys
            ]
            if not urgent:
                urgent=[
                    i for i,(_w,t,_key) in enumerate(choices)
                    if t.key in survival_supply_keys
                ]
            if not urgent:
                urgent=[
                    i for i,(_w,t,_key) in enumerate(choices)
                    if t.key in deadline_harvest_keys
                ]
            if urgent:
                choices=[choices[i] for i in urgent]
                feats=[feats[i] for i in urgent]
                # Keep PPO's urgent worker/task choice only when it preserves
                # the reachable deadline coverage. A greedy actor must not
                # give a flexible worker the sole job of a specialist worker.
                remaining=[job for job in direct_survival_tasks+deadline_harvest_tasks if job.key not in reserved]
                baseline,_,_=self._deadline_schedule(obs,remaining,self.turn_capacity_states,int(shed.get("WHEAT",0)))
                safe=[]
                for i,(w,t,_key) in enumerate(choices):
                    projected=self._advance_capacity_state(obs,t,self.turn_capacity_states[w],int(shed.get("WHEAT",0)))
                    if projected is None:
                        continue
                    states=dict(self.turn_capacity_states)
                    states[w]=projected[0]
                    pending=[job for job in remaining if job.key!=t.key]
                    covered,_,_=self._deadline_schedule(obs,pending,states,int(shed.get("WHEAT",0))-projected[1])
                    completed=covered|({t.key} if t.key in {job.key for job in remaining} else set())
                    if baseline<=completed:
                        safe.append(i)
                if safe:
                    choices=[choices[i] for i in safe]
                    feats=[feats[i] for i in safe]

            if not urgent:
                # Finish the planner's animal pipeline before ordinary actor
                # work. Preserve today's reachable maintenance/deadlines when
                # considering setup movement, clearing, BUILD and PICKUP too.
                maintenance=[job for job in self._maintenance_tasks(obs,tasks)
                             if job.key not in reserved]+self.turn_capacity_plants
                stock=int(shed.get("WHEAT",0))
                baseline,_,_=self._deadline_schedule(obs,maintenance,self.turn_capacity_states,stock)
                setup=[]
                for i,(w,t,_key) in enumerate(choices):
                    if not _animal_setup_task(t):
                        continue
                    projected=self._advance_capacity_state(obs,t,self.turn_capacity_states[w],stock)
                    if projected is None:
                        continue
                    states=dict(self.turn_capacity_states)
                    states[w]=projected[0]
                    covered,_,_=self._deadline_schedule(obs,maintenance,states,stock-projected[1])
                    if baseline<=covered:
                        setup.append(i)
                if setup:
                    # Prefer completing a placement, then supplying it, before
                    # starting more structures. PPO still assigns workers/jobs.
                    rank={"PLACE_ANIMAL":0,"PICKUP":1,"BUILD_COOP":2,"BUILD_PASTURE":2,"DIG":3}
                    stage=min(rank[choices[i][1].op] for i in setup)
                    setup=[i for i in setup if rank[choices[i][1].op]==stage]
                    choices=[choices[i] for i in setup]
                    feats=[feats[i] for i in setup]

            # PASS remains valid for workers with no feasible unreserved job.
            # Mask it per worker, so one busy worker does not hide another's
            # genuinely necessary idle action. PPO samples the filtered set.
            busy={w for w,t,_ in choices if t.op!="PASS"}
            keep=[i for i,(w,t,_key) in enumerate(choices) if t.op!="PASS" or w not in busy]
            choices=[choices[i] for i in keep]
            feats=[feats[i] for i in keep]

            if not choices:
                for w in workers:
                    actions[w]=["PASS"]
                    self.active_tasks.pop(w,None)
                break

            mat=np.stack(feats).astype(np.float32)
            ct=torch.as_tensor(mat,dtype=torch.float32,device=self.device)
            with torch.no_grad():
                raw_logits=self.model.logits(ct)
                behavior_logits=raw_logits if self.deterministic else raw_logits/self.rollout_temperature
                dist=Categorical(logits=behavior_logits)
                a=torch.argmax(raw_logits) if self.deterministic else dist.sample()
                lp=dist.log_prob(a)

            j=int(a.item())
            w,t,reservation_key=choices[j]
            pos=positions[w]
            target=t.target if t.target is not None else tuple(self.e.nearest_shed(pos))
            action=self.emit(obs,w,t)
            actions[w]=action
            avoidable_pass=(
                t.op=="PASS"
                and any(cw==w and ct.op!="PASS" for cw,ct,_ in choices)
            )
            record_reward_provenance(w,t,pos,target,action,avoidable_pass)
            self.turn_avoidable_plant_delay[w]=self.avoidable_plant_delay(w,t,choices)
            completed_now=(pos==target)
            actor_bonus=self.actor_bonus_for_choice(
                w,t,choices,completed=completed_now
            )
            # Planned PLANT reward-equivalent credit is held until finish_turn()
            # confirms the actual tile transition succeeded.
            reward_equiv_bonus=0.0
            self.candidate_counts.append(len(choices))
            sub=None
            if self.collect:
                sub=SubDecision(
                    mat.astype(np.float16),j,float(lp.item()),
                    actor_bonus,reward_equiv_bonus
                )
                subs.append(sub)
                if completed_now and t.planned and t.op=="PLANT":
                    self.turn_planned_plant_origins.append(sub)
                elif completed_now:
                    sub.reward_equiv_bonus+=self.planned_completion_reward_equiv(t)

            workers.remove(w)
            if t.op!="PASS":
                reserved.add(reservation_key)
            reserve_capacity(w,t)
            reserve_resources(t)

            if t.op!="PASS" and pos!=target:
                self.active_tasks[w]=t
                if self.collect and sub is not None and t.planned:
                    self.active_origins[w]=sub
                else:
                    self.active_origins.pop(w,None)
            else:
                cleanup=self._retirement_after_harvest(obs,t) if pos==target else None
                if cleanup is None:
                    self.active_tasks.pop(w,None)
                else:
                    self.active_tasks[w]=cleanup
                self.active_origins.pop(w,None)
                if pos==target:
                    self._track_resource_action(w,action)

        if self.collect:
            self.pending=TurnRecord(
                state,subs,old_value,int(obs["day"])*24+int(obs["hour"])
            )
        return [a or ["PASS"] for a in actions]

    def finish_turn(self,reward: RewardBreakdown):
        if not self.collect: return
        if self.pending is None: raise RuntimeError("missing pending turn")
        # Credit only engine-confirmed planned PLANT completions.  This mirrors
        # HARVEST reward semantics: emitting the action is not enough.
        completed=max(0,int(reward.planned_plants_completed))
        for origin in self.turn_planned_plant_origins[:completed]:
            origin.reward_equiv_bonus+=PLANNED_PLANT_REWARD_EQUIV
        self.turn_planned_plant_origins=[]
        self.pending.reward=float(reward.reward); self.pending.reward_breakdown=reward.as_dict(); self.records.append(self.pending); self.pending=None
