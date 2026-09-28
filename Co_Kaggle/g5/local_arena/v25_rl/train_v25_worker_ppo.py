#!/usr/bin/env python3
"""PPO/GAE training for v25 worker actions on static v20 losses.

Only farmer/hands are learned. The game still receives a market action list so
hiring, procurement, selling, and land purchases continue, but that list is
replayed from the recorded v20 history and is not generated, scored, or updated
by PPO. No final money, margin, WIN/TIE/LOSS, or market-price reward enters GAE.
"""
from __future__ import annotations
import argparse, copy, importlib.util, json, math, random, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

HERE=Path(__file__).resolve().parent
LOCAL_ARENA=HERE.parent
G5_ROOT=LOCAL_ARENA.parent
V20_RL=LOCAL_ARENA/"v20_rl"
if str(V20_RL) not in sys.path: sys.path.insert(0,str(V20_RL))
from evaluate_v20_v19_losses import _agent_observation,_environment_from_history,_field,_recorded_step_actions,_saved_final_rewards,_seed_hint,recorded_action_parity
from worker_policy import ActorCritic,CANDIDATE_FEATURE_NAMES,GLOBAL_FEATURE_NAMES,TurnRecord,WorkerPolicy
from worker_reward import ANIMAL_ESCAPE_PENALTY,CROP_DEATH_PENALTY,CROP_TO_WEED_PENALTY,LOST_HARVESTABLE_UNIT_PENALTY,PRODUCT_DELIVERED_REWARD,PRODUCT_GENERATED_REWARD,PRODUCT_HARVESTED_REWARD,PRODUCT_VALUE,RewardBreakdown,compute_worker_reward

DEFAULT_HISTORY_DIR=G5_ROOT/"game_history"/"v20"
DEFAULT_EXECUTOR=HERE/"v25_rl.py"
DEFAULT_OUTPUT_DIR=HERE/"runs"/"worker_ppo_static_v20"
CHECKPOINT_ALGORITHM="v25_static_worker_ppo_gae_v2_subdecision_ratio"

def load_executor(path):
    spec=importlib.util.spec_from_file_location(f"v25_ep_{time.time_ns()}",path)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot load {path}")
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def load_history(path):
    h=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(h.get("steps"),list) or len(h["steps"])<2: raise ValueError(f"{path}: invalid replay")
    return h

def loss_seat(h):
    r=_saved_final_rewards(h)
    if r[0]==r[1]: raise ValueError("tied v20 history")
    return 0 if r[0]<r[1] else 1

def split_histories(paths,fraction,seed):
    if len(paths)<2: raise ValueError("need >=2 histories")
    xs=list(paths); random.Random(seed).shuffle(xs); n=max(1,min(len(xs)-1,int(round(len(xs)*fraction))))
    return sorted(xs[n:]),sorted(xs[:n])

@dataclass
class EpisodeResult:
    episode:str
    seed:Any
    ok:bool
    loss_seat:int
    worker_reward:float
    turns:int
    reward_breakdown:dict
    mean_candidates:float
    max_candidates:int
    error:str=""

def worker_policy_action(e,policy,obs):
    """Return only the trainable farmer/hands component."""
    # Preserve the small opponent-style initialization performed by v25 agent()
    # because crop/animal planning can reference OPP_STYLE. Do not call agent():
    # agent() would also generate market_orders(), which is outside worker RL.
    if obs["day"]==0 and obs["hour"]==0:
        e.OPP_STYLE=None
    elif getattr(e,"OPP_STYLE",None) is None and obs["day"]==0 and obs["hour"]>0:
        other=obs["farms"][1-obs["player"]]
        e.OPP_STYLE="V16" if other["hires_today"]==5 and 20<=other["money"]<120 else "NORMAL"

    signals=e.production_signals(obs)
    animal=e.animal_plan(obs,signals)
    crops=e.crop_plan(obs,signals)
    workers=policy.unit_actions(obs,animal,crops)
    return {"farmer":workers[0],"hands":workers[1:]}


def recorded_market_action(recorded_candidate):
    """Frozen market input needed to keep the replayed game economy running."""
    if not isinstance(recorded_candidate,dict):
        raise RuntimeError("recorded candidate action is not a dict")
    return copy.deepcopy(recorded_candidate.get("market") or [])


def compose_environment_action(worker_action,market_action):
    """Combine learned workers with non-learned market input only for env.step()."""
    return {
        "farmer":worker_action["farmer"],
        "hands":worker_action["hands"],
        "market":market_action,
    }


def run_static_episode(path,model,device,executor_path,deterministic=False,collect=True):
    h=load_history(path); seat=loss_seat(h); other=1-seat; e=load_executor(executor_path)
    policy=WorkerPolicy(e,model,device,deterministic,collect)
    env=_environment_from_history(h); total=RewardBreakdown()
    try:
        for replay_step in range(1,len(h["steps"])):
            obs=_agent_observation(env,seat)
            recorded=_recorded_step_actions(h,replay_step)
            opponent=recorded[other]; recorded_candidate=recorded[seat]
            if opponent is None: raise RuntimeError(f"None opponent action at {replay_step}")
            if recorded_candidate is None: raise RuntimeError(f"None recorded candidate action at {replay_step}")

            # PPO controls only this object.
            worker_action=worker_policy_action(e,policy,obs)

            # The market stream is required for the game economy, but it is a
            # frozen replay input: no policy log-probability and no reward term.
            market_action=recorded_market_action(recorded_candidate)
            candidate=compose_environment_action(worker_action,market_action)

            actions=[None,None]; actions[seat]=candidate; actions[other]=opponent; env.step(actions)
            post=_agent_observation(env,seat)

            # Deliberately pass worker_action, not candidate. This makes it
            # impossible for market commands to enter direct reward attribution.
            reward=compute_worker_reward(e,obs,worker_action,post)
            total.add(reward); policy.finish_turn(reward)
        statuses=[str(_field(s,"status","")) for s in env.steps[-1]]; ok=statuses==["DONE","DONE"]
        return EpisodeResult(path.stem,_seed_hint(h),ok,seat,float(total.reward),len(policy.records) if collect else len(h["steps"])-1,total.as_dict(),float(np.mean(policy.candidate_counts)) if policy.candidate_counts else 0.0,max(policy.candidate_counts) if policy.candidate_counts else 0,"" if ok else f"status={statuses}"),policy.records
    except Exception as exc:
        return EpisodeResult(path.stem,_seed_hint(h),False,seat,float(total.reward),len(policy.records),total.as_dict(),float(np.mean(policy.candidate_counts)) if policy.candidate_counts else 0.0,max(policy.candidate_counts) if policy.candidate_counts else 0,f"{type(exc).__name__}: {exc}"),policy.records

def assign_gae(records,gamma,lam):
    """GAE over turn-level worker reward only. There is no terminal game reward."""
    adv=0.0
    for i in range(len(records)-1,-1,-1):
        r=records[i]; nv=0.0 if i==len(records)-1 else records[i+1].old_value
        delta=r.reward+gamma*nv-r.old_value; adv=delta+gamma*lam*adv
        r.advantage=float(adv); r.return_target=float(adv+r.old_value)

def subdecision_logprob_entropy(model,sub,device):
    c=torch.as_tensor(sub.candidates,dtype=torch.float32,device=device)
    d=Categorical(logits=model.logits(c))
    idx=torch.tensor(sub.action_index,device=device)
    return d.log_prob(idx),d.entropy().mean()

def ppo_update(model,opt,device,records,args):
    """PPO with one clipped ratio per worker subdecision.

    GAE/value targets remain turn-level.  Every worker assignment selected in a
    turn shares that turn's normalized advantage, but its PPO ratio uses only
    its own old/new log probability.  This avoids exponentiating a sum of many
    worker log-probability changes into one unstable joint-action ratio.
    """
    if not records:
        raise ValueError("no PPO records")

    turn_adv=np.asarray([r.advantage for r in records],np.float32)
    turn_adv=(turn_adv-turn_adv.mean())/(turn_adv.std()+1e-8)
    ret=np.asarray([r.return_target for r in records],np.float32)

    actor_samples=[
        (turn_idx,sub)
        for turn_idx,record in enumerate(records)
        for sub in record.subdecisions
    ]
    if not actor_samples:
        raise ValueError("no PPO worker subdecisions")

    actor_adv=np.asarray([turn_adv[turn_idx] for turn_idx,_ in actor_samples],np.float32)
    actor_old=np.asarray([sub.old_log_prob for _,sub in actor_samples],np.float32)

    actor_stats=[]
    value_stats=[]
    epochs_completed=0
    kl_early_stop=False

    for _ in range(args.ppo_epochs):
        actor_order=np.random.permutation(len(actor_samples))
        epoch_kls=[]

        for start in range(0,len(actor_order),args.minibatch_size):
            ids=actor_order[start:start+args.minibatch_size]
            new_lp=[]; ent=[]
            for q in ids:
                _,sub=actor_samples[int(q)]
                lp,e=subdecision_logprob_entropy(model,sub,device)
                new_lp.append(lp); ent.append(e)

            new_lp=torch.stack(new_lp)
            entropy=torch.stack(ent).mean()
            oldt=torch.as_tensor(actor_old[ids],dtype=torch.float32,device=device)
            at=torch.as_tensor(actor_adv[ids],dtype=torch.float32,device=device)
            log_ratio=new_lp-oldt
            ratio=torch.exp(torch.clamp(log_ratio,-20,20))
            unclipped=ratio*at
            clipped=torch.clamp(ratio,1-args.clip_ratio,1+args.clip_ratio)*at
            policy_loss=-torch.minimum(unclipped,clipped).mean()
            actor_loss=policy_loss-args.entropy_coef*entropy

            opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(model.actor.parameters(),args.max_grad_norm)
            opt.step()

            approx_kl=((ratio-1.0)-log_ratio).mean()
            clip_fraction=((ratio-1.0).abs()>args.clip_ratio).float().mean()
            epoch_kls.append(float(approx_kl.item()))
            actor_stats.append((
                float(policy_loss.item()),
                float(entropy.item()),
                float(approx_kl.item()),
                float(clip_fraction.item()),
                float(ratio.mean().item()),
                float(ratio.std(unbiased=False).item()),
                float(ratio.min().item()),
                float(ratio.max().item()),
            ))

        # The critic is still trained once per environment turn.
        value_order=np.random.permutation(len(records))
        for start in range(0,len(value_order),args.minibatch_size):
            ids=value_order[start:start+args.minibatch_size]
            values=torch.stack([
                model.value(torch.as_tensor(records[int(q)].state,dtype=torch.float32,device=device))
                for q in ids
            ])
            rt=torch.as_tensor(ret[ids],dtype=torch.float32,device=device)
            value_loss=.5*(values-rt).pow(2).mean()
            critic_loss=args.value_coef*value_loss

            opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(model.critic.parameters(),args.max_grad_norm)
            opt.step()
            value_stats.append(float(value_loss.item()))

        epochs_completed+=1
        mean_epoch_kl=float(np.mean(epoch_kls)) if epoch_kls else 0.0
        if args.target_kl>0 and mean_epoch_kl>args.target_kl:
            kl_early_stop=True
            break

    a=np.asarray(actor_stats,float)
    policy_loss=float(a[:,0].mean())
    entropy=float(a[:,1].mean())
    approx_kl=float(a[:,2].mean())
    clip_fraction=float(a[:,3].mean())
    ratio_mean=float(a[:,4].mean())
    ratio_std=float(a[:,5].mean())
    ratio_min=float(a[:,6].min())
    ratio_max=float(a[:,7].max())
    value_loss=float(np.mean(value_stats)) if value_stats else 0.0

    return {
        "policy_loss":policy_loss,
        "value_loss":value_loss,
        "entropy":entropy,
        "loss":policy_loss+args.value_coef*value_loss-args.entropy_coef*entropy,
        "approx_kl":approx_kl,
        "clip_fraction":clip_fraction,
        "ratio_mean":ratio_mean,
        "ratio_std":ratio_std,
        "ratio_min":ratio_min,
        "ratio_max":ratio_max,
        "return_mean":float(ret.mean()),
        "actor_samples":len(actor_samples),
        "mean_subdecisions_per_turn":float(len(actor_samples)/len(records)),
        "ppo_epochs_completed":epochs_completed,
        "kl_early_stop":kl_early_stop,
    }

def summary(results,phase):
    valid=[r for r in results if r.ok]
    out={
        "phase":phase,
        "episodes":len(results),
        "episodes_ok":len(valid),
        "errors":len(results)-len(valid),
        "mean_worker_reward":float(np.mean([r.worker_reward for r in valid])) if valid else None,
        "min_worker_reward":float(np.min([r.worker_reward for r in valid])) if valid else None,
        "max_worker_reward":float(np.max([r.worker_reward for r in valid])) if valid else None,
        "mean_candidates":float(np.mean([r.mean_candidates for r in valid])) if valid else None,
    }
    for k in RewardBreakdown.__dataclass_fields__:
        if k=="reward":
            continue
        values=[r.reward_breakdown.get(k,0) for r in valid]
        if values and isinstance(values[0],dict):
            keys=sorted({key for value in values for key in value})
            means={
                key:float(np.mean([float(value.get(key,0.0)) for value in values]))
                for key in keys
            }
            out["mean_"+k]=means
            for key,value in means.items():
                out[f"mean_{k}_{key.lower()}"]=value
        else:
            out["mean_"+k]=float(np.mean([float(value) for value in values])) if values else None
    return out

def evaluate(paths,model,device,executor,phase):
    model.eval(); rows=[]
    with torch.inference_mode():
        for i,p in enumerate(paths,1):
            r,_=run_static_episode(p,model,device,executor,True,False); rows.append(r)
            print(
                f"[{phase}] {i}/{len(paths)} {p.stem} "
                f"reward={r.worker_reward:+.1f} "
                f"planted={r.reward_breakdown.get('seeds_planted_total',0)} "
                f"crop_harvested={r.reward_breakdown.get('crop_units_harvested_total',0)} "
                f"animal_made={r.reward_breakdown.get('animal_product_units_generated_total',0)} "
                f"to_shed={r.reward_breakdown.get('product_units_moved_to_shed_total',0)} "
                f"escape={r.reward_breakdown.get('animals_escaped',0)} "
                f"weed={r.reward_breakdown.get('crops_to_weed',0)} "
                f"{'OK' if r.ok else r.error}",
                flush=True,
            )
    model.train(); return rows,summary(rows,phase)

def write_jsonl(path,row):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as f: f.write(json.dumps(row,sort_keys=True,default=str)+"\n")

def device_for(v):
    if v!="auto": return torch.device(v)
    if torch.cuda.is_available(): return torch.device("cuda")
    if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def save_checkpoint(path,model,opt,update,args,best):
    torch.save({"algorithm":CHECKPOINT_ALGORITHM,"update":update,"model_state_dict":model.state_dict(),"optimizer_state_dict":opt.state_dict(),"candidate_feature_names":CANDIDATE_FEATURE_NAMES,"global_feature_names":GLOBAL_FEATURE_NAMES,"best_validation_worker_reward":best,"reward_contract":{"product_value":PRODUCT_VALUE,"generated":PRODUCT_GENERATED_REWARD,"harvested":PRODUCT_HARVESTED_REWARD,"delivered":PRODUCT_DELIVERED_REWARD,"animal_escape":ANIMAL_ESCAPE_PENALTY,"crop_to_weed":CROP_TO_WEED_PENALTY,"crop_death":CROP_DEATH_PENALTY,"lost_harvestable_unit":LOST_HARVESTABLE_UNIT_PENALTY},"args":vars(args),"note":"PPO controls farmer/hands only; per-worker subdecision ratios; turn-level GAE/value; recorded market list is an unlearned env input; no final game result/money/margin/market reward"},path)

def load_checkpoint(path,model,opt,device):
    p=torch.load(path,map_location=device,weights_only=False)
    if p.get("algorithm")!=CHECKPOINT_ALGORITHM: raise ValueError("checkpoint algorithm mismatch")
    model.load_state_dict(p["model_state_dict"])
    if p.get("optimizer_state_dict"): opt.load_state_dict(p["optimizer_state_dict"])
    return int(p.get("update",-1))+1,float(p.get("best_validation_worker_reward",-math.inf))

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history-dir",type=Path,default=DEFAULT_HISTORY_DIR); p.add_argument("--executor",type=Path,default=DEFAULT_EXECUTOR); p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT_DIR); p.add_argument("--resume",type=Path)
    p.add_argument("--validation-fraction",type=float,default=.20); p.add_argument("--split-seed",type=int,default=20260928); p.add_argument("--training-seed",type=int,default=32525)
    p.add_argument("--updates",type=int,default=100); p.add_argument("--episodes-per-update",type=int,default=8); p.add_argument("--validate-every-updates",type=int,default=1); p.add_argument("--checkpoint-every-updates",type=int,default=1); p.add_argument("--max-training-hours",type=float,default=2.0)
    p.add_argument("--device",default="auto"); p.add_argument("--hidden",type=int,default=128); p.add_argument("--learning-rate",type=float,default=3e-4); p.add_argument("--gamma",type=float,default=.99); p.add_argument("--gae-lambda",type=float,default=.95); p.add_argument("--ppo-epochs",type=int,default=4); p.add_argument("--minibatch-size",type=int,default=128); p.add_argument("--clip-ratio",type=float,default=.2); p.add_argument("--target-kl",type=float,default=.02); p.add_argument("--value-coef",type=float,default=.5); p.add_argument("--entropy-coef",type=float,default=.01); p.add_argument("--max-grad-norm",type=float,default=.5); p.add_argument("--preflight-only",action="store_true")
    return p

def main():
    args=parser().parse_args(); hd=args.history_dir.expanduser().resolve(); ex=args.executor.expanduser().resolve(); out=args.output_dir.expanduser().resolve()
    if not hd.is_dir(): raise SystemExit(f"history dir missing: {hd}")
    if not ex.is_file(): raise SystemExit(f"executor missing: {ex}")
    paths=sorted(hd.glob("*.json"))
    if not paths: raise SystemExit("no histories")
    train,val=split_histories(paths,args.validation_fraction,args.split_seed); out.mkdir(parents=True,exist_ok=True); ck=out/"checkpoints"; ck.mkdir(parents=True,exist_ok=True)
    gate=recorded_action_parity(load_history(paths[0]))
    if not gate.get("exact"): raise SystemExit(f"static replay parity failed: {gate.get('mismatch')}")
    print(f"preflight replay parity OK: {paths[0].stem}",flush=True)
    if args.preflight_only: return 0
    device=device_for(args.device); random.seed(args.training_seed); np.random.seed(args.training_seed); torch.manual_seed(args.training_seed); rng=random.Random(args.training_seed)
    model=ActorCritic(len(CANDIDATE_FEATURE_NAMES),len(GLOBAL_FEATURE_NAMES),args.hidden).to(device); opt=torch.optim.Adam(model.parameters(),lr=args.learning_rate); start=0; best=-math.inf
    if args.resume: start,best=load_checkpoint(args.resume.expanduser().resolve(),model,opt,device)
    (out/"split.json").write_text(json.dumps({"train":[p.name for p in train],"validation":[p.name for p in val],"split_seed":args.split_seed},indent=2)+"\n")
    (out/"config.json").write_text(json.dumps({**vars(args),"executor":str(ex),"device_resolved":str(device),"algorithm":CHECKPOINT_ALGORITHM,"objective":"worker efficiency only; farmer/hands replaced by RL, recorded v20 market orders replayed unchanged; no final game result reward","product_value":PRODUCT_VALUE},indent=2,default=str)+"\n")
    rows,base=evaluate(val,model,device,ex,"baseline"); write_jsonl(out/"validation.jsonl",{"update":-1,**base})
    for r in rows: write_jsonl(out/"validation_episodes.jsonl",{"update":-1,**r.__dict__})
    if base["mean_worker_reward"] is not None:
        best=max(best,float(base["mean_worker_reward"]))
        # Make best.pt truthful even when the untrained baseline remains the
        # best held-out worker policy for the entire run.
        save_checkpoint(ck/"best.pt",model,opt,-1,args,best)
    started=time.perf_counter(); last=start-1
    for u in range(start,args.updates):
        if (time.perf_counter()-started)/3600>=args.max_training_hours: break
        results=[]; records=[]; attempts=0
        while len(results)<args.episodes_per_update:
            attempts+=1
            if attempts>args.episodes_per_update*4: raise RuntimeError("too many failed episodes")
            path=rng.choice(train); r,recs=run_static_episode(path,model,device,ex,False,True); write_jsonl(out/"episodes.jsonl",{"update":u,**r.__dict__})
            print(
                f"[train u{u}] {path.stem} "
                f"reward={r.worker_reward:+.1f} "
                f"planted={r.reward_breakdown.get('seeds_planted_total',0)} "
                f"crop_harvested={r.reward_breakdown.get('crop_units_harvested_total',0)} "
                f"animal_made={r.reward_breakdown.get('animal_product_units_generated_total',0)} "
                f"to_shed={r.reward_breakdown.get('product_units_moved_to_shed_total',0)} "
                f"escape={r.reward_breakdown.get('animals_escaped',0)} "
                f"weed={r.reward_breakdown.get('crops_to_weed',0)} "
                f"{'OK' if r.ok else r.error}",
                flush=True,
            )
            if not r.ok or not recs: continue
            assign_gae(recs,args.gamma,args.gae_lambda); records.extend(recs); results.append(r)
        stats=ppo_update(model,opt,device,records,args); last=u; metrics={"update":u,"turn_records":len(records),"elapsed_hours":(time.perf_counter()-started)/3600,**summary(results,"train"),**stats}; write_jsonl(out/"metrics.jsonl",metrics); print(json.dumps(metrics,sort_keys=True),flush=True)
        if u%args.checkpoint_every_updates==0:
            save_checkpoint(ck/f"update_{u:04d}.pt",model,opt,u,args,best); save_checkpoint(ck/"latest.pt",model,opt,u,args,best)
        if u%args.validate_every_updates==0:
            rows,s=evaluate(val,model,device,ex,f"validation u{u}"); write_jsonl(out/"validation.jsonl",{"update":u,**s})
            for r in rows: write_jsonl(out/"validation_episodes.jsonl",{"update":u,**r.__dict__})
            score=s.get("mean_worker_reward")
            if score is not None and float(score)>best: best=float(score); save_checkpoint(ck/"best.pt",model,opt,u,args,best); print(f"new best worker reward={best:+.3f}",flush=True)
    save_checkpoint(ck/"latest.pt",model,opt,last,args,best); return 0

if __name__=="__main__": raise SystemExit(main())
