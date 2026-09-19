#!/usr/bin/env python3
"""Evaluate a trained v20 constrained-task PPO checkpoint against frozen v19."""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from train_v20_ppo import (
    ActorCritic, DEFAULT_EXECUTOR, GLOBAL_FEATURE_NAMES, TASK_FEATURE_NAMES,
    choose_device, load_checkpoint, prepare_opponents, run_episode,
)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--executor",type=Path,default=DEFAULT_EXECUTOR)
    p.add_argument("--opponent",type=Path,default=DEFAULT_EXECUTOR)
    p.add_argument("--games",type=int,default=20,help="number of seeds; both seats are played")
    p.add_argument("--seed",type=int,default=42020)
    p.add_argument("--episode-steps",type=int,default=720)
    p.add_argument("--device",default="auto")
    p.add_argument("--output",type=Path)
    return p


def main():
    args=parser().parse_args();device=choose_device(args.device)
    checkpoint=args.checkpoint.expanduser().resolve()
    payload=torch.load(checkpoint,map_location=device,weights_only=False)
    hidden=int(payload.get("args",{}).get("hidden",64));baseline_scale=float(payload.get("baseline_scale",1.0))
    model=ActorCritic(len(TASK_FEATURE_NAMES),len(GLOBAL_FEATURE_NAMES),hidden).to(device)
    load_checkpoint(checkpoint,model,None,device);model.eval()
    rng=np.random.default_rng(args.seed);rows=[]
    with tempfile.TemporaryDirectory(prefix="v20_eval_") as tmp:
        opponent=prepare_opponents([str(args.opponent.expanduser().resolve())],Path(tmp))[0]
        for _ in range(args.games):
            seed=int(rng.integers(1,2_147_483_647))
            for seat in (0,1):
                result,_=run_episode(
                    model,device,args.executor.expanduser().resolve(),opponent,seed,seat,
                    args.episode_steps,baseline_scale,deterministic=True,
                    forced_baseline=False,collect_steps=False,
                )
                rows.append(asdict(result));print(json.dumps(rows[-1],sort_keys=True))
    ok=[r for r in rows if r["ok"]];margins=np.asarray([r["margin"] for r in ok],dtype=np.float64)
    summary={
        "checkpoint":str(checkpoint),"opponent":str(args.opponent),"games_requested":args.games*2,
        "games_ok":len(ok),"wins":int((margins>0).sum()) if len(margins) else 0,
        "ties":int((margins==0).sum()) if len(margins) else 0,
        "losses":int((margins<0).sum()) if len(margins) else 0,
        "win_rate":float((margins>0).mean()) if len(margins) else None,
        "mean_margin":float(margins.mean()) if len(margins) else None,
        "median_margin":float(np.median(margins)) if len(margins) else None,
    }
    print(json.dumps(summary,indent=2,sort_keys=True))
    if args.output:
        args.output.expanduser().resolve().write_text(
            json.dumps({"summary":summary,"games":rows},indent=2)+"\n",encoding="utf-8"
        )


if __name__=="__main__":main()
