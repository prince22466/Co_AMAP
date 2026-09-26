"""Evaluate selling-only changes with a frozen v21 checkpoint and static opponents."""
from __future__ import annotations

import argparse
import ast
import copy
import functools
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import random
import time
import sys
from pathlib import Path

import torch
import train_v21_static_history as train

MODEL = ARGS = None
ENGINE = None
POLICY = None
LAST_ENV = None
#CURRENT_POLICY = train.InventorySellingPolicy
CURRENT_POLICY = train.InventoryBinarySellingPolicy
EXPERIMENTS = {}


class ExperimentalPolicy(train.InventorySellingPolicy):
    """Screen generic selling parameters using only the current observation/history."""
    parameters = {}

    def orders(self, obs, held, total, reserve_wheat, reserve_fert):
        cfg = self.parameters
        prices = obs['market']['prices']
        day = obs['day']; turn = day*24+obs['hour']
        if self.price_history and turn < self.price_history[-1][0]:
            self.price_history.clear()
        window = cfg.get('window',72)
        self.price_history = [(t,p) for t,p in self.price_history if turn-window<t<turn]
        self.price_history.append((turn,dict(prices)))
        available = {c:max(0,int(held.get(c,0))) for c in train.DELIVERED_PRODUCTS}
        for c,reserve in [('WHEAT',0 if day==29 else reserve_wheat),('FERTILIZER',reserve_fert)]:
            if day < 25:
                if c == 'WHEAT' and cfg.get('feed_days'):
                    farm=obs['farms'][obs['player']]
                    animals=sum(isinstance(tile,dict) and 'animal' in tile for row in farm['tiles'] for tile in row)
                    reserve=max(reserve,int(math.ceil(animals*cfg['feed_days'])))
                if c == 'FERTILIZER':
                    reserve=max(reserve,cfg.get('fert_buffer',0))
            available[c] = min(available[c],max(0,int(total.get(c,0))-reserve))
        quantities = dict.fromkeys(available,0)
        for c,n in available.items():
            high = max(p[c] for _,p in self.price_history)
            good = prices[c] >= cfg.get('threshold',1.0)*high
            if cfg.get('forecast_hours'):
                inventory = obs['market']['inventory'][c]
                past = getattr(self,'inventory_history',[])
                reference = next(((t,inv,sold) for t,inv,sold in past if t>=turn-24),None)
                own_sold = getattr(self,'sold_totals',dict.fromkeys(available,0))
                drift = 0.0
                if reference is not None and turn>reference[0]:
                    drift=(inventory-reference[1][c]-(own_sold[c]-reference[2][c]))/(turn-reference[0])
                projected=inventory+drift*min(cfg['forecast_hours'],719-turn)
                future=ENGINE['market_price'](c,projected,obs['market'].get('params'))
                good=prices[c]>=future*cfg.get('future_discount',0.97)
            if cfg.get('declining'):
                old = self.price_history[max(0,len(self.price_history)-25)][1][c]
                good = prices[c] <= old
            if c in cfg.get('always',[]):
                good = True
            if day >= cfg.get('liquidate_day',25):
                good = True
            if good:
                quantities[c] = min(n,cfg.get('batch_limit',10000)) if day<29 else n
        ranked = sorted(available,key=lambda c:-prices[c])
        stock = sum(held.values())
        limit = cfg.get('late_stock',25) if day>=20 else cfg.get('stock_target',60) if stock>cfg.get('stock_trigger',90) else stock
        excess = max(0,stock-sum(quantities.values())-limit)
        cash = obs['farms'][obs['player']]['money']
        cash_target = cfg.get('cash_early',0) if day<16 else cfg.get('cash_late',0)
        shortage = max(0,cash_target-cash-sum(quantities[c]*max(1,prices[c]*.8) for c in quantities))
        for c in ranked:
            remaining = available[c]-quantities[c]
            needed = max(excess,int(math.ceil(shortage/max(1,prices[c]*.8))))
            extra = min(remaining,needed)
            quantities[c]+=extra
            excess=max(0,excess-extra)
            shortage=max(0,shortage-extra*max(1,prices[c]*.8))
        if cfg.get('rank_total'):
            ranked = sorted(available,key=lambda c:-prices[c]*quantities[c])
        orders = [['SELL',c,quantities[c]] for c in ranked if quantities[c]]
        orders = orders[:cfg.get('order_limit',9)] if day<29 else orders
        if cfg.get('forecast_hours'):
            if not hasattr(self,'sold_totals'):
                self.sold_totals=dict.fromkeys(available,0)
                self.inventory_history=[]
            self.inventory_history=[entry for entry in self.inventory_history if entry[0]>=turn-24]
            self.inventory_history.append((turn,dict(obs['market']['inventory']),dict(self.sold_totals)))
            for _,c,n in orders:
                if prices[c]>1:
                    self.sold_totals[c]+=n
        return orders


class ImmediatePolicy(train.InventorySellingPolicy):
    def orders(self, obs, held, total, reserve_wheat, reserve_fert):
        available = {c:max(0,int(held.get(c,0))) for c in train.DELIVERED_PRODUCTS}
        for c, reserve in [('WHEAT',0 if obs['day']==29 else reserve_wheat),('FERTILIZER',reserve_fert)]:
            available[c] = min(available[c],max(0,int(total.get(c,0))-reserve))
        return [['SELL',c,n] for c,n in available.items() if n]


class FastEnvironment:
    """Use the installed interpreter directly; validate against every saved state."""
    def __init__(self, history):
        global LAST_ENV
        from fast_arena import Struct
        self.configuration = Struct(copy.deepcopy(history['configuration']))
        self.info = copy.deepcopy(history.get('info',{}))
        self.done = False
        self.state = [Struct(copy.deepcopy(s)) for s in history['steps'][0]]
        for s in self.state:
            s.observation = Struct(s.observation)
        self.steps = [self.state]
        self.turn = 0
        self.daily = []
        self.transactions = [{},{}]
        self.action_digest = hashlib.sha256()
        LAST_ENV = self

    def _Environment__get_shared_state(self, seat):
        from fast_arena import Struct
        observation = copy.deepcopy(self.state[seat].observation)
        observation['step'] = self.turn
        return Struct(observation=observation)

    def step(self, actions):
        self.action_digest.update(json.dumps(actions,sort_keys=True,separators=(',',':')).encode())
        for s,a in zip(self.state,actions):
            s.action = a
        self.state[0].observation.step = self.turn
        ENGINE['interpreter'](self.state,self)
        self.turn += 1
        self.state[0].observation.step = self.turn
        self.steps = [self.state]
        self.done = all(s.status=='DONE' for s in self.state)
        if self.turn % 24 == 0 or self.done:
            self.daily.append(dict(turn=self.turn,
                money=[f['money'] for f in self.state[0].observation.farms],
                shed=[copy.deepcopy(s.observation.private['shed']) for s in self.state],
                prices=copy.deepcopy(self.state[0].observation.market['prices'])))


def gameplay_parity(history, fast):
    import evaluate_v20_v19_losses as replay
    env = FastEnvironment(history) if fast else replay._environment_from_history(history)
    def core(s):
        value = replay._state_core(s)
        value['observation'].pop('remainingOverageTime',None)
        return value
    for step,expected in enumerate(history['steps']):
        if step:
            env.step(train._recorded_step_actions(history,step))
        for seat in (0,1):
            diff = replay._first_diff(core(expected[seat]),core(env.steps[-1][seat]))
            if diff:
                return dict(exact=False,step=step,seat=seat,mismatch=diff)
    return dict(exact=True,steps=len(history['steps']),ignored_fields=['remainingOverageTime'])


def initialize(checkpoint, fast, policy, experiments, cache_features, policy_source=None):
    global MODEL, ARGS, ENGINE, POLICY, EXPERIMENTS, CURRENT_POLICY
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload['algorithm'] != train.CHECKPOINT_ALGORITHM:
        raise ValueError('Checkpoint algorithm mismatch')
    ARGS = train.build_parser().parse_args([])
    for key, value in payload['args'].items():
        setattr(ARGS, key, value)
    ARGS.current_epsilon = 0.0
    MODEL = train.QuantizedResidualQ(len(train.TASK_FEATURE_NAMES), len(train.GLOBAL_FEATURE_NAMES), ARGS.hidden, payload['quantization'])
    MODEL.load_state_dict(payload['online_state_dict'])
    MODEL.eval()
    POLICY = policy
    EXPERIMENTS = experiments
    if policy_source:
        node=next(n for n in ast.parse(Path(policy_source).read_text(encoding='utf-8')).body
                  if isinstance(n,ast.ClassDef) and n.name=='InventorySellingPolicy')
        scope=dict(vars(train))
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(policy_source),'exec'),scope)
        CURRENT_POLICY=scope['InventorySellingPolicy']
    if cache_features:
        original_loader = train._load_tree_free_executor
        def cached_loader(path,selector):
            module = original_loader(path,selector)
            original_features = module.task_features
            previous_obs = None
            cache = {}
            def task_features(obs,worker,task):
                nonlocal previous_obs
                if obs is not previous_obs:
                    cache.clear();previous_obs=obs
                position,operation,weight,resource=task
                key=(worker,tuple(position),tuple(operation),weight,resource)
                if key not in cache:
                    cache[key]=original_features(obs,worker,task)
                return cache[key]
            module.task_features=task_features
            return module
        train._load_tree_free_executor=cached_loader
        original_norm=train.norm_task
        @functools.lru_cache(maxsize=16384)
        def cached_norm(features):
            return original_norm(features)
        train.norm_task=lambda features:cached_norm(tuple(features))
    if policy == 'immediate':
        train.InventorySellingPolicy = ImmediatePolicy
    if fast:
        sys.path.insert(0,str(train.LOCAL_ARENA))
        from fast_arena import load_engine
        ENGINE = load_engine()
        original_commit = ENGINE['_commit_unit']
        def tracked_commit(op,item,price,farm,private,market,shed_capacity=100):
            ok = original_commit(op,item,price,farm,private,market,shed_capacity)
            if ok and LAST_ENV is not None:
                farms = LAST_ENV.state[0].observation.farms
                seat = 0 if farm is farms[0] else 1
                entry = LAST_ENV.transactions[seat].setdefault(op+':'+item,dict(units=0,money=0))
                entry['units']+=1;entry['money']+=price
            return ok
        ENGINE['_commit_unit'] = tracked_commit
        train._environment_from_history = FastEnvironment


def evaluate(job):
    path, parity, fast, policy = job
    if policy == 'current':
        train.InventorySellingPolicy = CURRENT_POLICY
    elif policy == 'immediate':
        train.InventorySellingPolicy = ImmediatePolicy
    else:
        ExperimentalPolicy.parameters = EXPERIMENTS[policy]
        train.InventorySellingPolicy = ExperimentalPolicy
    started = time.monotonic()
    if parity:
        with train._silence_stderr_during_optional_runtime_init():
            control = gameplay_parity(train._load_history(path),fast)
        if not control.get('exact'):
            return dict(episode=path.stem, policy=policy, ok=False, error='Recorded-action parity failed', control=control)
    with torch.inference_mode():
        row, _ = train.run_static_episode(path, MODEL, torch.device('cpu'), Path(ARGS.base_executor), ARGS, random.Random(1), deterministic=True, collect=False)
    row['elapsed_seconds'] = time.monotonic() - started
    row['policy'] = policy
    row['recorded_action_parity'] = True if parity else None
    if fast:
        row['daily'] = LAST_ENV.daily
        row['transactions'] = LAST_ENV.transactions
        row['actions_sha256'] = LAST_ENV.action_digest.hexdigest()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=train.DEFAULT_OUTPUT_DIR / 'checkpoints/update_0068.pt')
    parser.add_argument('--history-dir', type=Path, default=train.DEFAULT_HISTORY_DIR)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', default='')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--parity', action='store_true')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--policy', choices=['current','immediate'], default='current')
    parser.add_argument('--experiments', type=Path)
    parser.add_argument('--cache-features', action='store_true', help='Memoize pure feature extraction; never change Q arithmetic')
    parser.add_argument('--policy-source', type=Path, help='Load only InventorySellingPolicy from a saved trainer snapshot')
    args = parser.parse_args()
    experiments = json.loads(args.experiments.read_text()) if args.experiments else {}
    policies = list(experiments) if experiments else [args.policy]
    if args.output.exists():
        raise SystemExit(f'Refusing to overwrite {args.output}')
    paths = sorted(args.history_dir.glob('*.json'))
    if args.episodes:
        wanted = set(args.episodes.split(','))
        paths = [p for p in paths if p.stem in wanted]
        if len(paths) != len(wanted):
            raise SystemExit('Some requested episodes are missing')
    if not paths:
        raise SystemExit('No histories')
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha(args.checkpoint),
                    trainer_sha256=sha(Path(train.__file__)),
                    histories={p.name:sha(p) for p in paths}, mode='static recorded opponent; deterministic frozen Q model',
                    parity=args.parity,fast=args.fast,policy=args.policy,
                    evaluator_sha256=sha(Path(__file__)),parity_ignored_fields=['remainingOverageTime'],experiments=experiments,cache_features=args.cache_features)
    if args.policy_source:
        if args.experiments or args.policy!='current':
            raise SystemExit('--policy-source requires --policy current and no --experiments')
        metadata['policy_source']=str(args.policy_source.resolve())
        metadata['policy_source_sha256']=sha(args.policy_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the exact source used for this experiment beside its results.
    args.output.with_suffix('.trainer.py').write_bytes(Path(train.__file__).read_bytes())
    args.output.with_suffix('.evaluator.py').write_bytes(Path(__file__).read_bytes())
    rows = []
    def save():
        valid = [r for r in rows if r['ok']]
        wins = sum(r['margin'] > 0 for r in valid)
        summary = dict(completed=len(rows), games_total=len(paths), valid=len(valid), errors=len(rows)-len(valid),
                       wins=wins, win_rate=wins/len(paths),
                       mean_margin=sum(r['margin'] for r in valid)/len(valid) if valid else None,
                       target_reached=len(policies)==1 and len(rows)==len(paths) and len(valid)==len(paths) and wins/len(paths)>=0.6)
        if len(policies)>1:
            summary['by_policy'] = {}
            for name in policies:
                matches = [r for r in valid if r.get('policy')==name]
                summary['by_policy'][name] = dict(completed=len(matches),wins=sum(r['margin']>0 for r in matches),
                    mean_margin=sum(r['margin'] for r in matches)/len(matches) if matches else None)
            summary['games_total'] = len(paths)*len(policies)
            summary['win_rate'] = wins/summary['games_total']
        args.output.write_text(json.dumps(dict(metadata=metadata, summary=summary, matches=sorted(rows,key=lambda r:r['episode'])),indent=2,default=str)+'\n',encoding='utf-8')
        return summary
    save()
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=initialize, initargs=(args.checkpoint,args.fast,args.policy,experiments,args.cache_features,args.policy_source)) as pool:
        futures = {pool.submit(evaluate,(p,args.parity,args.fast,policy)):(p,policy) for policy in policies for p in paths}
        for future in as_completed(futures):
            path,policy = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = dict(episode=path.stem,policy=policy,ok=False,error=f'{type(exc).__name__}: {exc}')
            rows.append(row)
            summary = save()
            print(f'[{len(rows)}/{len(paths)*len(policies)}] {policy} {path.stem} {row.get("result", "ERROR")} margin={row.get("margin")} wins={summary["wins"]} seconds={row.get("elapsed_seconds")}',flush=True)
    print(json.dumps(save(),indent=2),flush=True)


if __name__ == '__main__':
    main()
