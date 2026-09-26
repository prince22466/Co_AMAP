"""Trace the current seller without changing actions; compare saved action hashes."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import evaluate_selling_policy as ev

ROOT = Path(__file__).resolve().parent / 'runs/selling_0068'
TRACE = []
BASE = ev.CURRENT_POLICY


class TracedPolicy(BASE):
    def orders(self, obs, held, total, reserve_wheat, reserve_fert):
        orders = super().orders(obs, held, total, reserve_wheat, reserve_fert)
        day, hour = int(obs['day']), int(obs['hour'])
        turn = day * 24 + hour
        prices = obs['market']['prices']
        chosen = {c: n for _, c, n in orders}
        reference = next(p for t, p in self.price_history if t >= turn - 24)
        cash = obs['farms'][obs['player']]['money']
        stock = sum(max(0, int(n)) for n in held.values())
        for c in ev.train.DELIVERED_PRODUCTS:
            n = max(0, int(held.get(c, 0)))
            reserve = (0 if day == 29 else reserve_wheat) if c == 'WHEAT' else reserve_fert if c == 'FERTILIZER' else 0
            n = min(n, max(0, int(total.get(c, 0)) - reserve))
            if not n:
                continue
            high = max(p[c] for _, p in self.price_history)
            reason = ('final_day' if day == 29 else 'capacity' if stock > 90 else
                      'near_high' if prices[c] >= .95 * high else
                      'rising' if prices[c] > reference[c] else 'flat_or_falling')
            if reason == 'rising' and (day >= 20 or cash < (2500 if day < 16 else 300)):
                reason = 'late_override' if day >= 20 else 'cash_override'
            q = chosen.get(c, 0)
            TRACE.append(dict(turn=turn, product=c, eligible=n, sold=q,
                              action='hold' if q == 0 else 'all' if q == n else 'half',
                              reason=reason, quote=prices[c], cash=cash, stock=stock))
        return orders


def run(episode):
    TRACE.clear()
    row = ev.evaluate((ev.train.DEFAULT_HISTORY_DIR / (episode + '.json'), False, True, 'current'))
    return dict(episode=episode, actions_sha256=row['actions_sha256'], margin=row['margin'], decisions=list(TRACE))


def initialize():
    ev.initialize(ev.train.DEFAULT_OUTPUT_DIR / 'checkpoints/update_0068.pt', True, 'current', {}, True)
    ev.CURRENT_POLICY = TracedPolicy


if __name__ == '__main__':
    output = ROOT / 'three_actions_decision_audit.json'
    if output.exists():
        raise SystemExit('Refusing to overwrite audit')
    expected = json.loads((ROOT / 'three_actions_screen.json').read_text())
    reference = {r['episode']: r for r in expected['matches']}
    rows = []
    with ProcessPoolExecutor(max_workers=4, initializer=initialize) as pool:
        for row in pool.map(run, reference):
            assert row['actions_sha256'] == reference[row['episode']]['actions_sha256']
            assert row['margin'] == reference[row['episode']]['margin']
            rows.append(row)
            print(row['episode'], 'identical actions; traced', len(row['decisions']), flush=True)
    output.write_text(json.dumps(dict(matches=rows), indent=2) + '\n', encoding='utf-8')
