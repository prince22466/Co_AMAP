# Reusable production-policy diagnostics

`diagnose.py` observes the official engine and reconstructs the four policy stages. It never changes the supplied agent file. `summarize_diagnostics.py` turns its evidence into a repeatable Markdown report, deployment/lifecycle tables, and accounting metrics.

## Run

From the repository root (PowerShell):

```powershell
python local_arena/static_reply/diagnose.py --agent local_arena/static_reply/v20_3.py --histories local_arena/static_reply/runs/v20_3/20260926T222057_211234Z/histories/111572223.json local_arena/static_reply/runs/v20_3/20260926T222057_211234Z/histories/111574474.json --output local_arena/static_reply/diagnostics/new_run --experiments feed_buffer fert_no_churn feed_and_fert ignore_opponent_capacity no_repeat_capacity deliver_products
python local_arena/static_reply/summarize_diagnostics.py local_arena/static_reply/diagnostics/new_run
python -m unittest discover -s local_arena/static_reply -p test_diagnose.py -v
```

Omit `--experiments` for a baseline-only audit. A new/empty output directory is required so old evidence remains intact. The engine is the already-installed `kaggle_environments`; the reporting code uses only Python's standard library. Agent inference and full state checks take time; each game/intervention is a fresh sequential replay. Engine instrumentation uses temporary global wrappers and must not run concurrently in threads within one process.

The candidate seat comes from `info.static_replay.candidate_seat`. Supply `--seat 0` or `--seat 1` for other recordings. The tool deliberately does not assume that our agent is the loser. An available archived agent hash must match. Every baseline action and every observation/reward/status must also match. Any mismatch raises an error instead of producing a misleading report.

The current adapter supports the `v20_3` production API and a default 10x10 board, 24-turn day, 720-state season, 100-unit shed, 10 market orders and default town cadence/hire costs. Other agent families require an adapter; changing a file path alone is not enough. The source file must provide `production_signals`, `animal_plan`, `crop_plan`, `market_orders`, `unit_actions`, `post_action_market_inventory`, and their associated constants.

## Method and interpretation

1. **Provenance and time alignment.** Record input/agent hashes and engine version. `steps[t]` is the pre-action observation; `steps[t+1].action` is the response. There are 719 executed decisions, not 720. The terminal observation is D29 H23; no further action can liquidate terminal stock.
2. **Decision reconstruction.** Call the exact agent sequentially with fresh globals per game. Retain signals, capacity components, both plans, deployment space, buy deadlines, floor(gap), cash constraints, order priorities, and actions. Exact equality with recorded actions is mandatory.
3. **Event isolation.** Detect new shops with a multiset (duplicates count). Detect opponent producers using tile position/species/planting or placement day (same-species replanting counts). Remove just one new shop/producer from the *same* observation, recompute signals and market orders, holding chosen worker actions fixed. This isolates the event's direct signal-to-market effect, not its possible worker-planning effect. Follow actual matching purchases for 24 and 72 turns and to the next purchase.
4. **Engine truth.** Replay and instrument actual successful market units, their prices, post-worker shed contents, worker effects, end-of-day feeding, animal escapes, drought and overflow. Verify every resulting state against the recording. Cash categories must reconstruct each player's final score exactly.
5. **Selling audit.** Compare actual post-worker available goods with submitted sells, applying the agent's stated wheat/fertilizer reserves. Distinguish unsent unit-turns, unfilled orders, carried stock, destruction, and terminal leftovers. Repeated unsent stock is not additive economic loss. Normal produce and reserved resources have separate interpretations.
6. **Lifecycle and timing.** Match each deployed asset to its harvests and fate. Match successful purchases to successful deployment FIFO to measure delays. All seeds/animals are fungible; FIFO attribution is an accounting convention. A dead animal's acquisition cost is not its counterfactual lifetime profit.
7. **Cash and fertilizer.** Report executed buys and sells together. Fertilizer FIFO distinguishes purchased resale P&L, consumed purchased inputs, and collected output. Do not call gross fertilizer purchases a loss. The FIFO decomposition has an explicit caveat if overflow occurs; engine cash accounting remains exact.
8. **Interventions.** Re-run from the initial state with one documented change and fixed opponent commands. Report both our score and the margin, because the opponent's score can change too. Interventions interact and their improvements cannot be added. These two selected losses do not establish performance across the competition.

## Diagnostic interventions

| Name | Change | Interpretation |
| --- | --- | --- |
| `feed_buffer` | From D1 through D28, insert wheat purchases toward two days of currently placed livestock feed, after sells and before other spending; preserve a 100-coin buffer | Tests the missing short-term feed procurement path; no guarantee that the scheduler feeds every animal |
| `fert_no_churn` | Remove all fertilizer `BUY_PRODUCT` orders | A broad stress test that also removes useful fertilizer, **not** a pure churn fix |
| `feed_and_fert` | Combine the above | Interaction test, not an additive prediction |
| `ignore_opponent_capacity` | Exclude opponent production capacity while preserving visible opponent feed demand | Tests sensitivity to opponent supply; not a recommendation to ignore competition |
| `no_repeat_capacity` | Existing crops contribute their current crop only, without automatic future same-crop replacement | Tests the assumed replanting line; new seeds still use the agent's original stream model |
| `deliver_products` | Force workers with non-reserve produce to return to the shed and deposit it | Tests transport immediacy, including its labor cost; market sell policy remains unchanged |

## Evidence files

Each `<episode>/<variant>/` contains:

- `summary.json`, `cash.csv`, `ledger.csv`: scores and executed accounting, including the opponent.
- `worker_effects.json`: worker action, position, before/after tile, inventory change.
- `losses.csv`, `enddays.csv`: losses and herd/feed snapshots.
- `selling.csv`: shed forecast vs engine stock, reserve, eligible and requested sell units.

Baseline additionally contains `turns.json` and `events.json`. The reporting command adds `derived.json`, `producers.csv`, `lifetimes.csv`, `deployment_lags.csv`, plus root `evidence_report.md` and `derived_metrics.json`.

To inspect one decision without opening a large JSON manually:

```powershell
python -c "import json; from pathlib import Path; t=json.loads(Path('local_arena/static_reply/diagnostics/new_run/111572223/baseline/turns.json').read_text()); print(json.dumps(t[24],indent=2))"
```

## Known boundaries

- Sale eligibility uses actual post-worker living-animal counts; the agent uses pre-worker counts. On an animal-placement turn, those reserve thresholds may differ. Check `worker_effects.json` before classifying reserve mismatches as forecasting bugs.
- A zero signal can mean public future supply covers remaining town demand while our animals still have no accessible feed today. The diagnostic records both the signal and physical stock rather than assuming one implies the other.
- The engine's crop/care mechanics are richer than the agent's capacity approximations. Forecasted capacity is a policy estimate, not guaranteed realized production.
- Individual event probes can miss joint effects when multiple events occur together. Their responses are not additive, and 24/72-turn windows can overlap.
- Unharvested terminal yields and forecast price opportunity are not valued as guaranteed cash. Full interventions are needed for economic impact.
