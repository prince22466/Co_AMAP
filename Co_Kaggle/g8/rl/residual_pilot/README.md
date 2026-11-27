# V23 direct-residual pilot

This simulator-only experiment tests whether bounded direct HVAC adjustments provide useful control authority and whether a small learned selector can exploit it. It does not change or submit anything in Co_AMAP.

## Run

Use Python 3.12 with the repository's [P0 simulator requirements](../p0/requirements-simulator.txt). The experiment was run at `37081740706db4d946b22bcedb4c15923dc5dc5b` using the checked-in simulator and scikit-learn 1.6.1.

From this directory, validate and summarize the committed records without rerunning the simulator:

```sh
python -m unittest -v test_pilot.py
python verify.py
python summarize.py
```

To rerun the sweep, fit the selector from the new training runs, and reproduce validation:

```sh
export G8_PILOT_OUT=/path/to/new/run-directory
python pilot.py train --workers 8
python pilot.py fit
python pilot.py val --workers 8
python verify.py
python summarize.py
```

By default, local outputs go under the ignored `runs/` directory. The committed records under `results/` are never overwritten. Set `G8_REPO` to use a different repository checkout. Training uses 25 paired runs (five weather cases across four training buildings, 1,152 five-minute steps each). Each fixed action changes **only one** channel: supply-air temperature by ±1 K or fan static pressure by ±120 Pa, on AHUs V23 already activates. Commands are clipped to the simulator action range, leave run switches unchanged, and use the same seeds and initial conditions as baseline. This is not a field-safety validation. A single supervisory option is selected, then applied to each active AHU; the pilot does not learn independent setpoints for each AHU.

For learning, four ridge regressions predict the difference in official loss from V23 for the four residuals, using matched three-hour blocks and gateway observations from the baseline block starts. Feature scaling and hyperparameters are selected on training only, with whole-building holdouts in training cross-validation. The labels compare trajectories under sustained interventions, not identical-state counterfactual actions. This is supervised offline policy fitting, **not** a new online RL run. At inference the fitted controller reselects the lowest predicted-loss option every 15 minutes or abstains if the gain is below the chosen margin.

Validation uses five paired cases on building 5, with 1,728 five-minute steps per run. It compares V23, the fitted selector, the best fixed training option, and a prespecified rule (use the colder supply setpoint if comfort mode is on or imminent and the 90th-percentile cooling error is above 0.7 K and exceeds the 90th-percentile heating error by more than 0.3 K). Building 8 was not accessed for scoring, fitting or selection. Scores are mean official simulator loss, lower is better. These six-day scores are not full-episode or leaderboard scores.

The generated `summary.json` contains paired scenario scores, weighted comfort/energy/carbon component deltas, early/later and occupancy-phase slices, selected action counts, and the number of steps with actual command changes compared with V23 at the same observation.

## Verified result

| Building 5, five six-day cases | Mean loss | Change vs V23 |
|---|---:|---:|
| V23 | 0.02020082 | — |
| Occupancy and demand rule | 0.01952506 | −3.35% |
| Fixed −1 K residual | 0.01900267 | −5.93% |
| Fitted selector | 0.01898807 | −6.00% |

The fixed residual and the fitted selector improved all five cases. The fixed residual delivered 98.8% of the fitted selector's absolute gain. The fitted selector was 7.40% better than V23 over days 1–3 and 5.30% better over days 4–6. Improvement was largest in the weighted comfort component, and both energy and carbon losses fell as well. The fixed residual is the better next candidate for a full-episode check because it is much simpler and achieved almost the same result. It was not tuned on validation or tested on building 8.

Verification covered all 45 simulator runs. Validating the unmodified baseline on the first 864 steps reproduced the existing P3 baseline in all five cases. The four pilot tests passed.

## Why this experiment

[P1](../p1/results/run1/history.json) ended at 0.01940303 on the original 864-step validation, 42.5% worse than V23, after its greedy policy collapsed to the Comfort preset. [P2](../p2/results/run1/history.json) accepted three mutations on sampled training batches but none beat V23 on validation. [P3](../p3/results/run1/history.json) improved from 0.01361835 to 0.01359954 (0.14%) on that window. Its entire gain came from Denver and Singapore; extending those two scenarios to six days reversed their average gain in the audit. This pilot tests direct bounded setpoint adjustments rather than another algorithm over the same profile menu.

In the training sweep, the fixed −1 K residual beat V23 in all five cases (mean loss 0.02337989 to 0.02215014, −5.26%). The reverse +1 K intervention increased mean loss by 5.65%; lowering or raising fan pressure in isolation increased mean loss by 0.56% and 1.06%, respectively. These are paired simulator results, not universal HVAC tuning rules.

| Building 5 weather case | V23 | Fixed −1 K | Fitted selector |
|---|---:|---:|---:|
| Atlanta | 0.02973948 | 0.02782243 | 0.02782243 |
| Denver | 0.01783128 | 0.01768045 | 0.01760764 |
| Fairbanks | 0.01080306 | 0.00971983 | 0.00971983 |
| Phoenix | 0.00844749 | 0.00821641 | 0.00821620 |
| Singapore | 0.03418280 | 0.03157423 | 0.03157423 |

The evidence supports promoting the fixed residual to a **candidate baseline**, not claiming a better RL model. Run matched, full 30-day episodes before using it as a training baseline. If the gain holds, learn when to back off or apply an alternative residual per AHU, and compare any learned policy against the fixed −1 K controller as well as V23.

Archived evidence: [training rollouts](results/train.jsonl), [validation rollouts](results/val.jsonl), [fitted selector and training-only CV](results/model.json).
