"""Build S25 and reuse the reviewed official evaluator runner, starting with a pilot.

Uses an already running E4B server. No GPU/weights, tests or harness are modified.
"""
import argparse
import asyncio
import json
from pathlib import Path
from build_submission25 import ARCHIVE, MODEL, build
from compare_submission24 import PILOT_IDS
from run_submission24_eval import evaluate as evaluate_official, nonnegative_minutes


def pilot_ready(run):
    return (run['resolved'] >= 3 and 'rich_4077' in run['solved_ids']
            and run['empty_patches'] <= 1 and not run['agent_error_ids'])


async def evaluate(args):
    # Passing an explicit archive prevents the reused runner from building S24.
    if args.archive is None:
        build()
        args.archive = ARCHIVE
    report = await evaluate_official(args)
    if report['stage'] == 'evaluation_complete':
        report['note'] = 'Use compare_submission25.py for complete S1 runs. Report every repeat.'
        if args.stage == 'pilot':
            report['pilot_ids'] = list(PILOT_IDS)
            report['pilot_target'] = '>=3/5, retain rich_4077, <=1 empty, no agent errors in EVERY repeat'
            report['ready_for_full_s1'] = all(pilot_ready(r) for r in report['runs'])
    (args.output / 'evaluation_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, help='Optional archive for a matched control; otherwise build S25')
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--snapshots', type=Path, required=True)
    p.add_argument('--wheels', type=Path)
    p.add_argument('--model-url', default='http://localhost:8000/v1')
    p.add_argument('--served-model', default=MODEL)
    p.add_argument('--sandbox', choices=['docker', 'subprocess'], default='subprocess')
    p.add_argument('--stage', choices=['pilot', 's1'], default='pilot')
    p.add_argument('--repeats', type=int, choices=[1, 2, 3], default=1)
    p.add_argument('--time-limit', type=nonnegative_minutes, default=0,
                   help='0 matches historical S1; 4 matches the submission timer')
    p.add_argument('--baseline-only', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = asyncio.run(evaluate(args))
    print(json.dumps(result, indent=2))
    if result['stage'] == 'baseline_blocked' or result.get('ready_for_full_s1') is False:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
