"""Evaluate S26 with the existing official E4B runner, preserving its evidence."""
import argparse
import asyncio
import json
from pathlib import Path
from build_submission26 import ARCHIVE, MODEL, build
from compare_submission24 import PILOT_IDS
from run_submission24_eval import evaluate as evaluate_official, nonnegative_minutes


def objective_pilot_pass(run):
    return (run['resolved'] >= 3 and 'rich_4077' in run['solved_ids']
            and run['empty_patches'] <= 1 and not run['agent_error_ids'])


async def evaluate(args):
    if args.archive is None:
        build()
        args.archive = ARCHIVE
    report = await evaluate_official(args)
    if report['stage'] == 'evaluation_complete' and args.stage == 'pilot':
        report['pilot_ids'] = list(PILOT_IDS)
        report['objective_pilot_pass_each_repeat'] = bool(report['runs']) and all(
            objective_pilot_pass(r) for r in report['runs'])
        report['ready_for_full_s1'] = False  # Must inspect actual real-model traces first.
        report['manual_gate'] = ('Read every model-emitted call/result, verify native edit and explicit '
                                 'nonempty submit_patch, inspect relevant tests, no repeated dispatch loops.')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'submission26_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Optional matched-control archive')
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--snapshots', type=Path, required=True)
    parser.add_argument('--wheels', type=Path)
    parser.add_argument('--model-url', default='http://localhost:8000/v1')
    parser.add_argument('--served-model', default=MODEL)
    parser.add_argument('--sandbox', choices=['docker', 'subprocess'], default='subprocess')
    parser.add_argument('--stage', choices=['pilot', 's1'], default='pilot')
    parser.add_argument('--repeats', type=int, choices=[1, 2, 3], default=2)
    parser.add_argument('--time-limit', type=nonnegative_minutes, default=4)
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(evaluate(args))
    print(json.dumps(report, indent=2))
    if report['stage'] == 'baseline_blocked' or (
        args.stage == 'pilot' and report.get('objective_pilot_pass_each_repeat') is False
    ):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
