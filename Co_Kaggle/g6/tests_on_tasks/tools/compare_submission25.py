"""Compare complete S1 runs against S21, S22 and S24; never infer untested wins."""
import argparse
import json
from pathlib import Path
from compare_submission24 import S1_IDS, load_rows, normalize, summarize


def compare(s21, s22, s24, s25):
    runs = {n: normalize(rows) for n, rows in [('s21', s21), ('s22', s22), ('s24', s24), ('s25', s25)]}
    summaries = {n: summarize(list(rows.values())) for n, rows in runs.items()}
    solved = set(summaries['s25']['solved_ids'])
    prior = set(summaries['s21']['solved_ids']) | set(summaries['s22']['solved_ids'])
    paired = {}
    for name in ['s21', 's22', 's24']:
        old = set(summaries[name]['solved_ids'])
        paired[name] = dict(gained=sorted(solved - old), lost=sorted(old - solved),
                            retained=sorted(old & solved), net_gain=len(solved) - len(old))
    target = (len(solved) >= 6 and paired['s24']['net_gain'] >= 5
              and not paired['s24']['lost'] and len(solved & prior) >= min(3, len(prior))
              and summaries['s25']['empty_patches'] <= 2 and not summaries['s25']['agent_error_ids'])
    return dict(summaries=summaries, paired=paired, meets_single_run_target=target,
        target='>=6/15; >=5 net gains over S24; retain all S24 wins and >=3 S21/S22 union wins; <=2 empty patches; no agent errors',
        limitation='Development target, not statistical significance. Compare matching model/runtime/budget manifests; repeat both candidates. Inspect all failures and cost, not only the best seed.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['s21', 's22', 's24', 's25']:
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result = compare(*(load_rows(getattr(args, n)) for n in ['s21', 's22', 's24', 's25']))
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
