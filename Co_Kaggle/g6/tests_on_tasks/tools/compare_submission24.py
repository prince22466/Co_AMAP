"""Compare complete matched S1 diagnostics; incomplete/duplicate runs fail closed."""
import argparse
import json
from pathlib import Path
import zipfile

S1_IDS = ('fastapi_14258', 'requests_7315', 'requests_7205', 'rich_4077',
          'rich_4076', 'rich_4075', 'rich_3718', 'rich_3934', 'rich_3882',
          'rich_3905', 'rich_3518', 'rich_3454', 'rich_3938', 'rich_3676', 'rich_3052')
PILOT_IDS = ('rich_4077', 'rich_3905', 'rich_3518', 'rich_3454', 'rich_3882')


def load_rows(path):
    path = Path(path)
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith('/notebook_task_results.jsonl')
                     or n == 'notebook_task_results.jsonl']
            if len(names) != 1:
                raise ValueError('Expected exactly one notebook results file in ' + str(path))
            data = z.read(names[0])
    elif path.is_dir():
        names = list(path.rglob('notebook_task_results.jsonl'))
        if not names:
            names = list(path.rglob('task_results.jsonl'))
        if len(names) != 1:
            raise ValueError('Select one run directory, not a collection of runs')
        data = names[0].read_bytes()
    else:
        data = path.read_bytes()
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def normalize(rows, expected=S1_IDS):
    result = {}
    for row in rows:
        name = row.get('instance_id') or row.get('task_id')
        if not name or name in result:
            raise ValueError('Missing/duplicate task ID: ' + str(name))
        if not isinstance(row.get('resolved'), bool):
            raise ValueError('Missing boolean resolved for ' + name)
        if not isinstance(row.get('agent_patch'), str):
            raise ValueError('Missing returned patch for ' + name)
        if row['resolved'] and not row['agent_patch'].strip():
            raise ValueError('Resolved with an empty patch; investigate baseline/environment: ' + name)
        result[name] = row
    if set(result) != set(expected):
        raise ValueError('Task set mismatch: ' + str(sorted(set(result) ^ set(expected))))
    return result


def summarize(rows):
    result = normalize(rows)
    solved = sorted(n for n, r in result.items() if r['resolved'])
    empty = sorted(n for n, r in result.items() if not r['agent_patch'].strip())
    errors = sorted(n for n, r in result.items() if r.get('error') or r.get('error_message'))
    return {'tasks': len(result), 'resolved': len(solved), 'solved_ids': solved,
            'empty_patches': len(empty), 'empty_ids': empty, 'agent_error_ids': errors}


def compare(baseline21, baseline22, candidate):
    summaries = {name: summarize(rows) for name, rows in
                 [('s21', baseline21), ('s22', baseline22), ('s24', candidate)]}
    prior = set(summaries['s21']['solved_ids']) | set(summaries['s22']['solved_ids'])
    solved = set(summaries['s24']['solved_ids'])
    paired = {}
    for name in ['s21', 's22']:
        old = set(summaries[name]['solved_ids'])
        paired[name] = {'gained': sorted(solved - old), 'lost': sorted(old - solved),
                        'retained': sorted(old & solved), 'net_gain': len(solved) - len(old)}
    # Predeclared practical target, not a claim of statistical significance.
    target = (len(solved) >= 6 and all(p['net_gain'] >= 4 for p in paired.values())
              and len(solved & prior) >= min(3, len(prior))
              and summaries['s24']['empty_patches'] <= 2
              and not summaries['s24']['agent_error_ids'])
    return {'summaries': summaries, 'paired': paired,
            'meets_single_run_target': target,
            'target': '>=6/15, >=4 net gains over EACH baseline, retain >=3 of prior union, <=2 empty patches, no agent errors',
            'limitation': 'Single-run development evidence only. Repeat matched runs; inspect final diffs and runtime/package manifests. No significance or hidden-score claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--s21', type=Path, required=True)
    parser.add_argument('--s22', type=Path, required=True)
    parser.add_argument('--s24', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = compare(*(load_rows(getattr(args, n)) for n in ['s21', 's22', 's24']))
    report['inputs'] = {n: str(getattr(args, n)) for n in ['s21', 's22', 's24']}
    text = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
