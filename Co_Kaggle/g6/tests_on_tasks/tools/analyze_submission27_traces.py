"""Observe repeated S26/S27 *printed* actions; never alter evaluator or patches.

Text printouts omit call arguments/line ranges. Same-path reads and edit-failure
streaks here are PROXIES, not proof of identical request arguments.
"""
import argparse
import json
from pathlib import Path
import re

TASK = re.compile(r'^\[candidate (\d+)/(\d+)\] Evaluating (\S+)')
RESULT = re.compile(r'-> resolved=(True|False), patch_chars=(\d+), tools=(\d+)')
ACTION = re.compile(r'^\s*\[\s*[\d.]+s\]\s*(?:📖 read_file|✏️ edit_file|🔧 run_command|🤖 submit_patch)\b')
PATH = re.compile(r'\(([^)]+)\)')


def analyze(text):
    tasks, current, last = [], None, None
    for line in text.splitlines():
        found = TASK.match(line)
        if found:
            current = {'index': int(found[1]), 'task_id': found[3], 'reads': 0,
                'edits': 0, 'failed_edits': 0, 'commands': 0, 'submits': 0,
                'same_path_read_streaks': 0, 'same_path_failed_edit_streaks': 0,
                'resolved': None, 'patch_chars': None, 'charged_calls': None}
            tasks.append(current)
            last = None
            continue
        if current is None:
            continue
        result = RESULT.search(line)
        if result:
            current['resolved'] = result[1] == 'True'
            current['patch_chars'] = int(result[2]); current['charged_calls'] = int(result[3])
            last = None
            continue
        # Only timestamped tool calls: printed final detail cards can duplicate them.
        if ACTION.match(line):
            kind = ('read' if 'read_file' in line else 'edit' if 'edit_file' in line
                    else 'command' if 'run_command' in line else 'submit')
            path = PATH.search(line)
            file = path[1] if path and kind in ('read', 'edit') else None
            if kind == 'read':
                current['reads'] += 1
                if last and last['kind'] == 'read' and last['path'] == file:
                    current['same_path_read_streaks'] += 1
            elif kind == 'edit':
                current['edits'] += 1
            elif kind == 'command':
                current['commands'] += 1
            else:
                current['submits'] += 1
            # Keep the previous failed edit for failed-edit streak analysis.
            prior = last
            last = {'kind': kind, 'path': file, 'failed': False, 'prior': prior}
        elif last and last['kind'] == 'edit' and ('❌' in line or 'error_type' in line):
            if not last['failed']:
                last['failed'] = True
                current['failed_edits'] += 1
                prior = last['prior']
                if prior and prior['kind'] == 'edit' and prior['failed'] and prior['path'] == last['path']:
                    current['same_path_failed_edit_streaks'] += 1
    known = [x for x in tasks if x['resolved'] is not None]
    return {'scope': 'Printed trace proxy analysis, NOT identical-argument detection',
            'tasks_seen': len(tasks), 'tasks_with_result': len(known),
            'resolved': sum(x['resolved'] for x in known),
            'empty_patches': sum(x['patch_chars'] == 0 for x in known),
            'charged_call_exhaustions': sum(x['charged_calls'] == 40 for x in known),
            'failed_edits_observed': sum(x['failed_edits'] for x in tasks),
            'same_path_read_streaks': sum(x['same_path_read_streaks'] for x in tasks),
            'same_path_failed_edit_streaks': sum(x['same_path_failed_edit_streaks'] for x in tasks),
            'tasks': tasks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('printout', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = analyze(args.printout.read_text(encoding='utf-8'))
    output = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end='')


if __name__ == '__main__':
    main()
