"""Optional candidate-only verification. Never certifies a baseline regression.

The model must select a task-relevant probe. This tool confirms that it executed
against the checked-out module and does not silently treat setup errors as passes.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

DRIVER = r"""
import importlib, json, os, pathlib, sys, traceback
root = pathlib.Path.cwd().resolve()
sys.path[:0] = [str(root / 'src'), str(root)]
module, probe, changed = json.loads(sys.argv[1])
origin = None
touched = set()
def trace(frame, event, arg):
    if event == 'line':
        path = pathlib.Path(frame.f_code.co_filename)
        if path.is_absolute() and path.is_relative_to(root):
            name = path.relative_to(root).as_posix()
            if frame.f_lineno in changed.get(name, []):
                touched.add((name, frame.f_lineno))
    return trace
status = 'failed'
details = ''
try:
    if module:
        obj = importlib.import_module(module)
        origin = getattr(obj, '__file__', None)
        if not origin or not pathlib.Path(origin).resolve().is_relative_to(root):
            raise RuntimeError('Import does not come from checkout: ' + str(origin))
    sys.settrace(trace)
    try:
        exec(compile(probe, '<focused-task-probe>', 'exec'), {'__name__': '__main__'})
        status = 'candidate_passed'
    finally:
        sys.settrace(None)
except BaseException:
    details = traceback.format_exc(limit=4)[-1000:]
print('S22_VERIFICATION:' + json.dumps({'status': status, 'origin': origin, 'details': details, 'touched': sorted(touched)[:25]}))
"""

def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], text=True,
                          capture_output=True, timeout=7)

def workspace():
    for p in [Path('/workspace'), Path.cwd(), *[Path(t) for t in os.getenv('PYTHONPATH', '').split(os.pathsep) if t]]:
        for root in [p, *p.parents]:
            if (root / '.git').exists():
                r = git(root, 'rev-parse', '--show-toplevel')
                if r.returncode == 0:
                    return Path(r.stdout.strip()).resolve()
    raise ValueError('Repository checkout missing')

def nontrivial(probe):
    tree = ast.parse(probe)
    assertions = [x for x in ast.walk(tree) if isinstance(x, ast.Assert)]
    return bool(assertions) and all(not (
        isinstance(a.test, ast.Constant) or (
            isinstance(a.test, ast.Compare) and all(isinstance(n, ast.Constant)
                for n in [a.test.left, *a.test.comparators]))
    ) for a in assertions)

def changed_python_lines(root):
    """Line ranges added/modified in the checkout; not semantic test coverage."""
    out = git(root, 'diff', '--unified=0', 'HEAD')
    if out.returncode:
        raise ValueError('Unable to inspect changed lines')
    current = None
    result = {}
    for line in out.stdout.splitlines():
        if line.startswith('+++ b/'):
            candidate = line[6:]
            current = candidate if candidate.endswith('.py') else None
        elif line.startswith('+++ /dev/null'):
            current = None
        elif current and line.startswith('@@ '):
            match = re.search(r'\+(\d+)(?:,(\d+))?\s+@@', line)
            if match:
                start = int(match.group(1))
                count = int(match.group(2) if match.group(2) is not None else '1')
                result.setdefault(current, set()).update(range(start, start + count))
    others = git(root, 'ls-files', '--others', '--exclude-standard')
    if others.returncode:
        raise ValueError('Unable to inspect untracked Python files')
    for name in others.stdout.splitlines():
        if name.endswith('.py') and (root / name).is_file():
            result[name] = set(range(1, len((root / name).read_text().splitlines()) + 1))
    return {name: sorted(lines) for name, lines in result.items() if lines}


def diff_fingerprint(root):
    diff = git(root, 'diff', '--binary', 'HEAD')
    if diff.returncode:
        raise ValueError('git diff HEAD failed')
    untracked = git(root, 'ls-files', '--others', '--exclude-standard')
    if untracked.returncode:
        raise ValueError('Unable to enumerate untracked files')
    lines = sorted(untracked.stdout.splitlines())
    digest = hashlib.sha256(diff.stdout.encode())
    for name in lines:
        f = root / name
        if f.is_file() and not f.is_symlink() and f.resolve().is_relative_to(root):
            digest.update(name.encode() + b'\0' + f.read_bytes())
    return digest.hexdigest(), bool(diff.stdout or lines)

def verify(root, module, probe, timeout=9):
    root = Path(root).resolve()
    if not nontrivial(probe):
        return {'status': 'unverified', 'reason': 'Need a nontrivial task-specific Python assertion'}
    before, has_patch = diff_fingerprint(root)
    changed = changed_python_lines(root)
    if not has_patch:
        return {'status': 'unverified', 'reason': 'No candidate patch is visible'}
    if not changed:
        return {'status': 'unverified', 'reason': 'No changed Python lines; inspect non-Python changes separately'}
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(root / 'src'), str(root)])
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    try:
        done = subprocess.run([sys.executable, '-c', DRIVER, json.dumps([module, probe, changed])],
                              cwd=root, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {'status': 'environment_or_probe_failure', 'reason': 'Probe timed out'}
    after, _ = diff_fingerprint(root)
    if after != before:
        return {'status': 'blocked', 'reason': 'Probe modified the candidate checkout; re-inspect diff'}
    lines = [x.removeprefix('S22_VERIFICATION:') for x in done.stdout.splitlines()
             if x.startswith('S22_VERIFICATION:')]
    try:
        result = json.loads(lines[-1]) if lines else {}
    except (ValueError, IndexError):
        result = {}
    output = done.stdout + '\n' + done.stderr
    passed = (done.returncode == 0 and result.get('status') == 'candidate_passed'
              and not re.search(r'\b(FAILED|ERROR collecting|ERROR at setup)\b', output, re.I))
    touched = result.get('touched') or []
    return {'status': ('candidate_passed' if touched else 'insufficient_evidence')
            if passed else 'environment_or_probe_failure',
            'changed_code_executed': bool(touched), 'executed_changed_lines': touched[:10],
            'origin': result.get('origin'), 'details': result.get('details', '')[-700:],
            'stdout_tail': output[-500:], 'patch_sha256': before, 'candidate_passed': passed and bool(touched),
            'baseline_verified': False, 'behavior_verified': False,
            'next': 'Compare with a baseline failure and inspect the task contract; a passing candidate probe is insufficient.'}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--module', default='')
    p.add_argument('--probe', required=True)
    try:
        a = p.parse_args()
        result = verify(workspace(), a.module, a.probe)
    except Exception as exc:
        result = {'status': 'blocked', 'reason': type(exc).__name__ + ': ' + str(exc)[:400]}
    print(json.dumps(result)[:3400])

if __name__ == '__main__':
    main()
