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
module, probe = json.loads(sys.argv[1])
origin = None
status = 'failed'
details = ''
try:
    if module:
        obj = importlib.import_module(module)
        origin = getattr(obj, '__file__', None)
        if not origin or not pathlib.Path(origin).resolve().is_relative_to(root):
            raise RuntimeError('Import does not come from checkout: ' + str(origin))
    exec(compile(probe, '<focused-task-probe>', 'exec'), {'__name__': '__main__'})
    status = 'candidate_passed'
except BaseException:
    details = traceback.format_exc(limit=4)[-1000:]
print('S22_VERIFICATION:' + json.dumps({'status': status, 'origin': origin, 'details': details}))
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
    if not has_patch:
        return {'status': 'unverified', 'reason': 'No candidate patch is visible'}
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(root / 'src'), str(root)])
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    try:
        done = subprocess.run([sys.executable, '-c', DRIVER, json.dumps([module, probe])],
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
    return {'status': 'candidate_passed' if passed else 'environment_or_probe_failure',
            'origin': result.get('origin'), 'details': result.get('details', '')[-700:],
            'stdout_tail': output[-500:], 'patch_sha256': before, 'candidate_passed': passed,
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
