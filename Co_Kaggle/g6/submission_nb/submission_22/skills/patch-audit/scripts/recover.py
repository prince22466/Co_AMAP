"""Optional bounded recovery from unsuccessful native exact-string edits.

This script is NOT an intercept for native edit_file or submit_patch. It never
guesses a fuzzy replacement and cannot grant successful behavioral verification.
"""
import argparse
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

MAX_FILE = 1_000_000
MAX_CHANGE = 40_000
MAX_FAILURES = 2

def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], capture_output=True,
                          text=True, timeout=6)

def workspace():
    candidates = [Path('/workspace'), Path.cwd()]
    candidates.extend(Path(p) for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p)
    for candidate in candidates:
        for root in (candidate, *candidate.parents):
            if (root / '.git').exists():
                r = git(root, 'rev-parse', '--show-toplevel')
                if r.returncode == 0:
                    return Path(r.stdout.strip()).resolve()
    raise ValueError('Repository workspace not found')

def safe_file(root, name):
    p = Path(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name:
        raise ValueError('Expected a relative checkout path')
    parts = {v.lower() for v in p.parts}
    base = p.name.lower()
    if parts & {'.git', 'tests', 'test', 'testing', '__pycache__'}:
        raise ValueError('Refusing to modify test/harness directories')
    if base.startswith(('test_', 'conftest', 'requirements', '.adk_exec_')):
        raise ValueError('Refusing to modify test/dependency scripts')
    if base in {'pytest.ini', '.pytest.ini', '_swegemma_stubs.py', 'sitecustomize.py'}:
        raise ValueError('Refusing to modify evaluation environment')
    out = root / p
    if out.is_symlink() or not out.resolve().is_relative_to(root):
        raise ValueError('Path escapes checkout or is symlink')
    if not out.is_file() or out.stat().st_size > MAX_FILE:
        raise ValueError('Missing, non-file or oversized target')
    return out

def definitions(src):
    tree = ast.parse(src)
    return {n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}

def evidence(path, old, text):
    lines = text.splitlines()
    fragment = old.strip().splitlines()[0][:120] if old.strip() else ''
    ranked = difflib.get_close_matches(fragment, lines, n=1, cutoff=0.25)
    if ranked:
        row = lines.index(ranked[0])
        lo, hi = max(0, row - 3), min(len(lines), row + 4)
    else:
        lo, hi = 0, min(7, len(lines))
    return {'path': str(path), 'context': [f'{n + 1}: {lines[n]}' for n in range(lo, hi)]}

def failure_state(root):
    # Keep retry state scoped to this Git checkout, not a global /tmp workspace
    # path that could be reused by a later unrelated evaluation task.
    gitdir = git(root, 'rev-parse', '--absolute-git-dir')
    if gitdir.returncode == 0:
        directory = Path(gitdir.stdout.strip())
        if directory.is_dir():
            return directory / 's22_native_edit_attempts.json'
    key = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:20]
    return Path(tempfile.gettempdir()) / ('s22-native-edit-' + key + '.json')

def attempt(root, path, old, new):
    root = Path(root).resolve()
    target = safe_file(root, path)
    if not isinstance(old, str) or not old or not isinstance(new, str):
        raise ValueError('Both nonempty old and new parameters are required')
    if max(len(old.encode()), len(new.encode())) > MAX_CHANGE:
        raise ValueError('Replacement exceeds 40 KB')
    before = target.read_text(encoding='utf-8')
    count = before.count(old)
    key = hashlib.sha256((str(target) + '\0' + old + '\0' + new).encode()).hexdigest()
    state_file = failure_state(root)
    try:
        state = json.loads(state_file.read_text())
    except (OSError, ValueError):
        state = {}
    if count != 1:
        fingerprint = hashlib.sha256(before.encode()).hexdigest()
        record_key = key + ':' + fingerprint
        tries = state.get(record_key, 0) + 1
        state[record_key] = tries
        state_file.write_text(json.dumps(state))
        return {'status': 'retry_blocked' if tries >= MAX_FAILURES else 'anchor_mismatch',
                'matches': count, 'attempts_same_file_and_anchor': tries,
                'next': 'Do not repeat. Read this precise context and choose a different short unique anchor.',
                'nearby': evidence(path, old, before), 'behavior_verified': False}
    after = before.replace(old, new, 1)
    if after == before:
        return {'status': 'blocked', 'reason': 'Replacement makes no change', 'behavior_verified': False}
    baseline_had_syntax_error = False
    if path.endswith('.py'):
        try:
            after_definitions = definitions(after)
        except (SyntaxError, ValueError) as exc:
            return {'status': 'blocked', 'reason': 'Candidate Python syntax: ' + str(exc),
                    'behavior_verified': False}
        try:
            before_definitions = definitions(before)
        except (SyntaxError, ValueError):
            # Native edits can leave a syntactically broken working tree.
            # A structurally valid repair must not be rejected merely
            # because the pre-repair source cannot be parsed.
            before_definitions = set()
            baseline_had_syntax_error = True
        lost = before_definitions - after_definitions
        if lost:
            return {'status': 'blocked', 'reason': 'Removed top-level definitions: ' + ', '.join(sorted(lost)),
                    'behavior_verified': False}
    old_mode = target.stat().st_mode
    prior_visible = git(root, 'status', '--porcelain', '--untracked-files=all', '--', path)
    if prior_visible.returncode:
        return {'status': 'blocked', 'reason': 'Unable to inspect pre-edit git status', 'behavior_verified': False}
    try:
        target.write_text(after, encoding='utf-8')
        target.chmod(old_mode)
        check = git(root, 'diff', '--check')
        if check.returncode:
            target.write_text(before, encoding='utf-8')
            target.chmod(old_mode)
            return {'status': 'blocked', 'reason': 'Whitespace error; edit rolled back',
                    'detail': (check.stdout + check.stderr)[-500:], 'behavior_verified': False}
        visible = git(root, 'status', '--porcelain', '--untracked-files=all', '--', path)
        if visible.returncode or not visible.stdout.strip():
            if not visible.returncode and prior_visible.stdout.strip():
                # Undoing a failed native edit can restore the exact original.
                # Preserve the valid repair but do not advertise a patch.
                return {'status': 'restored_baseline', 'path': path, 'submittable': False,
                        'next': 'Working-tree error repaired. Make the requested functional change before submitting.',
                        'behavior_verified': False, 'submission_verified': False}
            target.write_text(before, encoding='utf-8')
            target.chmod(old_mode)
            return {'status': 'blocked', 'reason': 'File edit not visible to git status; rolled back',
                    'next': 'Choose a tracked or submittable file; ignored files cannot form patches.',
                    'behavior_verified': False}
    except BaseException:
        target.write_text(before, encoding='utf-8')
        target.chmod(old_mode)
        raise
    return {'status': 'edited', 'path': path,
            'file_sha256': hashlib.sha256(after.encode()).hexdigest(),
            'baseline_had_syntax_error': baseline_had_syntax_error,
            'next': 'Run the same focused behavior check; inspect git diff HEAD before submit_patch.',
            'behavior_verified': False, 'submission_verified': False}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--filepath', dest='path', required=True)
    parser.add_argument('--old', '--old_string', dest='old', required=True)
    parser.add_argument('--new', '--new_string', dest='new', required=True)
    try:
        p = parser.parse_args()
        result = attempt(workspace(), p.path, p.old, p.new)
    except Exception as exc:
        result = {'status': 'blocked', 'reason': type(exc).__name__ + ': ' + str(exc)[:320],
                  'behavior_verified': False}
    print(json.dumps(result)[:4000])

if __name__ == '__main__':
    main()
