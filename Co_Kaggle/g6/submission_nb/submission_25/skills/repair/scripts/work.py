"""Small cooperative repair workbench, executed only inside the task sandbox.

This is a reliability tool, not a security boundary against malicious Python.
No task IDs, reference patches, network calls, model calls, or host callbacks.
"""
import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

MAX_FILE = 1_000_000
MAX_EDITS = 12
CHECK_SECONDS = 6
HELP = 'ops: search(query,path?), read(path,start?,end?), edit(path,start,end,text), probe(code), test(path,query?), check, diff, status'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def clipped(text, budget):
    """Bound JSON-encoded size, including backslashes and non-ASCII characters."""
    while len(json.dumps(text)) > budget:
        text = text[:max(0, len(text) * 3 // 4)]
    return text


def execute(argv, root, timeout=CHECK_SECONDS, env=None):
    # Kill descendants as well as the immediate process on timeout. Spool noisy
    # output to disk instead of filling the skill process's memory or context.
    with tempfile.TemporaryFile() as out:
        proc = subprocess.Popen(argv, cwd=root, env=env, stdout=out,
                                stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        size = out.tell()
        out.seek(max(0, size - 2400))
        return dict(exit_code=proc.returncode, timed_out=timed_out,
                    output=clipped(out.read().decode('utf-8', errors='replace'), 2400))


def git(root, *args):
    p = subprocess.run(['git', *args], cwd=root, capture_output=True, timeout=5)
    if p.returncode:
        raise ValueError(p.stderr.decode(errors='replace')[:400])
    return p.stdout


def workspace():
    for raw in [*os.environ.get('PYTHONPATH', '').split(os.pathsep), '/workspace', str(Path.cwd())]:
        if raw:
            p = Path(raw).resolve()
            for candidate in (p, *p.parents):
                if (candidate / '.git').is_dir():
                    return candidate
    raise ValueError('Cannot find task checkout')


def safe(root, name):
    p = Path(name)
    if not name or len(name) > 300 or p.is_absolute() or '..' in p.parts or '.git' in p.parts:
        raise ValueError('Use a relative repository path without .. or .git')
    out = root / p
    if any(x.is_symlink() for x in (out, *out.parents)):
        raise ValueError('Symlinks are not supported')
    if not out.resolve().is_relative_to(root):
        raise ValueError('Path escapes checkout')
    return out


def is_test(name):
    p = Path(name)
    return (any(x.lower() in {'tests', 'test', 'testing'} for x in p.parts)
            or p.name.startswith(('test_', 'conftest')) or p.name.endswith('_test.py'))


def protected(name):
    p = Path(name)
    return (is_test(name) or '__pycache__' in p.parts or p.name.startswith('.adk_exec_')
            or p.name in {'setup.py', 'sitecustomize.py', 'usercustomize.py', '_swegemma_stubs.py'})


def contents(path):
    if not path.is_file() or path.stat().st_size > MAX_FILE:
        raise ValueError('Need an existing file smaller than 1 MB')
    return path.read_bytes()


def changed(root):
    return [p for p in git(root, 'diff', 'HEAD', '--name-only', '-z').decode().split('\0') if p]


def revision(root):
    return digest(git(root, 'diff', 'HEAD', '--binary'))


def meaningful_python(data):
    class StripDocs(ast.NodeTransformer):
        def visit_Module(self, node):
            self.generic_visit(node)
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body.pop(0)
            return node
        visit_ClassDef = visit_FunctionDef = visit_AsyncFunctionDef = visit_Module
    return ast.dump(StripDocs().visit(ast.parse(data)), include_attributes=False)


# Probe assertions and repository activity are recorded separately from stdout.
# They prevent an accidental empty/print-only check from being labelled PASS.
PROBE_RUNNER = '''import json, os, sys, traceback
root, code_file, evidence_file = sys.argv[1:]
sys.path[:0] = [root + '/src', root]
code = open(code_file).read()
seen = set()
probe_lines = set()
def trace(frame, event, arg):
    filename = frame.f_code.co_filename
    if event == 'line':
        if filename == '<repair-probe>': probe_lines.add(frame.f_lineno)
        elif filename.startswith(root + os.sep): seen.add(filename[len(root)+1:])
    return trace if filename == '<repair-probe>' or filename.startswith(root + os.sep) else None
sys.settrace(trace)
try:
    exec(compile(code, '<repair-probe>', 'exec'), {'__name__': '__main__'})
finally:
    sys.settrace(None)
    with open(evidence_file, 'w') as f:
        json.dump({'files': sorted(seen), 'probe_lines': sorted(probe_lines)}, f)
'''


class Workbench:
    def __init__(self, root):
        self.root = root.resolve()
        self.store = self.root / '.git' / 's25-workbench'
        self.store.mkdir(exist_ok=True)
        self.state_path = self.store / 'state.json'
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            'reads': {}, 'attempts': 0, 'failed_edits': [], 'guards': [], 'checks': {}, 'last': None}
        self.recover()

    def save(self):
        tmp = self.store / 'state.tmp'
        tmp.write_text(json.dumps(self.state))
        tmp.replace(self.state_path)

    def recover(self):
        journal = self.store / 'pending.json'
        if journal.exists():
            info = json.loads(journal.read_text())
            safe(self.root, info['path']).write_bytes((self.store / 'before.bin').read_bytes())
            journal.unlink()
            self.state['reads'].pop(info['path'], None)
            self.save()

    def files(self):
        return [n for n in git(self.root, 'ls-files', '-z').decode().split('\0') if n]

    def status(self):
        names = self.files()
        return dict(status='OK', help=HELP, edit_attempts=self.state['attempts'],
                    edit_limit=MAX_EDITS, changed=changed(self.root),
                    source_examples=[n for n in names if n.endswith('.py') and not protected(n)][:12],
                    test_examples=[n for n in names if n.endswith('.py') and is_test(n)][:8])

    def search(self, query, path=''):
        if not query or len(query) > 200:
            raise ValueError('Supply a short search query (literal or regex)')
        try:
            pattern = re.compile(query, re.I)
        except re.error:
            pattern = re.compile(re.escape(query), re.I)
        names = self.files()
        if path:
            safe(self.root, path)
            names = [n for n in names if n == path or n.startswith(path.rstrip('/') + '/')]
        # Documentation/examples often precede the package alphabetically and
        # otherwise consume every result slot before implementation is searched.
        auxiliary = {'docs', 'doc', 'examples', 'example', 'benchmarks', 'benchmark', 'scripts', 'tools'}
        names.sort(key=lambda n: (not n.endswith('.py'),
                                 any(p in auxiliary for p in Path(n).parts[:-1]), n))
        result, tests, paths = [], [], []
        for name in names:
            if pattern.search(name):
                paths.append(name)
            if not name.endswith(('.py', '.md', '.toml', '.cfg', '.rst')):
                continue
            try:
                lines = contents(safe(self.root, name)).decode('utf-8').splitlines()
            except (ValueError, UnicodeError):
                continue
            target = tests if is_test(name) else result
            if len(target) >= 7:
                continue
            for i, line in enumerate(lines, 1):
                if pattern.search(line):
                    target.append(clipped(f'{name}:{i}: {line[:150]}', 210))
                    if len(target) >= 7:
                        break
        return dict(status='OK', source_matches=result, test_matches=tests,
                    matching_paths=[clipped(n, 120) for n in paths[:6]], next='Read the actual matching function and callers; narrow query/path if needed.')

    def read(self, path, start=1, end=0):
        data = contents(safe(self.root, path))
        lines = data.decode('utf-8').splitlines()
        end = min(end or start + 69, start + 99, len(lines))
        if start < 1 or start > end:
            raise ValueError(f'Invalid range; file has {len(lines)} lines')
        rendered, used, actual_end = [], 0, start - 1
        for i in range(start - 1, end):
            row = f'{i + 1}: {lines[i]}'
            if used + len(json.dumps(row)) > 3000:
                break
            rendered.append(row)
            used += len(json.dumps(row)) + 2
            actual_end = i + 1
        if actual_end < start:
            raise ValueError('Line too long for a safe numbered edit; select a different range')
        self.state['reads'][path] = dict(hash=digest(data), start=start, end=actual_end)
        self.save()
        return dict(status='OK', path=path, total_lines=len(lines), start=start, end=actual_end,
                    source='\n'.join(rendered), next='edit replaces inclusive whole lines from this read')

    def run_check(self, spec):
        # Preserve the incoming working tree even if a model-written probe (or
        # test) tries to repair source itself. Snapshot only existing differences;
        # unchanged tracked files can be restored from the original HEAD.
        head = git(self.root, 'rev-parse', 'HEAD').decode().strip()
        untracked = set(git(self.root, 'ls-files', '--others', '--exclude-standard', '-z').decode().split('\0')) - {''}
        dirty = set(changed(self.root)) | untracked
        saved = {}
        for name in dirty:
            file = safe(self.root, name)
            saved[name] = (file.read_bytes(), file.stat().st_mode) if file.is_file() else None
        modes = {}
        for row in git(self.root, 'ls-tree', '-r', '-z', head).decode().split('\0'):
            if row:
                meta, name = row.split('\t', 1)
                modes[name] = int(meta.split()[0], 8)
        try:
            result = self.execute_check(spec)
        finally:
            after_untracked = set(git(self.root, 'ls-files', '--others', '--exclude-standard', '-z').decode().split('\0')) - {''}
            touched = []
            for name in sorted(set(changed(self.root)) | dirty | after_untracked):
                file = safe(self.root, name)
                if name in saved:
                    original = saved[name]
                elif name in modes:
                    original = (git(self.root, 'show', head + ':' + name), modes[name])
                else:
                    original = None
                current = (file.read_bytes(), file.stat().st_mode) if file.is_file() else None
                same = current == original if current is None or original is None else (
                    current[0] == original[0] and (current[1] & 0o777) == (original[1] & 0o777))
                if not same:
                    touched.append(name)
                    if original is None:
                        file.unlink(missing_ok=True)
                    else:
                        file.parent.mkdir(parents=True, exist_ok=True)
                        file.write_bytes(original[0])
                        file.chmod(original[1] & 0o777)
            if touched:
                result = dict(status='BLOCKED', reason='Check modified repository files; changes restored',
                              restored_paths=touched[:12])
        return result

    def execute_check(self, spec):
        pythonpath = os.pathsep.join([str(self.root / 'src'), str(self.root), os.environ.get('PYTHONPATH', '')])
        env = dict(os.environ, PYTHONPATH=pythonpath,
                   PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
        with tempfile.TemporaryDirectory(prefix='s25-check-') as folder:
            temp = Path(folder)
            # Never reuse bytecode from before a same-size, same-second edit.
            # This also avoids removing any repository-owned cache files.
            env['PYTHONPYCACHEPREFIX'] = str(temp / 'pycache')
            if spec['op'] == 'probe':
                tree = ast.parse(spec['code'])
                assertions = {n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assert)}
                if not assertions:
                    return dict(status='BLOCKED', reason='Probe needs an assertion of observable behavior')
                (temp / 'probe.txt').write_text(spec['code'])
                (temp / 'runner.py').write_text(PROBE_RUNNER)
                evidence = temp / 'evidence.json'
                result = execute([sys.executable, str(temp / 'runner.py'), str(self.root),
                                  str(temp / 'probe.txt'), str(evidence)], self.root, env=env)
                covered = json.loads(evidence.read_text()) if evidence.exists() else {}
                valid = bool(assertions & set(covered.get('probe_lines', []))) and bool(covered.get('files'))
                result['repository_files'] = covered.get('files', [])[:12]
                result['assertion_executed'] = bool(assertions & set(covered.get('probe_lines', [])))
                result['behavior_failure'] = bool(result['exit_code'] and 'AssertionError' in result['output'] and valid)
            else:
                target = spec['path'].split('::', 1)[0]
                file = safe(self.root, target)
                if not file.is_file() or file.suffix != '.py':
                    return dict(status='BLOCKED', reason='Test must name an existing Python test file/node; use search')
                xml = temp / 'junit.xml'
                argv = [sys.executable, '-m', 'pytest', '-q', spec['path'], '--maxfail=1',
                        '--disable-warnings', '-p', 'no:cacheprovider', '--junitxml=' + str(xml)]
                if spec.get('query'):
                    argv.extend(['-k', spec['query']])
                result = execute(argv, self.root, env=env)
                cases = list(ET.parse(xml).getroot().iter('testcase')) if xml.exists() else []
                executed = [c for c in cases if c.find('skipped') is None and c.find('error') is None]
                valid = bool(executed)
                result['executed_tests'] = len(executed)
                result['behavior_failure'] = any(c.find('failure') is not None for c in executed)
            result['status'] = ('PASS' if result['exit_code'] == 0 and not result['timed_out'] and valid
                                else 'FAIL' if result['behavior_failure'] and not result['timed_out'] else 'BLOCKED')
            return result

    def check(self, spec=None):
        spec = spec or self.state['last']
        if not spec:
            raise ValueError('No saved check; run probe(code) or test(path) first')
        self.state['last'] = spec
        self.save()
        result = self.run_check(spec)
        key = digest(json.dumps(spec, sort_keys=True).encode())
        old = self.state['checks'].get(key, {})
        baseline_failed = old.get('baseline_failed', False) or (
            not changed(self.root) and result['status'] == 'FAIL')
        result['fail_to_pass'] = bool(baseline_failed and changed(self.root) and result['status'] == 'PASS')
        self.state['checks'][key] = dict(spec=spec, revision=revision(self.root),
                                       baseline_failed=baseline_failed, result=result)
        if result['status'] == 'PASS' and spec not in self.state['guards'] and len(self.state['guards']) < 2:
            self.state['guards'].append(spec)
        self.save()
        return result

    def edit(self, path, start, end, text):
        file = safe(self.root, path)
        if file.suffix != '.py' or protected(path):
            raise ValueError('Edit existing implementation .py files only; tests/setup are protected')
        before = contents(file)
        key = digest(before + json.dumps([path, start, end, text, self.state['reads'].get(path)], sort_keys=True).encode())
        if key in self.state['failed_edits']:
            raise ValueError('Identical failed edit blocked. Read the error/source and change your edit, or submit.')
        if self.state['attempts'] >= MAX_EDITS:
            raise ValueError('Edit attempt limit reached. Inspect diff and submit the retained patch.')
        self.state['attempts'] += 1
        self.save()
        try:
            read = self.state['reads'].get(path)
            if not read or read['hash'] != digest(before):
                raise ValueError('Stale/missing read. Read this file again before editing.')
            if not read['start'] <= start <= end <= read['end']:
                raise ValueError('Edit range must be inside the most recent read of this file')
            lines = before.decode('utf-8').splitlines(keepends=True)
            if text and not text.endswith('\n'):
                text += '\n'
            after = (''.join(lines[:start-1]) + text + ''.join(lines[end:])).encode()
            if after == before:
                raise ValueError('No change; current source already equals replacement')
            if len(after) > MAX_FILE:
                raise ValueError('Edited file is too large')
            compile(after, path, 'exec', dont_inherit=True)
            # Durable journal precedes mutation. Normal failures roll back in
            # finally; a killed skill is recovered on the next operation.
            (self.store / 'before.bin').write_bytes(before)
            journal = self.store / 'pending.json'
            journal.write_text(json.dumps(dict(path=path)))
            accepted = False
            try:
                file.write_bytes(after)
                guards = []
                for spec in self.state['guards']:
                    result = self.run_check(spec)
                    guards.append(result)
                    if result['status'] != 'PASS':
                        raise ValueError('Edit breaks a saved passing check; rolled back. ' + json.dumps(result)[:1600])
                accepted = True
            finally:
                if not accepted:
                    file.write_bytes(before)
                journal.unlink()
            self.state['reads'].pop(path, None)
            self.save()
            return dict(status='EDITED', path=path, guards_passed=len(guards),
                        next='Run check to reuse your reproduction, then diff. Read again before another edit.')
        except Exception:
            self.state['failed_edits'].append(key)
            self.save()
            raise

    def diff(self):
        paths = changed(self.root)
        substantive, invalid = [], []
        for path in paths:
            if path.endswith('.py'):
                try:
                    data = contents(safe(self.root, path))
                    baseline = git(self.root, 'show', 'HEAD:' + path)
                    if meaningful_python(data) != meaningful_python(baseline):
                        substantive.append(path)
                except (ValueError, SyntaxError) as exc:
                    invalid.append(f'{path}: {exc}')
        rev = revision(self.root)
        checks = [dict(op=x['spec']['op'], status=x['result']['status'],
                       fail_to_pass=x['result']['fail_to_pass'], current=x['revision'] == rev)
                  for x in self.state['checks'].values()]
        patch = git(self.root, 'diff', 'HEAD', '--no-ext-diff', '--unified=3').decode(errors='replace')
        return dict(status='EMPTY' if not paths else 'INVALID' if invalid else 'PATCH',
                    changed=paths, substantive_python=substantive, invalid=invalid,
                    checks=checks[-6:], patch=clipped(patch, 2600), patch_truncated=len(json.dumps(patch)) > 2600,
                    note='Only current checks describe these bytes. PASS is not semantic proof; docstrings/comments alone are not a behavior fix.')


def main():
    parser = argparse.ArgumentParser(description=HELP)
    parser.add_argument('--op', default='status', choices=['status', 'search', 'read', 'edit', 'probe', 'test', 'check', 'diff'])
    parser.add_argument('--path', default='')
    parser.add_argument('--query', default='')
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    parser.add_argument('--text')
    parser.add_argument('--code', default='')
    args = parser.parse_args()
    try:
        root = workspace()
        lock_path = root / '.git' / 's25-workbench.lock'
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            work = Workbench(root)
            if args.op == 'search': result = work.search(args.query, args.path)
            elif args.op == 'read': result = work.read(args.path, args.start if args.start is not None else 1, args.end or 0)
            elif args.op == 'edit':
                if not args.path or args.start is None or args.end is None or args.text is None:
                    raise ValueError('edit requires path, start, end and text; no implicit deletion or range')
                result = work.edit(args.path, args.start, args.end, args.text)
            elif args.op == 'probe': result = work.check(dict(op='probe', code=args.code))
            elif args.op == 'test': result = work.check(dict(op='test', path=args.path, query=args.query))
            elif args.op == 'check': result = work.check()
            elif args.op == 'diff': result = work.diff()
            else: result = work.status()
    except Exception as exc:
        result = dict(status='REJECTED', reason=f'{type(exc).__name__}: {exc}', help=HELP)
    # Typical responses fit comfortably below the harness's 5,000-char limit.
    # A pathological diagnostic must still return parseable JSON.
    encoded = json.dumps(result)
    if len(encoded) > 4800:
        encoded = json.dumps(dict(status=result.get('status'), output_truncated=True,
                                  detail=clipped(str(result), 4100), next='Request a narrower range/query.'))
        if args.op == 'read':
            work.state['reads'].pop(args.path, None)
            work.save()  # A truncated read must never authorize invisible lines.
    print(encoded)


if __name__ == '__main__':
    main()
