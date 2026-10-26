"""Cooperative reliability gate; not a security sandbox against malicious probes."""
import argparse
import ast
import hashlib
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

MAX_FILE = 1_000_000
MAX_WRITE = 100_000
MAX_EDITS = 12
TIMEOUT = 5  # two probes + optional two focused tests fit the 30s skill timeout


def run(argv, cwd, timeout=TIMEOUT, env=None):
    p = subprocess.run(argv, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout)
    return p


def git(root, *args):
    p = run(['git', *args], root)
    if p.returncode:
        raise ValueError(p.stderr[:700])
    return p.stdout


def workspace():
    for raw in [*os.environ.get('PYTHONPATH', '').split(os.pathsep), '/workspace', str(Path.cwd())]:
        if not raw:
            continue
        p = Path(raw).resolve()
        for candidate in (p, *p.parents):
            if (candidate / '.git').exists():
                return candidate
    raise ValueError('Cannot locate repository workspace')


def protected(path):
    p = Path(path)
    return (any(x in {'.git', 'tests', 'test', 'testing', '__pycache__'} for x in p.parts)
            or p.name.startswith(('test_', 'requirements', 'conftest', '.adk_exec_'))
            or p.name.endswith(('_test.py', '.lock', '.pth'))
            or p.name in {'pytest.ini', 'pyproject.toml', 'setup.cfg', 'setup.py', 'tox.ini',
                          'test.sh', 'coverage.sh', 'sitecustomize.py', 'usercustomize.py'})


def safe(root, name):
    p = Path(name)
    if not name or p.is_absolute() or '..' in p.parts or '.git' in p.parts:
        raise ValueError('Use a relative repository file path, without .. or .git')
    out = root / p
    if any(x.is_symlink() for x in [out, *out.parents] if x != root.parent):
        raise ValueError('Symlinks are not supported')
    if not out.resolve().is_relative_to(root.resolve()):
        raise ValueError('Path escapes checkout')
    return out


def clone(root, dest, baseline):
    git(root, 'clone', '--quiet', '--shared', '--no-checkout', str(root), str(dest))
    git(dest, '-c', 'core.hooksPath=/dev/null', 'checkout', '--quiet', '--detach', baseline)


def changes(root):
    # Include ignored new files too, rather than silently losing a candidate.
    names = set(git(root, 'diff', '--name-only', 'HEAD').splitlines())
    names.update(git(root, 'ls-files', '--others').splitlines())
    return sorted(n for n in names if '__pycache__' not in Path(n).parts and not Path(n).name.startswith('.adk_exec_'))


def digest(root):
    h = hashlib.sha256()
    for n in changes(root):
        p = safe(root, n)
        h.update(n.encode() + b'\0' + (p.read_bytes() if p.is_file() else b'<deleted>') + b'\0')
    return h.hexdigest()


def definitions(tree):
    return {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def audit(root):
    names = changes(root)
    if len(names) > 20:
        raise ValueError('More than 20 changed files')
    added = {}
    for n in names:
        if protected(n):
            raise ValueError('Protected test/config path: ' + n)
        p = safe(root, n)
        if not p.is_file() or p.stat().st_size > MAX_FILE:
            raise ValueError('Deletion or oversized file: ' + n)
        text = p.read_text()
        if n.endswith('.py'):
            tree = ast.parse(text, filename=n)
            base = run(['git', 'show', 'HEAD:' + n], root)
            if base.returncode == 0:
                removed = definitions(ast.parse(base.stdout)) - definitions(tree)
                if removed:
                    raise ValueError('Removed top-level definitions: ' + ', '.join(sorted(removed)))
                diff = git(root, 'diff', '--unified=0', 'HEAD', '--', n)
                lines = set()
                for start, count in re.findall(r'^@@ .* \+(\d+)(?:,(\d+))? @@', diff, re.M):
                    lines.update(range(int(start), int(start) + int(count or 1)))
            else:
                lines = set(range(1, len(text.splitlines()) + 1))
            added[n] = lines
    p = run(['git', 'diff', '--check'], root)
    if p.returncode:
        raise ValueError('Whitespace errors: ' + p.stdout[:500])
    return names, added


# Runs in a disposable checkout. Setup errors are distinguished from task assertions.
DRIVER = r'''
import contextlib, importlib, io, json, os, pathlib, sys, traceback
root=pathlib.Path.cwd().resolve()
sys.path[:0]=[str(root/'src'),str(root)]
config=json.loads(pathlib.Path(sys.argv[1]).read_text())
seen=set(); output=io.StringIO(); status='setup_error'; detail=''; origin=None
try:
    with contextlib.redirect_stdout(output),contextlib.redirect_stderr(output):
        if config['module']:
            mod=importlib.import_module(config['module'])
            origin=getattr(mod,'__file__',None)
            if not origin or not pathlib.Path(origin).resolve().is_relative_to(root):
                raise RuntimeError('Imported module is outside checkout: '+str(origin))
        def trace(frame,event,arg):
            if event=='line':
                p=pathlib.Path(frame.f_code.co_filename)
                if p.is_absolute() and p.is_relative_to(root):
                    seen.add((str(p.relative_to(root)),frame.f_lineno))
            return trace
        sys.settrace(trace)
        try:
            exec(compile(config['probe'],'<task-probe>','exec'),{'__name__':'__main__'})
            status='pass'
        except AssertionError:
            status='assertion_failed'; detail=traceback.format_exc(limit=5)
        finally:
            sys.settrace(None)
except BaseException:
    detail=traceback.format_exc(limit=5)
print(json.dumps(dict(status=status,detail=detail[-1500:],output=output.getvalue()[-1500:],origin=origin,executed=sorted(seen))))
'''


class Gate:
    def __init__(self, root):
        self.root = Path(root).resolve()
        key = hashlib.sha256(str(self.root).encode()).hexdigest()[:20]
        self.home = Path(tempfile.gettempdir()) / ('submission20-' + key)
        self.statefile = self.home / 'state.json'
        self.stage = self.home / 'candidate'
        if not self.statefile.exists():
            if changes(self.root):
                raise ValueError('Initial live workspace is dirty; refuse to overwrite it')
            self.home.mkdir(exist_ok=True, mode=0o700)
            baseline = git(self.root, 'rev-parse', 'HEAD').strip()
            if self.stage.exists():
                shutil.rmtree(self.stage)
            clone(self.root, self.stage, baseline)
            self.state = dict(baseline=baseline, edits=0, failures={}, live_hash=digest(self.root),
                              published_paths=[], check=None)
            self.save()
        self.state = json.loads(self.statefile.read_text())
        if git(self.root, 'rev-parse', 'HEAD').strip() != self.state['baseline']:
            raise ValueError('Workspace baseline changed; refuse stale state')

    def save(self):
        temp = self.statefile.with_suffix('.tmp')
        temp.write_text(json.dumps(self.state))
        temp.replace(self.statefile)

    def read(self, path, start=1):
        p = safe(self.stage, path)
        lines = p.read_text().splitlines()
        start = max(1, int(start))
        return dict(status='read', path=path, total_lines=len(lines),
                    content='\n'.join(f'{i+1}: {x}' for i,x in enumerate(lines) if start-1 <= i < start+79)[:9000])

    def edit(self, path, old=None, new=None, content=None):
        p = safe(self.stage, path)
        if protected(path):
            raise ValueError('Protected test/config path: ' + path)
        if self.state['edits'] >= MAX_EDITS:
            raise ValueError('12 edit attempts exhausted; check/publish remain available')
        key = hashlib.sha256(json.dumps([path,old,new,content]).encode()).hexdigest()
        if self.state['failures'].get(key, 0) >= 2:
            return dict(status='retry_blocked', next='Change the anchor or write a complete small file.',
                        context=self.read(path) if p.is_file() else None)
        self.state['edits'] += 1
        before = p.read_bytes() if p.exists() else None
        full_write = content is not None
        try:
            if content is None:
                if old is None or new is None or not old:
                    raise ValueError('edit requires path, nonempty old, and new; no probe is required')
                text = p.read_text()
                if text.count(old) != 1:
                    raise ValueError('old must match exactly once; read the candidate and use a unique anchor')
                content = text.replace(old, new, 1)
            if len(content.encode()) > (MAX_WRITE if full_write else MAX_FILE):
                raise ValueError('Full writes limited to 100 KB; focused edits support files up to 1 MB')
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            audit(self.stage)
            self.state['verified_hash'] = None
            return dict(status='staged', path=path, live_workspace_changed=False)
        except Exception as exc:
            if before is None:
                p.unlink(missing_ok=True)
            else:
                p.write_bytes(before)
            self.state['failures'][key] = self.state['failures'].get(key, 0) + 1
            return dict(status='edit_rejected', error=str(exc), rolled_back=True,
                        next='Read candidate context; correct the edit, not the tests.')
        finally:
            self.save()

    def verify(self, spec):
        names, added = audit(self.stage)
        if not names:
            return dict(status='blocked', reason='empty_candidate')
        probe = spec.get('probe', '')
        if not probe or len(probe) > 12000:
            raise ValueError('check requires a Python probe of at most 12000 characters')
        tree = ast.parse(probe)
        if not any(isinstance(n, ast.Assert) for n in ast.walk(tree)):
            raise ValueError('Probe must assert actual behavior, not just print a result')
        module = spec.get('module', '')
        if added and not re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*', module):
            raise ValueError('Changed Python requires module for local-import provenance')
        test = spec.get('test', '')
        if test:
            testpath = test.split('::', 1)[0]
            safe(self.stage, testpath)
            known = set(git(self.stage, 'ls-files').splitlines())
            if testpath not in known or not Path(testpath).name.startswith('test_') or not testpath.endswith('.py'):
                raise ValueError('Regression test must be a discovered, tracked test_*.py path/node')
        reports = {}
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        for label in ['baseline', 'candidate']:
            with tempfile.TemporaryDirectory(prefix='s20-check-') as temp:
                dest = Path(temp) / 'repo'
                clone(self.root, dest, self.state['baseline'])
                if label == 'candidate':
                    for n in names:
                        target = safe(dest, n)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(safe(self.stage, n), target)
                env['PYTHONPATH'] = os.pathsep.join([str(dest/'src'),str(dest)])
                cfg = Path(temp) / 'probe.json'
                cfg.write_text(json.dumps(dict(probe=probe,module=module)))
                before = digest(dest)
                result = run([sys.executable, '-c', DRIVER, str(cfg)], dest, env=env)
                try:
                    report = json.loads(result.stdout)
                except ValueError:
                    report = dict(status='setup_error',detail=(result.stderr+result.stdout)[-1500:])
                if result.returncode or digest(dest) != before:
                    report.update(status='setup_error', detail='Probe exited abnormally or changed checkout files')
                if re.search(r'\b(FAILED|FAILURE|Traceback)\b', report.get('output',''), re.I):
                    report.update(status='reported_failure')
                if test:
                    t = run([sys.executable,'-m','pytest',test,'-q','--maxfail=1','-p','no:cacheprovider'],dest,env=env)
                    report['regression'] = dict(exit_code=t.returncode, output=(t.stdout+t.stderr)[-2000:])
                    if t.returncode or digest(dest) != before:
                        report['status'] = 'regression_or_environment_failure'
                reports[label] = report
        base, candidate = reports['baseline'], reports['candidate']
        touched = any(line in added.get(path, set()) for path,line in candidate.get('executed',[]))
        passed = (base['status']=='assertion_failed' and candidate['status']=='pass'
                  and (not added or touched))
        # Keep output bounded; the detailed execution list is unnecessary for the model.
        for report in reports.values():
            report.pop('executed', None)
        return dict(status='verified' if passed else 'blocked', reports=reports,
                    changed_python_executed=touched, candidate_hash=digest(self.stage),
                    meaning='Behavioral evidence for this probe, not proof of complete task correctness')

    def check(self, spec=None, publish=False):
        if spec is not None:
            self.state['check'] = spec
            self.save()
        spec = self.state.get('check')
        if not spec:
            return dict(status='blocked', reason='No behavioral check configured; call check(module,probe)')
        if digest(self.root) != self.state['live_hash']:
            return dict(status='blocked', reason='Live workspace changed outside the gate')
        result = self.verify(spec)
        if result['status'] != 'verified' or not publish:
            return result
        # Verification ran on copies; candidate must still match the exact checked bytes.
        if digest(self.stage) != result['candidate_hash']:
            return dict(status='blocked', reason='Candidate changed during verification')
        names, _ = audit(self.stage)
        affected = sorted(set(names) | set(self.state['published_paths']))
        backup = {n: (safe(self.root,n).read_bytes() if safe(self.root,n).exists() else None) for n in affected}
        try:
            for n in affected:
                target = safe(self.root,n)
                if n in names:
                    data = safe(self.stage,n).read_bytes()
                else:
                    original = run(['git','show',self.state['baseline']+':'+n],self.root)
                    data = original.stdout.encode() if original.returncode == 0 else None
                if data is None:
                    target.unlink(missing_ok=True)
                else:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(data)
            self.state['live_hash'] = digest(self.root)
            self.state['published_paths'] = names
            self.save()
        except Exception:
            for n,data in backup.items():
                target=safe(self.root,n)
                if data is None: target.unlink(missing_ok=True)
                else: target.write_bytes(data)
            raise
        return dict(status='published',paths=names,verified_hash=result['candidate_hash'],
                    next='Call native submit_patch now; stop after submitting.')

    def dispatch(self, args):
        op=args.get('op','status')
        if op=='read': return self.read(args.get('path',''),args.get('start',1))
        if op=='search':
            text=args.get('text','')
            if not text: raise ValueError('search requires literal text')
            p=run(['git','grep','-n','-I','-F','--',text],self.stage)
            return dict(status='search',matches=p.stdout[:9000],next='Read a focused range near a match')
        if op=='edit': return self.edit(args.get('path',''),old=args.get('old'),new=args.get('new'))
        if op=='write':
            if 'content' not in args: raise ValueError('write requires path and complete content')
            return self.edit(args.get('path',''),content=args['content'])
        if op=='check': return self.check({k:args.get(k,'') for k in ['module','probe','test']})
        if op=='publish': return self.check(publish=True)
        if op!='status': raise ValueError('Unknown operation; use status/read/search/edit/write/check/publish')
        return dict(status='ready',edits_remaining=MAX_EDITS-self.state['edits'],
                    candidate_paths=changes(self.stage),published_paths=self.state['published_paths'],
                    examples=[{'op':'read','path':'relative/file.py','start':'1'},
                              {'op':'edit','path':'relative/file.py','old':'exact text','new':'replacement'},
                              {'op':'check','module':'package','probe':'import package\nassert package.actual_call() == expected'},
                              {'op':'publish'}])


def main():
    parser=argparse.ArgumentParser(exit_on_error=False)
    for field in ['op','path','start','text','old','new','content','module','probe','test']:
        aliases={'path':'filepath','old':'old_string','new':'new_string'}
        flags=['--'+field]+(['--'+aliases[field]] if field in aliases else [])
        parser.add_argument(*flags,dest=field)
    try:
        parsed,unknown=parser.parse_known_args()
        if unknown: raise ValueError('Unknown arguments: '+str(unknown)+'. Use op/path/old/new/content/module/probe/test.')
        args={k:v for k,v in vars(parsed).items() if v is not None}
        root=workspace()
        key=hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:20]
        lock_path=Path(tempfile.gettempdir())/('submission20-'+key+'.lock')
        with lock_path.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            result=Gate(root).dispatch(args)
    except Exception as exc:
        result=dict(status='blocked',error=type(exc).__name__+': '+str(exc))
    print(json.dumps(result))


if __name__=='__main__': main()
