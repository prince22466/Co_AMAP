"""Run a short assertion check against workspace imports in a fresh process."""
import argparse
import ast
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile


_CHILD_RUNNER = r'''
import ast
import importlib
import json
from pathlib import Path
import sys
import uuid

request = json.loads(Path(sys.argv[1]).read_text())
status = Path(sys.argv[2])
result = {"completed": False, "module_origin": None,
          "assertions_passed": 0, "assertions_failed": 0}
status.write_text(json.dumps(result))

def record(passed):
    result["assertions_passed" if passed else "assertions_failed"] += 1

# Count executed assertions, including failures caught by the reproduction.
# Random globals avoid collisions and single underscores avoid class mangling.
prefix = "_skill_" + uuid.uuid4().hex
record_name, error_name = prefix + "_record", prefix + "_error"

class Instrument(ast.NodeTransformer):
    def visit_Assert(self, node):
        def mark(passed):
            return ast.Expr(ast.Call(ast.Name(record_name, ast.Load()),
                                     [ast.Constant(passed)], []))
        return ast.copy_location(ast.Try(
            body=[node],
            handlers=[ast.ExceptHandler(ast.Name(error_name, ast.Load()), None,
                                       [mark(False), ast.Raise()])],
            orelse=[mark(True)], finalbody=[]), node)

try:
    if request["module"]:
        module = importlib.import_module(request["module"])
        origin = getattr(module, "__file__", None)
        result["module_origin"] = origin
        if not origin or not Path(origin).resolve().is_relative_to(Path(request["root"])):
            result["origin_error"] = True
            raise RuntimeError("requested module did not load from the workspace")
    tree = ast.fix_missing_locations(Instrument().visit(ast.parse(request["code"])))
    namespace = {"__name__": "__main__", record_name: record, error_name: AssertionError}
    exec(compile(tree, "<skill-check>", "exec", optimize=0), namespace)
    result["completed"] = True
finally:
    status.write_text(json.dumps(result))
'''


def check(workspace, code, module=None, timeout=15):
    root = Path(workspace).resolve(strict=True)
    if not root.is_dir() or len(str(root)) > 1024 or not 1 <= timeout <= 20:
        raise ValueError("workspace must exist and fit 1024 characters; timeout must be 1–20 seconds")
    if len(code) > 16000:
        raise ValueError("check exceeds 16000 characters; reduce it to the discriminating case")
    tree = ast.parse(code, filename="<skill-check>")
    assertions = sum(isinstance(node, ast.Assert) for node in ast.walk(tree))
    if module and (len(module) > 300 or not all(part.isidentifier() for part in module.split("."))):
        raise ValueError("module must be a dotted Python module name")
    paths = [str(p) for p in (root / "src", root) if p.is_dir()]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(paths),
               PYTHONDONTWRITEBYTECODE="1", PYTHONOPTIMIZE="0")
    # The child's marker file distinguishes normal completion from sys.exit(0).
    # Runtime results stay in /tmp, outside the captured repository patch.
    with tempfile.TemporaryDirectory(prefix="submission9-check-") as scratch:
        temp = Path(scratch)
        status = temp / "status.json"
        request = temp / "request.json"
        request.write_text(json.dumps({"root": str(root), "module": module, "code": code}))
        script = temp / "check.py"
        script.write_text(_CHILD_RUNNER, encoding="utf-8")
        timed_out = False
        with (temp / "stdout").open("wb") as stdout, (temp / "stderr").open("wb") as stderr:
            process = subprocess.Popen(
                [sys.executable, "-B", str(script), str(request), str(status)], cwd=root, env=env,
                stdout=stdout, stderr=stderr, start_new_session=(os.name == "posix"),
            )
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                process.wait()
        state = json.loads(status.read_text()) if status.exists() else {}

        def tail(path):
            with path.open("rb") as handle:
                size = path.stat().st_size
                handle.seek(max(0, size - 1400))
                return handle.read(1400).decode("utf-8", errors="replace"), size > 1400

        stdout, stdout_cut = tail(temp / "stdout")
        stderr, stderr_cut = tail(temp / "stderr")
        outcome = (
            "timeout" if timed_out else
            "import_origin_error" if state.get("origin_error") else
            "failed" if process.returncode or state.get("assertions_failed") else
            "incomplete" if not state.get("completed") else
            "passed" if state.get("assertions_passed") else
            "no_assertions_executed" if assertions else "exit_zero_without_assertions"
        )
        result = {
            "outcome": outcome, "returncode": process.returncode,
            "assertions_in_source": assertions, "completed": state.get("completed", False),
            "assertions_passed": state.get("assertions_passed", 0),
            "assertions_failed": state.get("assertions_failed", 0),
            "module_origin": state.get("module_origin"), "workspace": str(root),
            "stdout_tail": stdout, "stderr_tail": stderr,
            "output_truncated": stdout_cut or stderr_cut,
            "scope": "Only this check; assertion coverage and hidden-test success are not established.",
        }
        while len(json.dumps(result, ensure_ascii=False)) > 4400:
            key = max(("stdout_tail", "stderr_tail", "module_origin", "workspace"),
                      key=lambda k: len(result[k] or ""))
            result[key] = result[key][max(1, len(result[key]) // 2):]
            result["output_truncated"] = True
        return result



import hashlib
import re


def git(root, *args, ok=(0,)):
    p = subprocess.run(['git', '-C', str(root), *args], capture_output=True, timeout=8)
    if p.returncode not in ok:
        raise ValueError(p.stderr.decode(errors='replace')[-700:])
    return p.stdout


def workspace():
    # ADK changes cwd to a materialized skill directory. The subprocess harness
    # exports the REAL checkout in PYTHONPATH; Docker uses /workspace.
    candidates = [Path(p) for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p]
    candidates += [Path('/workspace'), Path.cwd()]
    for candidate in candidates:
        for path in [candidate, *candidate.parents]:
            if (path / '.git').exists():
                root = Path(git(path, 'rev-parse', '--show-toplevel').decode().strip()).resolve()
                if root.is_dir():
                    return root
    raise ValueError('No Git workspace found in PYTHONPATH, /workspace or cwd. Check sandbox setup.')


def protected(path):
    p = Path(path)
    return (any(part.startswith('.') or part in {'tests', 'test', 'testing', '__pycache__'} for part in p.parts)
            or p.name.startswith(('test_', 'conftest', 'requirements'))
            or p.name.endswith(('_test.py', '.lock', '.pth'))
            or p.name in {'pyproject.toml', 'setup.py', 'setup.cfg', 'tox.ini', 'pytest.ini',
                          'Pipfile', 'Makefile', 'environment.yml', 'environment.yaml',
                          'sitecustomize.py', 'usercustomize.py', '_swegemma_stubs.py'})


def local(root, name, editing=False):
    p = Path(name)
    if not name or p.is_absolute() or '..' in p.parts:
        raise ValueError('Use a relative repository path without traversal.')
    target = root / p
    if any(q.is_symlink() for q in [target, *target.parents] if q.is_relative_to(root)):
        raise ValueError('Symlinks are not supported.')
    if not target.resolve().is_relative_to(root):
        raise ValueError('Path escapes repository.')
    if editing and protected(name):
        raise ValueError('Protected test/configuration path; fix production code instead.')
    return target


def baseline(root):
    try:
        return git(root, 'rev-parse', '--verify', '_swegemma_baseline').decode().strip()
    except ValueError:
        return 'HEAD'


def changed(root):
    raw = git(root, 'diff', '--name-only', '-z', baseline(root))
    raw += git(root, 'ls-files', '--others', '--exclude-standard', '-z')
    return sorted({n.decode() for n in raw.split(b'\0') if n
                   and not n.decode().startswith('.adk_exec_')
                   and '__pycache__' not in Path(n.decode()).parts})


def digest(root):
    h = hashlib.sha256()
    for name in changed(root):
        h.update(name.encode() + b'\0')
        p = root / name
        h.update(p.read_bytes() if p.is_file() and not p.is_symlink() else b'<deleted-or-symlink>')
    return h.hexdigest()


def state_path(root):
    p = Path(git(root, 'rev-parse', '--git-dir').decode().strip())
    return (p if p.is_absolute() else root / p) / 'submission14_workbench.json'


def inspect(root, args):
    if args.query:
        if len(args.query) > 180:
            raise ValueError('Keep query under 180 characters.')
        out = git(root, 'grep', '-n', '-F', '-e', args.query, '--', '.', ':!*.ipynb', ok=(0, 1))
        lines = out.decode(errors='replace').splitlines()
        return {'outcome': 'ok', 'matches': lines[:32], 'truncated': len(lines) > 32}
    if not args.path:
        files = git(root, 'ls-files').decode().splitlines()
        return {'outcome': 'ok', 'files': files[:90], 'truncated': len(files) > 90}
    p = local(root, args.path)
    source = p.read_text()
    lines = source.splitlines()
    start, end = args.start, args.end
    if args.symbol:
        tree = ast.parse(source)
        nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                 and n.name == args.symbol]
        if not nodes:
            raise ValueError('Symbol not found; use query or an explicit line range.')
        if len(nodes) > 1:
            return {'outcome': 'ambiguous', 'locations': [{'start': n.lineno, 'end': n.end_lineno} for n in nodes]}
        node = nodes[0]
        start = min([node.lineno] + [n.lineno for n in getattr(node, 'decorator_list', [])])
        end = node.end_lineno
    if start < 1 or end < start:
        raise ValueError('Invalid line range.')
    end = min(end, start + 119, len(lines))
    return {'outcome': 'ok', 'path': args.path, 'start': start, 'end': end,
            'source': '\n'.join(f'{i+1}: {lines[i]}' for i in range(start-1, end)),
            'total_lines': len(lines)}


def edit(root, args):
    p = local(root, args.path, editing=True)
    if args.action == 'create':
        if p.exists():
            raise ValueError('create only accepts a new file; use replace for existing files.')
        new = args.new
    else:
        source = p.read_text()
        count = source.count(args.old) if args.old else 0
        if count != 1:
            raise ValueError(f'Expected one exact match; found {count}. Inspect the relevant lines and use unique text.')
        new = source.replace(args.old, args.new, 1)
        if new == source:
            raise ValueError('Replacement changes nothing.')
    if len(new.encode()) > 500_000:
        raise ValueError('File exceeds edit size limit.')
    if p.suffix == '.py':
        ast.parse(new, filename=args.path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(new)
    return {'outcome': 'edited', 'path': args.path, 'syntax_checked': p.suffix == '.py'}


def audit(root, state):
    names = changed(root)
    errors = []
    for name in names:
        if protected(name):
            errors.append(f'protected path: {name}')
        p = root / name
        if p.is_symlink():
            errors.append(f'symlink: {name}')
        elif p.suffix == '.py' and p.exists():
            try:
                ast.parse(p.read_bytes(), filename=name)
            except SyntaxError as exc:
                errors.append(f'{name}:{exc.lineno}: {exc.msg}')
    diffcheck = subprocess.run(['git', '-C', str(root), 'diff', '--check', baseline(root)],
                               capture_output=True, timeout=8)
    if diffcheck.returncode:
        errors.append(diffcheck.stdout.decode(errors='replace')[:900])
    if not names:
        errors.append('empty patch')
    evidence = state.get('check', {})
    fresh = bool(evidence) and evidence.get('patch_sha256') == digest(root)
    verified = fresh and evidence.get('outcome') == 'passed'
    diff = git(root, 'diff', '--no-ext-diff', '--unified=2', baseline(root)).decode(errors='replace')
    return {'outcome': 'failed' if errors else 'clean', 'changed_paths': names, 'errors': errors,
            'patch_sha256': digest(root), 'verification': 'passed' if verified else 'unverified',
            'evidence_fresh': fresh, 'last_check': {k:v for k,v in evidence.items() if k not in {'stdout_tail', 'stderr_tail', 'code'}}, 'diff': diff[:2400],
            'diff_truncated': len(diff) > 2400,
            'note': 'New files are listed and hashed; inspect them separately. Local evidence does not establish hidden-test success.'}


def run(args):
    root = workspace()
    path = state_path(root)
    state = json.loads(path.read_text()) if path.exists() else {}
    signature = hashlib.sha256((json.dumps(vars(args), sort_keys=True) + digest(root)).encode()).hexdigest()
    if signature in state.get('failed_calls', []):
        return {'outcome': 'error', 'error': 'Repeated failed call blocked. Change arguments or inspect the relevant source.'}
    # Bound skill calls too: swegemma counts native calls separately in some releases.
    state['calls'] = state.get('calls', 0) + 1
    try:
        if state['calls'] > 32 and args.action != 'audit':
            raise ValueError('Workbench budget exhausted; return your handoff and finalize.')
        if args.action == 'inspect':
            result = inspect(root, args)
        elif args.action in {'replace', 'create'}:
            result = edit(root, args)
        elif args.action == 'check':
            if not args.module or not args.code:
                raise ValueError('check requires module and code with a behavioral assertion.')
            before = digest(root)
            result = check(root, args.code, args.module, min(args.timeout, 15))
            if digest(root) != before:
                result['outcome'] = 'failed'
                result['error'] = 'Check mutated the patch; verification is invalid.'
            state['check'] = {**result, 'patch_sha256': before, 'code': args.code[:4000]}
        else:
            result = audit(root, state)
    except (OSError, ValueError, SyntaxError, subprocess.SubprocessError) as exc:
        result = {'outcome': 'error', 'error': str(exc)[:700]}
    if result['outcome'] in {'error', 'failed', 'timeout', 'import_origin_error'}:
        state.setdefault('failed_calls', []).append(signature)
    path.write_text(json.dumps(state))
    return result


def main():
    parser = argparse.ArgumentParser(description='Bounded repository repair workbench')
    parser.add_argument('--action', choices=['inspect', 'replace', 'create', 'check', 'audit'], default='audit')
    for key in ('path', 'query', 'symbol', 'old', 'new', 'module', 'code'):
        parser.add_argument('--' + key, default='')
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=100)
    parser.add_argument('--timeout', type=int, default=12)
    try:
        result = run(parser.parse_args())
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result = {'outcome': 'error', 'error': str(exc)[:700]}
    # Structured failure is explicit even when ADK's transport status is success.
    # ADK truncates stdout at the transport boundary. Keep valid JSON below it.
    while len(json.dumps(result, ensure_ascii=False)) > 4600:
        fields = [(len(v), k) for k, v in result.items() if isinstance(v, (str, list)) and len(v) > 30]
        if not fields:
            break
        _, key = max(fields)
        result[key] = result[key][:max(1, len(result[key]) // 2)]
        result['output_truncated'] = True
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
