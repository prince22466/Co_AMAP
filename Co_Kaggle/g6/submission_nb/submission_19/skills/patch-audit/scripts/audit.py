"""Bounded read-only structural audit; no task-specific knowledge or behavioral PASS."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess

MAX_FILES=40
MAX_BYTES=1_000_000


def git(root,*args,allow_failure=False):
    r=subprocess.run(['git','-C',str(root),*args],capture_output=True,timeout=8)
    if r.returncode and not allow_failure:raise ValueError(r.stderr.decode(errors='replace')[:300])
    return r


def workspace():
    candidates=[Path(p) for p in os.environ.get('PYTHONPATH','').split(os.pathsep) if p]
    candidates += [Path('/workspace'),Path.cwd()]
    for candidate in candidates:
        for path in [candidate,*candidate.parents]:
            if (path/'.git').exists():
                return Path(git(path,'rev-parse','--show-toplevel').stdout.decode().strip()).resolve()
    raise ValueError('No repository checkout found in PYTHONPATH, /workspace or cwd')


def protected(name):
    p=Path(name)
    return (any(x in {'.git','tests','test','testing'} for x in p.parts)
        or p.name.startswith(('test_','conftest','requirements'))
        or p.name.endswith(('_test.py','.lock','.pth'))
        or p.name in {'pytest.ini','pyproject.toml','setup.cfg','setup.py','tox.ini',
                     'test.sh','coverage.sh','sitecustomize.py','usercustomize.py'})


def definitions(tree):
    # Removing a class statement while leaving its indented body can be syntactically
    # valid: this catches the Submission 18 class-erasure failure beyond AST parsing.
    return {n.name for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef))}


def audit(root):
    root=Path(root).resolve()
    base='_swegemma_baseline' if git(root,'rev-parse','--verify','_swegemma_baseline',allow_failure=True).returncode==0 else 'HEAD'
    names=git(root,'diff','--name-only','-z',base).stdout+git(root,'ls-files','--others','--exclude-standard','-z').stdout
    names=sorted({os.fsdecode(n) for n in names.split(b'\0') if n
        and '__pycache__' not in Path(os.fsdecode(n)).parts and not os.fsdecode(n).startswith('.adk_exec_')})
    blockers=[];removed=[];h=hashlib.sha256()
    if not names:blockers.append('empty patch')
    if len(names)>MAX_FILES:blockers.append('audit limit exceeded: more than 40 changed files')
    for name in names[:MAX_FILES]:
        p=root/name
        h.update(name.encode()+b'\0')
        if protected(name):blockers.append('protected test/config path: '+name)
        if any(q.is_symlink() for q in [p,*p.parents] if q.is_relative_to(root)):
            blockers.append('symlink path: '+name);continue
        if not p.resolve().is_relative_to(root):blockers.append('path escapes checkout: '+name);continue
        if p.exists() and (not p.is_file() or p.stat().st_size>MAX_BYTES):
            blockers.append('unreadable/oversized path: '+name);continue
        data=p.read_bytes() if p.exists() else b''
        h.update(b'present\0'+data if p.exists() else b'deleted\0')
        if p.suffix!='.py':continue
        try:after=ast.parse(data,filename=name) if p.exists() else ast.Module(body=[],type_ignores=[])
        except (SyntaxError,ValueError) as exc:
            blockers.append(f'{name}: {exc}');continue
        before=git(root,'show',f'{base}:{name}',allow_failure=True)
        if before.returncode==0:
            try:lost=definitions(ast.parse(before.stdout))-definitions(after)
            except (SyntaxError,ValueError):lost=set()
            removed.extend(f'{name}:{n}' for n in sorted(lost))
    whitespace=git(root,'diff','--check',base,allow_failure=True)
    if whitespace.returncode:blockers.append('diff whitespace errors: '+whitespace.stdout.decode(errors='replace')[:400])
    return {'status':'blocked' if blockers else ('review_required' if removed else 'structurally_clean'),
        'changed_paths':names[:MAX_FILES], 'blockers':blockers[:12], 'removed_definitions':removed[:12],
        'patch_sha256':h.hexdigest() if len(names)<=MAX_FILES and not blockers else None,
        'behavior_verified':False,
        'next':'Repair blockers; review removed definitions; run a discovered targeted test or actual-output assertion. Re-audit after edits.',
        'limits':'40 files, 1 MB per file. Structural audit only; no test execution, import provenance or semantic correctness guarantee.'}

if __name__=='__main__':
    try:result=audit(workspace())
    except (OSError,ValueError,subprocess.SubprocessError) as exc:result={'status':'error','error':str(exc)[:500],'behavior_verified':False}
    # Emit valid bounded JSON, retaining the non-verification flag on oversized output.
    if len(json.dumps(result))>4500:
        result={'status':'blocked','blockers':['Audit output exceeded limit; inspect a smaller patch'], 'behavior_verified':False}
    print(json.dumps(result))
