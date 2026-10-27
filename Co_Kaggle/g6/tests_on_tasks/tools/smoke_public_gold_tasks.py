"""CPU-only external smoke of public task gold fixes, *not* official swegemma verification.

Evaluates identical task-specific regression tests on clean base and with gold:
- red original (nonzero pytest, not timeout) -> green gold (pytest exit 0)
- always from each dataset base_commit; applies test_patch to both versions
- no mutation of task definitions; rejects malformed patches and package shadowing
Expects network, git, pytest and runtime dependencies in CI. Never uses an LLM.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

def paths(patch):
    return list(dict.fromkeys(re.findall(r"(?m)^\+\+\+ b/([^\n]+)", patch)))

def call(argv, cwd, timeout=45, env=None):
    try:
        p = subprocess.run(argv, cwd=cwd, env=env, text=True, capture_output=True,
                           timeout=timeout, check=False)
        return p.returncode, (p.stdout + "\n" + p.stderr)[-1600:]
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT " + " ".join(argv)[:90]

def candidate(t):
    sources = paths(t["patch"])
    tests = paths(t["test_patch"])
    if not sources or not tests: return False
    if any(not s.endswith(".py") for s in tests):return False
    if any(s.startswith("/dev/") or s.startswith("../") for s in sources + tests):return False
    if t["repo"] == "Textualize/rich":
        return (len(sources) <= 2 and len(tests) <= 2 and
                len(t["patch"]) < 8000 and len(t["test_patch"]) < 14000 and
                t["instance_id"] != "rich_3468")  # published no-patch pass
    if t["repo"] == "fastapi/fastapi":
        return t["instance_id"] in {
          "fastapi_14258","fastapi_14786","fastapi_14479",
          "fastapi_14360","fastapi_14458","fastapi_14303",
          "fastapi_13537","fastapi_9425","fastapi_14492",
          "fastapi_13920","fastapi_11194",
        }
    if t["repo"] == "psf/requests":
        return t["instance_id"] in {"requests_7315","requests_6644","requests_7205","requests_7309","requests_7427"}
    return False

def check_task(t, repo, env):
    task=t["instance_id"]
    checkout_code, msg=call(["git","checkout","--quiet","-f","--detach",t["base_commit"]],repo,70)
    if checkout_code:
        fetch_code,_=call(["git","fetch","--quiet","--depth=1","origin",t["base_commit"]],repo,85)
        if fetch_code: return dict(id=task,status="FETCH_FAIL",note=msg)
        checkout_code,msg=call(["git","checkout","--quiet","-f","--detach",t["base_commit"]],repo,70)
        if checkout_code:return dict(id=task,status="CHECKOUT_FAIL",note=msg)
    call(["git","clean","-fdxq"],repo,25)
    test_files=paths(t["test_patch"])
    if any(".." in Path(x).parts for x in test_files):
        return dict(id=task,status="UNSAFE_PATH")
    with tempfile.TemporaryDirectory() as tmp:
        patch=Path(tmp,"tests.patch");patch.write_text(t["test_patch"].rstrip("\n") + "\n")
        check,err=call(["git","apply","--check",str(patch)],repo,15)
        if check:return dict(id=task,status="TEST_PATCH_CONFLICT",note=err)
        check,err=call(["git","apply",str(patch)],repo,15)
        if check:return dict(id=task,status="TEST_PATCH_FAILED",note=err)
        pkg={"Textualize/rich":"rich","fastapi/fastapi":"fastapi","psf/requests":"requests"}[t["repo"]]
        c,out=call([sys.executable,"-c",f"import {pkg}; print({pkg}.__file__)"],repo,18,env)
        if c:return dict(id=task,status="IMPORT_FAIL",note=out)
        package_path=out.strip().splitlines()[-1]
        if not package_path.startswith(str(repo)):
            return dict(id=task,status="WRONG_PACKAGE_IMPORT",note=package_path)
        cmd=[sys.executable,"-m","pytest","-q","--disable-warnings","--maxfail=2","-p","no:cacheprovider",*test_files]
        b,baseout=call(cmd,repo,48,env)
        patch.write_text(t["patch"].rstrip("\n") + "\n")
        check,err=call(["git","apply","--check",str(patch)],repo,15)
        if check:return dict(id=task,status="GOLD_PATCH_CONFLICT",baseline_exit=b,note=err)
        check,err=call(["git","apply",str(patch)],repo,15)
        if check:return dict(id=task,status="GOLD_PATCH_FAILED",baseline_exit=b,note=err)
        g,goldout=call(cmd,repo,48,env)
        status="RED_GREEN" if b in (1,2) and g==0 else "GOLD_GREEN_BASELINE_NOT_RED" if b==0 and g==0 else "GOLD_FAILURE"
        if b==124 or g==124: status="TIMEOUT"
        return dict(id=task,repo=t["repo"],status=status,baseline_exit=b,gold_exit=g,
                    source_files=paths(t["patch"]),test_files=test_files,
                    gold_chars=len(t["patch"]),test_chars=len(t["test_patch"]),
                    baseline_tail=baseout[-330:],gold_tail=goldout[-450:])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("tasks")
    p.add_argument("--output",default="task-gold-smoke.json")
    p.add_argument("--repos",nargs="*",help="Optional repository filter")
    args=p.parse_args()
    all_tasks=[json.loads(l) for l in Path(args.tasks).read_text().splitlines() if l.strip()]
    tasks=[t for t in all_tasks if candidate(t) and
           (not args.repos or t["repo"] in args.repos)]
    print("CANDIDATES",len(tasks),dict(Counter(t["repo"] for t in tasks)),flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        clones={}
        results=[]
        for t in tasks:
            group=t["repo"]
            if group not in clones:
                target=Path(tmp,group.replace("/","_"))
                code,msg=call(["git","clone","--quiet","--filter=blob:none","--no-checkout",
                    "https://github.com/"+group+".git",str(target)],Path(tmp),180)
                if code: 
                    print("CLONE_FAIL",group,msg[-600:],flush=True)
                    clones[group]=None
                else:clones[group]=target
            repo=clones[group]
            if repo is None:
                result=dict(id=t["instance_id"],status="CLONE_FAIL")
            else:
                env={**os.environ,"PYTHONPATH":str(repo / "src" if group == "psf/requests" else repo),
                     "PYTHONDONTWRITEBYTECODE":"1",
                     "PYTEST_DISABLE_PLUGIN_AUTOLOAD":"1","PIP_DISABLE_PIP_VERSION_CHECK":"1"}
                result=check_task(t,repo,env)
            results.append(result)
            print("RESULT",json.dumps(result,ensure_ascii=False),flush=True)
        Path(args.output).write_text(json.dumps(results,indent=2,ensure_ascii=False)+"\n")
        print("TOTALS",json.dumps(dict(Counter(r["status"] for r in results))),flush=True)

if __name__=="__main__":
    main()
