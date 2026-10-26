"""CPU baseline diagnostic (executes imports and tests, never edits test configuration). Run on a clean task checkout with its test environment."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def preflight(root,module,test,python=sys.executable,timeout=120):
    root=Path(root).resolve()
    status=subprocess.run(['git','-C',str(root),'status','--porcelain'],capture_output=True,text=True,timeout=10)
    if status.returncode or status.stdout.strip():raise ValueError('Use an initialized, clean baseline checkout')
    # Accept one real test file/node, never a shell command or unrestricted option.
    filename=test.split('::',1)[0];path=(root/filename).resolve()
    if not path.is_relative_to(root) or not path.is_file() or filename.startswith('-'):
        raise ValueError('Discover an existing test file/node inside the checkout first')
    if not module or not all(p.isidentifier() for p in module.split('.')):raise ValueError('Expected dotted module name')
    env={**os.environ,'PYTHONPATH':str(root/'src')+os.pathsep+str(root),'PYTHONDONTWRITEBYTECODE':'1'}
    probe='import importlib,json; m=importlib.import_module('+repr(module)+'); print(json.dumps({"origin":getattr(m,"__file__",None)}))'
    imported=subprocess.run([python,'-c',probe],cwd=root,env=env,capture_output=True,text=True,timeout=timeout)
    report={'module':module,'test':test,'import_exit_code':imported.returncode,'import_stderr':imported.stderr[-2500:],
            'local_import':False,'baseline_passed':False,'ready':False}
    if imported.returncode:return report
    try:origin=json.loads(imported.stdout.splitlines()[-1])['origin']
    except (ValueError,KeyError,IndexError):return report
    report['origin']=origin
    report['local_import']=bool(origin) and Path(origin).resolve().is_relative_to(root)
    if not report['local_import']:return report
    test_run=subprocess.run([python,'-m','pytest',test,'-q','--maxfail=1','-p','no:cacheprovider'],cwd=root,env=env,
                            capture_output=True,text=True,timeout=timeout)
    report.update(test_exit_code=test_run.returncode,test_stdout=test_run.stdout[-4500:],test_stderr=test_run.stderr[-1500:],
                  baseline_passed=test_run.returncode==0,ready=test_run.returncode==0)
    # A known failing baseline can be legitimate; inspect it rather than skipping it.
    report['note']='ready means this selected baseline test passes with local imports, not that task fixtures or all dependencies are validated. Baseline failures require classification; do not drop tasks from the denominator.'
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',required=True);p.add_argument('--module',required=True)
    p.add_argument('--test',required=True);p.add_argument('--python',default=sys.executable);a=p.parse_args()
    try:r=preflight(a.workspace,a.module,a.test,a.python)
    except (ValueError,OSError,subprocess.SubprocessError) as e:r={'ready':False,'error':str(e)}
    print(json.dumps(r,indent=2))
    if not r['ready']:raise SystemExit(1)
