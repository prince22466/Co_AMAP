"""Inspect official untrusted S21 evaluation ZIP with standard-library only.

Read ZIP members in memory: no code from an archived task is executed.
Exploration mode is intentionally verbose for one-off diagnostics.
"""
import argparse
import json
import re
from pathlib import Path
from zipfile import ZipFile, BadZipFile

def inspect(archive):
    with ZipFile(archive) as z:
        infos=z.infolist()
        if len(infos)>5000:
            raise ValueError("Unexpectedly many archive members")
        print("ARCHIVE",archive,"FILES",len(infos),
              "COMPRESSED_BYTES",Path(archive).stat().st_size)
        print("ZIP_MEMBER_LIST_BEGIN")
        for info in infos[:500]:
            print(json.dumps({"path":info.filename,"size":info.file_size}))
        print("ZIP_MEMBER_LIST_END")
        json_members=[i for i in infos if i.filename.endswith(".json")]
        print("JSON_MEMBER_COUNT",len(json_members))
        print("ZIP_JSON_SCHEMA_BEGIN")
        for info in json_members[:90]:
            if info.file_size>25_000_000:
                print("SKIP",info.filename,"oversized");continue
            try:
                obj=json.loads(z.read(info).decode("utf-8"))
            except (ValueError,UnicodeError) as exc:
                print("INVALID",info.filename,str(exc)[:130]);continue
            if isinstance(obj,dict):
                shape={k:(list(v)[:15] if isinstance(v,dict) else "list["+str(len(v))+"]" if isinstance(v,list) else repr(v)[:140])
                       for k,v in obj.items()}
            elif isinstance(obj,list):
                shape={"list_length":len(obj),
                       "first_object_keys":list(obj[0]) if obj and isinstance(obj[0],dict) else []}
            else:shape={"type":type(obj).__name__}
            print(json.dumps({"path":info.filename,"shape":shape},ensure_ascii=False))
        print("ZIP_JSON_SCHEMA_END")
        other=[i for i in infos if any(x in i.filename.lower()
               for x in ("readme","summary","report","validation","verification","result","test_output"))
               and not i.filename.endswith(".json") and i.file_size<30000]
        print("ZIP_TEXT_PREVIEW_BEGIN")
        for info in other[:20]:
            text=z.read(info).decode("utf-8",errors="replace")
            print("MEMBER",info.filename,"preview",repr(text[:1400]))
        print("ZIP_TEXT_PREVIEW_END")


def task_evidence(archive, comparison=None):
    """Read validator stdout for each of the 15 tasks. Do not infer baseline."""
    with ZipFile(archive) as z:
        names=z.namelist()
        raw={re.search(r"/raw_results/([^/]+)\\.json$",name).group(1):name
             for name in names if re.search(r"/raw_results/([^/]+)\\.json$",name)}
        if len(raw)!=15:
            raise ValueError("Expected 15 raw_result rows, found "+str(len(raw)))
        old={}
        if comparison:
            with ZipFile(comparison) as previous:
                for name in previous.namelist():
                    m=re.search(r"/raw_results/([^/]+)\\.json$",name)
                    if m:
                        try: old[m.group(1)]=json.loads(previous.read(name))
                        except (ValueError,UnicodeError):pass
        print("TASK_VALIDATION_BEGIN")
        for task in sorted(raw):
            r=json.loads(z.read(raw[task]))
            output=r.get("test_output") or ""
            exc=[]
            for line in output.splitlines():
                if re.search(r"(?:^\\s*E\\s+|(?:ModuleNotFoundError|ImportError|SyntaxError|IndentationError|AttributeError|TypeError|NameError|AssertionError|RuntimeError|ValueError|RecursionError|fixture|[Ee]rror collecting))",line):
                    clean=line.strip()[:260]
                    if clean not in exc:exc.append(clean)
            patch=r.get("agent_patch") or ""
            files=re.findall(r"(?m)^diff --git a/(.*?) b/",patch)
            prior=old.get(task)
            result={
                "id":task,"test_exit_code":r.get("test_exit_code"),
                "resolved":r.get("resolved"),"agent_status":r.get("status"),
                "error_message":r.get("error_message"),
                "patch_chars":len(patch),"changed_files":files,
                "output_head":output[:2200],"output_tail":output[-1450:],
                "exception_examples":exc[:9],
                "patch_first_1800":patch[:1800],
                "previous_run_present":bool(prior),
                "previous_test_exit_code":prior.get("test_exit_code") if prior else None,
                "previous_test_exception_examples":(
                    [line.strip()[:180] for line in (prior.get("test_output") or "").splitlines()
                     if re.search(r"ModuleNotFoundError|ImportError|Error collecting|recursiv|fixture|SyntaxError",line)][:4]
                     if prior else []),
            }
            print(json.dumps(result,ensure_ascii=False))
        print("TASK_VALIDATION_END")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("archive",type=Path)
    p.add_argument("--compare",type=Path)
    p.add_argument("--task-evidence",action="store_true")
    args=p.parse_args()
    if args.task_evidence:task_evidence(args.archive,args.compare)
    else:inspect(args.archive)
