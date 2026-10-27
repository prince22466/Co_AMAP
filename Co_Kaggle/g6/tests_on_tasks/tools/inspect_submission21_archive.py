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

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("archive",type=Path)
    args=p.parse_args()
    inspect(args.archive)
