"""Materialize fixed public-task quick sets from audited instance IDs.

Usage:
 python materialize_quick_task_set.py tasks.jsonl quick_15_ids.json /tmp/quick15.jsonl
No issue fixes, test patches, or evaluator behavior are rewritten.
"""
import argparse
import json
from pathlib import Path


def materialize(tasks_path, ids_path, output_path):
    ids=json.loads(Path(ids_path).read_text(encoding="utf-8"))
    if isinstance(ids,dict):
        ids=ids.get("task_ids")
    if not isinstance(ids,list) or len(ids) not in (15,30):
        raise ValueError("Expected an array of 15 or 30 task IDs")
    if len(set(ids))!=len(ids) or not all(isinstance(x,str) for x in ids):
        raise ValueError("Task IDs must be unique strings")
    rows=[json.loads(s) for s in Path(tasks_path).read_text(encoding="utf-8").splitlines() if s.strip()]
    indexed={t["instance_id"]:t for t in rows}
    if len(indexed)!=len(rows):
        raise ValueError("Duplicate instance IDs in source dataset")
    missing=set(ids)-set(indexed)
    if missing:
        raise ValueError("Unknown instance IDs: "+", ".join(sorted(missing)))
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    with Path(output_path).open("w",encoding="utf-8") as f:
        for id in ids:
            f.write(json.dumps(indexed[id],ensure_ascii=False)+"\n")
    print(f"Wrote {len(ids)} original benchmark tasks to {output_path}")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source_tasks_jsonl",type=Path)
    p.add_argument("ids_json",type=Path)
    p.add_argument("output_jsonl",type=Path)
    args=p.parse_args()
    materialize(args.source_tasks_jsonl,args.ids_json,args.output_jsonl)
