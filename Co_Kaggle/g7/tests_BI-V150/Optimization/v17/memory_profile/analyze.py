#!/usr/bin/env python3
"""Offline trace/metadata/Prometheus summaries. Missing observations stay missing."""
import argparse
import collections
import csv
import gzip
import json
import math
from pathlib import Path
import re
import statistics


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def union_us(intervals):
    end = float("-inf")
    total = 0
    for a,b in sorted(intervals):
        total += max(0,b-max(a,end))
        end = max(end,b)
    return total


def trace_summary(path):
    opener = gzip.open if path.suffix==".gz" else open
    with opener(path, "rt") as f:
        data = json.load(f)
    events = data.get("traceEvents", []) if isinstance(data, dict) else data
    groups = {}
    shapes = collections.Counter()
    intervals = []
    graph = 0
    mem_alloc = mem_free = 0
    memory_events = 0
    for e in events:
        name = str(e.get("name", ""))
        cat = str(e.get("cat", "")).lower()
        args = e.get("args") or {}
        lower = name.lower()
        if "graphlaunch" in lower or "graph_launch" in lower:
            graph += 1
        if name == "[memory]" and isinstance(args.get("Bytes"), (int,float)):
            memory_events += 1
            b = args["Bytes"]
            mem_alloc += max(b,0)
            mem_free += max(-b,0)
        if e.get("ph") != "X" or not isinstance(e.get("dur"), (int,float)):
            continue
        dur = e["dur"]
        if cat=="kernel":
            kind = "kernel"
            if isinstance(e.get("ts"), (float,int)):
                intervals.append((e["ts"],e["ts"]+dur))
        elif "memcpy" in cat or "memset" in cat:
            kind = "device_copy" if "memcpy" in cat else "device_memset"
        elif "runtime" in cat or "driver" in cat:
            if not any(x in lower for x in ("synchroniz","wait","memcpy","memset","graphlaunch","graph_launch")):
                continue
            kind = "host_api"
        elif cat=="cpu_op":
            kind = "cpu_op"
        else:
            continue
        key = kind,name
        row = groups.setdefault(key, dict(kind=kind, name=name, calls=0, summed_us=0,
                                          reported_copy_bytes=0, copy_byte_events=0))
        row["calls"] += 1
        row["summed_us"] += dur
        if kind=="device_copy":
            for field in ("bytes","Bytes"):
                if isinstance(args.get(field), (int,float)):
                    row["reported_copy_bytes"] += args[field]
                    row["copy_byte_events"] += 1
                    break
        if kind=="cpu_op" and ("Input Dims" in args or "Input type" in args):
            shapes[(name,json.dumps(args.get("Input Dims")),json.dumps(args.get("Input type")))] += 1
    for row in groups.values():
        if not row["copy_byte_events"]:
            row["reported_copy_bytes"] = None
    span = max(b for a,b in intervals)-min(a for a,b in intervals) if intervals else None
    busy = union_us(intervals) if intervals else None
    overview = dict(trace=str(path), kernel_events=len(intervals), observed_kernel_span_us=span,
                    observed_kernel_union_us=busy,
                    unobserved_within_span_us=span-busy if span is not None else None,
                    graph_launch_events=graph, graph_visibility_unverified=bool(graph),
                    memory_events=memory_events,
                    allocator_alloc_bytes=mem_alloc if memory_events else None,
                    allocator_free_bytes=mem_free if memory_events else None)
    return overview, list(groups.values()), [
        dict(name=k[0], input_dims=k[1], input_types=k[2], calls=v) for k,v in shapes.items()]


def jsonl(path):
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                yield json.loads(line)


def metric_summary(path):
    # Keep series labels separate; no combining gauges/counters across engines.
    metrics = collections.defaultdict(list)
    boundaries = collections.defaultdict(dict)
    wanted = ("kv_cache_usage", "num_preemptions", "num_requests_running",
              "num_requests_waiting", "prefix_cache", "prompt_tokens",
              "generation_tokens")
    for row in jsonl(path):
        for line in row.get("metrics","").splitlines():
            if line.startswith("#") or not line.startswith("vllm:"):
                continue
            match = re.match(r'^(\S+(?:\{.*\})?)\s+([-+\deE.]+)(?:\s+\d+)?$',line)
            # Labels may contain spaces; prefer a split from the right otherwise.
            if not match:
                continue
            key = match[1]
            if any(x in key for x in wanted) and "_created" not in key:
                try:
                    value = float(match[2])
                    if math.isfinite(value):
                        group=(row["workload"],key)
                        if row.get("boundary") in ("before","after"):
                            boundaries[group][row["boundary"]]=value
                        else:
                            metrics[group].append(value)
                except ValueError:
                    pass
    result=[]
    for k in sorted(set(metrics)|set(boundaries)):
        v=metrics[k]
        before=boundaries[k].get("before")
        after=boundaries[k].get("after")
        counter=k[1].split("{")[0].endswith("_total")
        ordered=[x for x in [before,*v,after] if x is not None]
        monotonic=all(a<=b for a,b in zip(ordered,ordered[1:]))
        exact=counter and before is not None and after is not None and monotonic
        result.append(dict(workload=k[0],series=k[1],samples=len(v),
            minimum=min(v) if v else None,maximum=max(v) if v else None,
            mean=statistics.fmean(v) if v else None,
            first=v[0] if v else None,last=v[-1] if v else None,
            before=before,after=after,
            counter_delta=after-before if exact else None,
            sampled_counter_delta=v[-1]-v[0] if counter and monotonic and len(v)>1 else None))
    return result



def numeric_metric(value):
    try:
        return math.isfinite(float(value.replace(",","")))
    except ValueError:
        return False


def native_rows(path):
    """Recognize explicit long CSV or raw wide CSV with a units row.
    Do not infer units, metric columns or kernel IDs from arbitrary log lines.
    """
    header = units = mode = None
    memory = re.compile(r"dram|hbm|memory|cache|l2|l1|stall|bandwidth|throughput|occupancy|matrix|tensor", re.I)

    def record(row, name, value, unit, section=""):
        return dict(report=path.name, kernel=row.get("kernel name",""),
                    launch_id=row.get("id",""), section=section,
                    metric=name, value=value, unit=unit,
                    memory_related=bool(memory.search(name)),
                    numeric_value=numeric_metric(value))

    for line in path.read_text(errors="replace").splitlines():
        if not line.strip() or line.startswith(("COMMAND ","==","PROBE_JSON ")):
            continue
        try:
            fields = next(csv.reader([line]))
        except csv.Error:
            continue
        norm = [x.strip().lower().replace("_", " ") for x in fields]
        if "kernel name" in norm and "id" in norm:
            header = norm
            original = [x.strip() for x in fields]
            mode = "long" if "metric name" in norm and "metric value" in norm else "wide"
            units = None
            continue
        if header is None or len(fields) != len(header):
            continue
        row = dict(zip(header,fields))
        if mode=="long":
            name,value=row["metric name"].strip(),row["metric value"].strip()
            if name and value and row.get("kernel name","").strip():
                yield record(row,name,value,row.get("metric unit",""),row.get("section name",""))
        elif not row.get("id","").strip() and not row.get("kernel name","").strip():
            # Raw reports can include an explicit unit row. Do not assume
            # a unit when the header has no such row.
            units=fields
        elif row.get("id","").strip().isdigit() and row.get("kernel name","").strip():
            # Raw wide reports identify metric columns by metric syntax or
            # a recognizable metric name. Skip launch metadata fields.
            metadata={"id","process id","process name","host name","kernel name","context",
                      "stream","device","device id","grid size","block size","section name",
                      "kernel time","gpu name","cc","invocations","grid","block","context id","stream id"}
            for i,(name,value) in enumerate(zip(original,fields)):
                if header[i] in metadata or not value.strip():
                    continue
                if ("__" in name or "." in name or memory.search(name)) and numeric_metric(value):
                    yield record(row,name,value,units[i] if units else "")


def native_summary(root):
    metrics, probes, reports = [], [], []
    for path in sorted((root/"native").glob("*.counters.csv")):
        rows = list(native_rows(path))
        metrics.extend(rows)
        reports.append(dict(report=path.name, parsed_rows=len(rows),
                            memory_related_rows=sum(r["memory_related"] and r["numeric_value"] for r in rows),
                            status="parsed" if rows else "unverified_no_recognized_metric_rows"))
    for path in sorted((root/"native").glob("*.timing.log")):
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith("PROBE_JSON "):
                row = json.loads(line[len("PROBE_JSON "):])
                probes.append(dict(report=path.name, **{
                    k: json.dumps(v) if isinstance(v,list) else v for k,v in row.items()}))
    write_csv(root/"native_metrics.csv", metrics)
    write_csv(root/"native_report_coverage.csv", reports)
    write_csv(root/"probe_payloads.csv", probes)
    return dict(native_reports=len(reports), parsed_native_rows=len(metrics),
                unverified_native_reports=sum(not r["parsed_rows"] for r in reports),
                timed_probes=len(probes))


def summarize(root):
    overviews, allgroups, allshapes, errors = [],[],[],[]
    capture_map={}
    for window in jsonl(root/"windows.jsonl"):
        for file in window.get("files",[]):
            capture_map[str(root/file)]=(window["workload"],window["index"])
    for path in sorted((root/"traces").rglob("*.json*")):
        try:
            overview, groups, shapes = trace_summary(path)
            workload,index=capture_map.get(str(path),("unknown",None))
            label=dict(workload=workload,window_index=index)
            overviews.append(dict(**overview,**label))
            allgroups.extend(dict(trace=str(path),**label,**r) for r in groups)
            allshapes.extend(dict(trace=str(path),**label,**r) for r in shapes)
        except Exception as e:
            errors.append(dict(path=str(path), error=str(e)))
    write_csv(root/"trace_overview.csv", overviews)
    write_csv(root/"trace_events.csv", allgroups)
    write_csv(root/"operator_shapes.csv", allshapes)
    write_csv(root/"capacity.csv", metric_summary(root/"metrics.jsonl"))
    rows = list(jsonl(root/"batch_metadata.jsonl"))
    windows = list(jsonl(root/"workloads.jsonl"))
    for row in rows:
        row["workload"] = next((w["workload"] for w in windows
                                if w["start"]<=row["timestamp"]<=w["end"]), "other")
    simple = [{k:v for k,v in r.items() if not isinstance(v,list)} for r in rows]
    write_csv(root/"batch_steps.csv", simple)
    by_kind = collections.defaultdict(list)
    for r in rows:
        by_kind[(r["workload"],r.get("batch_kind_by_prompt_progress",r["batch_kind_by_query_length"]))].append(r)
    write_csv(root/"batch_summary.csv", [
        dict(workload=k[0], kind=k[1], sampled_steps=len(v),
             mean_requests=statistics.fmean(r["requests"] for r in v),
             resumed_requests=sum(r.get("resumed_requests",0) for r in v),
             repeated_scheduled_tokens=sum(r.get("repeated_scheduled_tokens",0) for r in v),
             resumed_replay_scheduled_tokens=sum(r.get("resumed_replay_scheduled_tokens",0) for r in v),
             mean_scheduled=statistics.fmean(r["scheduled"] for r in v),
             mean_kv_payload_proxy_bytes=statistics.fmean(r["unique_kv_payload_bytes_per_layer"] for r in v),
             mean_padded_query_positions_proxy=statistics.fmean(r["padded_query_positions_proxy"] for r in v),
             mean_active_tail_slack_tokens_proxy=statistics.fmean(r["active_tail_slack_tokens_proxy"] for r in v))
        for k,v in by_kind.items()])
    (root/"analysis_status.json").write_text(json.dumps(dict(
        traces=len(overviews), trace_errors=errors, sampled_batch_steps=len(rows),
        **native_summary(root),
        note="Kernel durations are summed; CPU scopes overlap; graph gaps are not idle time. "
             "Allocator bytes are not HBM traffic; metadata values are proxies."),indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("run",type=Path)
    summarize(p.parse_args().run)
