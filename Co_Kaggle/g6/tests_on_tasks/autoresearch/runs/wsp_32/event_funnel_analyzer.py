"""True 5-Stage Causal Event-Chain Funnel Analyzer over frozen evaluator `diagnostics.zip` archives.

Replaces the mislabeled `premature_submits` proxy (`len(ok_edits)==0 or tool_calls<=15`) with exact
chronological episode events per task:
  Stage 1 (`ok_edit`)            : >=1 `edit_file` / `write_file` returned `status == 'ok'`
  Stage 2 (`valid_test_ran`)     : After an `ok_edit`, `run_command` executed a real test (`exit_code in {0, 1}`,
                                   excluding `exit_code in {2, 4, 5}` bad path / collection error & command timeouts)
  Stage 3 (`defect_exposed`)     : `valid_test_ran` after `ok_edit` exited non-zero (`FAILED` / `Error` / `AssertionError` / `TypeError`)
  Stage 4 (`post_fail_ok_edit`)  : After `defect_exposed`, agent applied >=1 subsequent `ok_edit` (actual repair attempt!)
  Stage 5 (`verified_submit`)    : After final `ok_edit`, a test passed (`exit_code == 0` with `passed`) AND `submit_patch` called
  Outcome (`grader_pass`)        : Official harness grader `resolved == True`

Also classifies exact terminal failure mode:
  - `UNCHECKED_SUBMIT`      : `submit_patch` called after `ok_edit` with ZERO `valid_test_ran` since that edit
  - `BAD_TEST_PATH_SUBMIT`  : Ran `pytest` on wrong path (`exit_code in {4, 5}`, 0 tests ran) then called `submit_patch`
  - `RATIONALIZED_FAIL_SUB` : Observed `defect_exposed` (`exit_code == 1/2`), made NO subsequent `ok_edit`, called `submit_patch`
  - `REPAIR_ATTEMPTED_FAIL` : Observed `defect_exposed`, applied `post_fail_ok_edit`, submitted, still failed grader
  - `TIMEOUT_EDIT_ERR_LOOP` : Timed out (`submits == 0`) with >=3 `FileEditError`s
  - `TIMEOUT_SLOW_CMD`      : Timed out (`submits == 0`) after >=45s `run_command`
  - `TIMEOUT_NO_EDIT`       : Timed out with `ok_edits == 0`
  - `VERIFIED_WRONG_SPEC`   : Agent's own test passed + submitted, but official grader test failed (incomplete edge case)
"""
from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from pathlib import Path

TEST_CMD_RE = re.compile(
    r"(?:^|\s|&&|;)(?:pytest|python(?:3)?\s+-m\s+pytest|python(?:3)?\s+-c\s+)"
)


def parse_diagnostics_zip(diag_path: Path, resolved_ids: set[str] | None = None) -> dict:
    if diag_path.is_dir():
        names = [p.relative_to(diag_path).as_posix() for p in diag_path.rglob("*") if p.is_file()]
        read_bytes = lambda rel: (diag_path / rel).read_bytes()
    else:
        zf = zipfile.ZipFile(diag_path, "r")
        names = zf.namelist()
        read_bytes = lambda rel: zf.read(rel)
    try:
        res_files = [n for n in names if n.endswith("results.json") or n.endswith("summary.json")]
        resolved_from_zip = set()
        rec_meta = {}
        for rf in res_files:
            try:
                data = json.loads(read_bytes(rf))
                rows = data if isinstance(data, list) else data.get("results") or data.get("records") or []
                if isinstance(rows, list):
                    for r in rows:
                        if isinstance(r, dict) and "instance_id" in r:
                            rec_meta[r["instance_id"]] = r
                            if r.get("resolved") is True:
                                resolved_from_zip.add(r["instance_id"])
            except Exception:
                pass
        res_set = resolved_ids if resolved_ids is not None else resolved_from_zip

        tool_files = sorted(n for n in names if ("report/" in n) and n.endswith("_tools.json"))
        episodes = []
        if tool_files:
            for tf in tool_files:
                tid = Path(tf).name.removesuffix("_tools.json")
                tools = json.loads(read_bytes(tf))
                episodes.append(analyze_tool_trace(tid, tools, tid in res_set, rec_meta.get(tid, {})))
        else:
            ev_files = sorted(n for n in names if ("observer/" in n) and n.endswith("/events.jsonl"))
            for ef in ev_files:
                tid = Path(ef).parent.name
                evs = [json.loads(s) for s in read_bytes(ef).splitlines() if s.strip()]
                tools = events_to_tools(evs)
                episodes.append(analyze_tool_trace(tid, tools, tid in res_set, rec_meta.get(tid, {})))
    finally:
        if not diag_path.is_dir():
            zf.close()

    stage_counts = {
        "total_tasks": len(episodes),
        "s1_ok_edit": sum(1 for e in episodes if e["s1_ok_edit"]),
        "s2_valid_test_ran": sum(1 for e in episodes if e["s2_valid_test_ran"]),
        "s3_defect_exposed": sum(1 for e in episodes if e["s3_defect_exposed"]),
        "s4_post_fail_ok_edit": sum(1 for e in episodes if e["s4_post_fail_ok_edit"]),
        "s5_verified_submit": sum(1 for e in episodes if e["s5_verified_submit"]),
        "grader_pass": sum(1 for e in episodes if e["resolved"]),
        "closed_loop_repairs_solved": sum(1 for e in episodes if e["s4_post_fail_ok_edit"] and e["resolved"]),
    }
    modes = dict(Counter(e["mode"] for e in episodes))
    return {"funnel": stage_counts, "modes": modes, "episodes": episodes}


def events_to_tools(evs: list[dict]) -> list[dict]:
    calls_by_id = {}
    out = []
    for ev in evs:
        for c in ev.get("calls") or []:
            calls_by_id[c.get("tool_call_id")] = c
        for r in ev.get("responses") or []:
            cid = r.get("tool_call_id")
            call = calls_by_id.get(cid, {})
            resp = r.get("response")
            if isinstance(resp, str):
                try:
                    resp = json.loads(resp)
                except Exception:
                    resp = {"raw": resp}
            resp = resp or {}
            st = resp.get("status", "ok" if r.get("function_name") == "run_command" else "error")
            out.append({
                "tool_name": r.get("function_name") or call.get("function_name"),
                "arguments": call.get("arguments") or {},
                "status": st,
                "result": resp,
            })
    return out


def analyze_tool_trace(tid: str, tools: list[dict], resolved: bool, meta: dict) -> dict:
    s1_ok_edit = False
    s2_valid_test_ran = False
    s3_defect_exposed = False
    s4_post_fail_ok_edit = False
    s5_verified_submit = False

    checked_since_last_edit = False
    passed_since_last_edit = False
    bad_path_test_seen = False
    ok_edits = 0
    edit_errors = 0
    submits = 0
    test_cmds = []

    for t in tools:
        fn = t.get("tool_name")
        args = t.get("arguments") or t.get("args") or {}
        res = t.get("result") or {}
        if isinstance(res, str):
            try:
                res = json.loads(res)
            except Exception:
                res = {"raw": res}
        st = t.get("status") or res.get("status")

        if fn in ("edit_file", "write_file"):
            if st == "ok" or res.get("status") == "ok":
                ok_edits += 1
                s1_ok_edit = True
                if s3_defect_exposed:
                    s4_post_fail_ok_edit = True
                checked_since_last_edit = False
                passed_since_last_edit = False
            else:
                edit_errors += 1
        elif fn == "run_command":
            cmd = str(args.get("command") or res.get("command") or "").strip()
            details = res.get("details") if isinstance(res.get("details"), dict) else {}
            ec = res.get("exit_code", res.get("returncode"))
            if ec is None:
                ec = details.get("exit_code", details.get("returncode"))
            err_type = str(res.get("error_type") or "")
            stdout = "\n".join(
                str(x)
                for x in (
                    res.get("stdout"),
                    res.get("stderr"),
                    res.get("error_message"),
                    details.get("stdout"),
                    details.get("stderr"),
                )
                if x
            )
            is_timeout = ("Timeout" in err_type) or ("timed out" in stdout.lower())
            is_test = bool(TEST_CMD_RE.search(cmd)) or ("pytest" in cmd)
            if s1_ok_edit and is_test:
                test_cmds.append({"cmd": cmd[:80], "ec": ec, "err_type": err_type or None})
                if is_timeout:
                    pass
                elif ec in (4, 5) or "collected 0 items" in stdout or "file or directory not found" in stdout.lower() or "no tests ran" in stdout.lower():
                    bad_path_test_seen = True
                elif ec == 0 and ("passed" in stdout or "python" in cmd):
                    s2_valid_test_ran = True
                    checked_since_last_edit = True
                    passed_since_last_edit = True
                elif ec not in (0, None):
                    s2_valid_test_ran = True
                    checked_since_last_edit = True
                    s3_defect_exposed = True
        elif fn == "submit_patch":
            submits += 1
            if s1_ok_edit and passed_since_last_edit:
                s5_verified_submit = True

    stop_reason = meta.get("stop_reason")
    if resolved:
        mode = "SOLVED_AFTER_REPAIR" if s4_post_fail_ok_edit else ("SOLVED_VERIFIED" if s2_valid_test_ran else "SOLVED_UNCHECKED")
    elif submits == 0 or stop_reason == "session_timeout":
        if not s1_ok_edit:
            mode = "TIMEOUT_NO_EDIT"
        elif edit_errors >= 3:
            mode = "TIMEOUT_EDIT_ERR_LOOP"
        else:
            mode = "TIMEOUT_OTHER"
    else:
        if not s1_ok_edit:
            mode = "SUBMIT_WITHOUT_EDIT"
        elif not checked_since_last_edit and bad_path_test_seen:
            mode = "BAD_TEST_PATH_SUBMIT"
        elif not checked_since_last_edit:
            mode = "UNCHECKED_SUBMIT"
        elif passed_since_last_edit:
            mode = "VERIFIED_WRONG_SPEC"
        elif s4_post_fail_ok_edit:
            mode = "REPAIR_ATTEMPTED_FAIL"
        else:
            mode = "RATIONALIZED_FAIL_SUB"

    return {
        "instance_id": tid,
        "resolved": resolved,
        "mode": mode,
        "tool_calls": len(tools),
        "ok_edits": ok_edits,
        "edit_errors": edit_errors,
        "submits": submits,
        "s1_ok_edit": s1_ok_edit,
        "s2_valid_test_ran": s2_valid_test_ran,
        "s3_defect_exposed": s3_defect_exposed,
        "s4_post_fail_ok_edit": s4_post_fail_ok_edit,
        "s5_verified_submit": s5_verified_submit,
        "post_edit_tests": test_cmds,
    }
