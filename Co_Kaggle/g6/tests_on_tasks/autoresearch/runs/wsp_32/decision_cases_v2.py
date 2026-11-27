"""Extract post-edit decision cases AND preservation states (verified, edit, localize) from discovery runs."""
import hashlib
import json
import re
import zipfile
from pathlib import Path

TEST_CMD = re.compile(
    r"(?:^|\s|&&|;)(?:pytest|python(?:3)?\s+-m\s+pytest|python(?:3)?\s+-c\s+.*?\bassert\b)"
)


def read_cases_from_runs(runs: list[Path]) -> list[dict]:
    all_cases = []
    for run in runs:
        run = Path(run)
        requests = [json.loads(s) for s in (run / "requests.jsonl").read_text().splitlines() if s.strip()]
        rendered = {
            r["request_sha"]: r["tokens"]
            for r in map(json.loads, (run / "replay_tokens.jsonl").read_text().splitlines())
        }
        metrics = json.loads((run / "neon_metrics.json").read_text())
        with zipfile.ZipFile(run / "diagnostics.zip") as z:
            tasks = json.loads(z.read(next(n for n in z.namelist() if n.endswith("/selected_tasks.json"))))
            # Assign each HTTP request to the unique task in `tasks` whose problem_statement lines match it!
            def _unique_line(ps: str, all_ps: list[str]) -> str:
                cands = [ln.strip() for ln in ps.splitlines() if len(ln.strip()) >= 8]
                unique_cands = [ln for ln in cands if sum(1 for other in all_ps if ln in other) == 1]
                pool = unique_cands or cands
                return max(pool, key=len)

            all_ps = [t["problem_statement"] for t in tasks]
            for task in tasks:
                issue = task["instance_id"]
                needle = _unique_line(task["problem_statement"], all_ps)
                relevant = [r for r in requests if any(needle in str(m.get("content") or "") for m in r["messages"])]
                prefix = next(n for n in z.namelist() if n.endswith(f"/observer/{issue}/events.jsonl"))
                events = [json.loads(s) for s in z.read(prefix).splitlines() if s.strip()]
                ledger = [
                    json.loads(s)
                    for s in z.read(prefix.replace("events.jsonl", "requests.jsonl")).splitlines()
                    if s.strip()
                ]
                if len(relevant) not in (len(ledger), len(ledger) - 1):
                    raise ValueError(f"Cannot match exact HTTP requests for {issue}: {len(ledger)} vs {len(relevant)} (needle={needle!r})")

                current = None
                edited = checked = attempted = False
                any_edit_yet = False
                calls_by_id = {}
                path = None
                loc_captured = 0

                for event in events:
                    if event["kind"] == "model_start":
                        current = int(event["request_id"].split("_")[-1]) - 1
                    elif event.get("calls") and current is not None:
                        calls_by_id.update({c["tool_call_id"]: c for c in event["calls"]})
                        if len(event["calls"]) != 1:
                            continue
                        call = event["calls"][0]
                        action, args = call["function_name"], call["arguments"] or {}
                        if current >= len(relevant):
                            continue
                        request = relevant[current]
                        digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
                        tokens = rendered.get(digest)
                        if not tokens:
                            continue
                        count = ledger[current].get("input_tokens")
                        if count and len(tokens) != count:
                            continue

                        role = None
                        if edited and action == "submit_patch":
                            if checked and issue in metrics["resolved_ids"]:
                                role = "verified"
                            elif not attempted:
                                role = "unchecked"
                        elif edited and not checked and action == "run_command" and TEST_CMD.search(args.get("command", "")):
                            role = "check"
                        elif action == "edit_file" and args.get("old_string") and args.get("new_string"):
                            role = "edit"
                        elif not any_edit_yet and action in ("read_file", "run_command") and loc_captured < 2:
                            role = "localize"
                            loc_captured += 1

                        if role:
                            all_cases.append(
                                dict(
                                    issue=issue,
                                    repo=task["repo"],
                                    snapshot=task["base_commit"],
                                    role=role,
                                    filepath=path,
                                    action=action,
                                    arguments=args,
                                    tokens=tokens,
                                    prompt_sha=digest,
                                    observed_input_tokens=count,
                                    source_run=run.name,
                                )
                            )
                    elif event.get("responses"):
                        for response in event["responses"]:
                            data = response.get("response") or {}
                            if isinstance(data, str):
                                try:
                                    data = json.loads(data)
                                except Exception:
                                    data = {}
                            if response["function_name"] in ("edit_file", "write_file") and data.get("status") == "ok":
                                edited, checked, attempted = True, False, False
                                any_edit_yet = True
                                path = data.get("filepath")
                            elif response["function_name"] == "run_command" and current is not None:
                                call = calls_by_id.get(response["tool_call_id"])
                                if call:
                                    cmd = (call.get("arguments") or {}).get("command", "")
                                    output = str(data.get("stdout", ""))
                                    if TEST_CMD.search(cmd):
                                        attempted = True
                                    if TEST_CMD.search(cmd) and data.get("exit_code", data.get("returncode")) == 0 and (
                                        re.search(r"\b\d+ passed\b", output) or "python" in cmd
                                    ):
                                        checked = True
    return all_cases
