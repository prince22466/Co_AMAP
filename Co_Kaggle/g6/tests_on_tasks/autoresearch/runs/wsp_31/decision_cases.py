"""Pair exact development requests with the observed action and the prior tool outcomes."""
import hashlib
import json
import re
import zipfile
from pathlib import Path

TEST = re.compile(r"(?:^|\s)(?:pytest|python(?:3)?\s+-m\s+pytest)\b")


def read_cases(run):
    run = Path(run)
    requests = [json.loads(s) for s in (run / "requests.jsonl").read_text().splitlines()]
    rendered = {r["request_sha"]: r["tokens"] for r in map(json.loads, (run/"replay_tokens.jsonl").read_text().splitlines())}
    with zipfile.ZipFile(run/"diagnostics.zip") as z:
        tasks = json.loads(z.read(next(n for n in z.namelist() if n.endswith("/selected_tasks.json"))))
        metrics = json.loads((run/"neon_metrics.json").read_text())
        cases = []
        for task in tasks:
            issue = task["instance_id"]
            # The router sends the fully expanded issue in the HTTP input.
            title = task["problem_statement"].splitlines()[0].strip()
            relevant = [r for r in requests if any(title in str(m.get("content") or "") for m in r["messages"])]
            prefix = next(n for n in z.namelist() if n.endswith(f"/observer/{issue}/events.jsonl"))
            events = [json.loads(s) for s in z.read(prefix).splitlines()]
            ledger = [json.loads(s) for s in z.read(prefix.replace("events.jsonl","requests.jsonl")).splitlines()]
            if len(ledger) != len(relevant):
                raise ValueError(f"Cannot match exact HTTP requests for {issue}: {len(ledger)} vs {len(relevant)}")
            current = None
            edited = checked = attempted = False
            calls_by_id = {}
            path = None
            for event in events:
                if event["kind"] == "model_start":
                    current = int(event["request_id"].split("_")[-1]) - 1
                elif event.get("calls") and current is not None:
                    calls_by_id.update({c["tool_call_id"]:c for c in event["calls"]})
                    if len(event["calls"]) != 1:
                        continue
                    call = event["calls"][0]
                    action, args = call["function_name"], call["arguments"]
                    request = relevant[current]
                    digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
                    tokens = rendered[digest]
                    count = ledger[current].get("input_tokens")
                    if count and len(tokens) != count:
                        # An incompatible server template cannot establish the observed prefix.
                        continue
                    role = None
                    if edited and action == "submit_patch":
                        role = "verified" if checked and issue in metrics["resolved_ids"] else ("unchecked" if not attempted else None)
                    elif edited and not checked and action == "run_command" and TEST.search(args.get("command","")):
                        role = "check"
                    elif action == "edit_file" and args.get("old_string") and args.get("new_string"):
                        role = "edit"
                    if role:
                        cases.append(dict(issue=issue, snapshot=task["base_commit"], role=role,
                                          filepath=path, action=action, arguments=args, tokens=tokens,
                                          prompt_sha=digest, observed_input_tokens=count))
                elif event.get("responses"):
                    for response in event["responses"]:
                        data = response.get("response",{})
                        if response["function_name"] in ("edit_file","write_file") and data.get("status") == "ok":
                            edited, checked, attempted = True, False, False
                            path = data.get("filepath")
                        elif response["function_name"] == "run_command" and current is not None:
                            call = calls_by_id.get(response["tool_call_id"])
                            if call:
                                cmd = call["arguments"].get("command","")
                                output = str(data.get("stdout",""))
                                if TEST.search(cmd):
                                    attempted = True
                                if TEST.search(cmd) and data.get("exit_code",data.get("returncode")) == 0 and re.search(r"\b\d+ passed\b",output):
                                    checked = True
    return cases
