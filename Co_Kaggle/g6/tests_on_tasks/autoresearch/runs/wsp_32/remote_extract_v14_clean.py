"""Extract 100% clean, zero-off-by-one immediate-tool-response cases across all 19 disjoint snapshots (02+03+04) and replay with vLLM."""
import hashlib
import json
import os
import re
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path("/content/wsp")
DATA = ROOT / "input/competitions/gemma-4-developer-agent"

TEST_CMD = re.compile(
    r"(?:^|\s|&&|;)(?:pytest|python(?:3)?\s+-m\s+pytest|python(?:3)?\s+-c\s+.*?\bassert\b)"
)


def extract_all_clean_turns(runs: list[Path]) -> list[dict]:
    """Chronologically align every single-tool turn to its EXACT immediate preceding tool response."""
    out = []
    for run in runs:
        run = Path(run)
        if not (run / "requests.jsonl").is_file():
            continue
        reqs = [json.loads(s) for s in (run / "requests.jsonl").read_text().splitlines() if s.strip()]
        tok_map = {
            r["request_sha"]: r["tokens"]
            for r in map(json.loads, (run / "replay_tokens.jsonl").read_text().splitlines())
        }
        metrics = json.loads((run / "neon_metrics.json").read_text())
        with zipfile.ZipFile(run / "diagnostics.zip") as dz:
            tasks = json.loads(dz.read(next(n for n in dz.namelist() if n.endswith("/selected_tasks.json"))))
            all_ps = [t["problem_statement"] for t in tasks]

            for task in tasks:
                issue = task["instance_id"]
                cands = [ln.strip() for ln in task["problem_statement"].splitlines() if len(ln.strip()) >= 8]
                uniq = [ln for ln in cands if sum(1 for o in all_ps if ln in o) == 1] or cands
                needle = max(uniq, key=len)
                relevant = [r for r in reqs if any(needle in str(m.get("content") or "") for m in r["messages"])]

                pref = next(n for n in dz.namelist() if n.endswith(f"/observer/{issue}/events.jsonl"))
                evs = [json.loads(s) for s in dz.read(pref).splitlines() if s.strip()]
                ledger = [
                    json.loads(s)
                    for s in dz.read(pref.replace("events.jsonl", "requests.jsonl")).splitlines()
                    if s.strip()
                ]

                cur = -1
                imm_state = "TURN1_INIT"
                has_ok_edit = False
                checked_since_edit = False
                last_filepath = None
                calls_by_id = {}

                for ev in evs:
                    if ev["kind"] == "model_start":
                        cur += 1
                    elif ev.get("calls") and 0 <= cur < min(len(relevant), len(ledger)):
                        for c in ev["calls"]:
                            calls_by_id[c["tool_call_id"]] = c
                        if len(ev["calls"]) != 1:
                            continue
                        call = ev["calls"][0]
                        fn = call["function_name"]
                        args = call["arguments"] or {}
                        req_obj = relevant[cur]
                        sha = hashlib.sha256(json.dumps(req_obj, sort_keys=True).encode()).hexdigest()
                        tokens = tok_map.get(sha)
                        if not tokens:
                            continue
                        obs_in = ledger[cur].get("input_tokens")
                        if obs_in and len(tokens) != obs_in:
                            continue

                        # Determine exact clean role WITHOUT any off-by-one or stale `edited` contamination!
                        clean_role = None
                        if imm_state == "IMMEDIATE_EDIT_OK":
                            clean_role = "imm_edit_ok"
                        elif has_ok_edit and not checked_since_edit and fn == "submit_patch" and imm_state != "AFTER_SUBMIT":
                            clean_role = "post_edit_prem_submit"
                        elif imm_state == "IMMEDIATE_EDIT_ERR" and imm_state != "AFTER_SUBMIT":
                            clean_role = "imm_edit_err"
                        elif imm_state == "IMMEDIATE_READ_FULL":
                            clean_role = "imm_read_full"
                        elif imm_state == "IMMEDIATE_READ_TRUNC":
                            clean_role = "imm_read_trunc"
                        elif imm_state == "IMMEDIATE_TEST_PASS":
                            clean_role = "imm_test_pass"
                        elif imm_state == "TURN1_INIT":
                            clean_role = "turn1_init"

                        if clean_role:
                            out.append({
                                "issue": issue,
                                "repo": task["repo"],
                                "snapshot": task["base_commit"],
                                "clean_role": clean_role,
                                "imm_state": imm_state,
                                "has_ok_edit": has_ok_edit,
                                "checked_since_edit": checked_since_edit,
                                "resolved": issue in metrics.get("resolved_ids", []),
                                "filepath": last_filepath or args.get("filepath"),
                                "action": fn,
                                "arguments": args,
                                "tokens": tokens,
                                "prompt_sha": sha,
                                "turn_idx": cur,
                                "source_run": run.name,
                            })
                    elif ev.get("responses"):
                        r0 = ev["responses"][-1]
                        rfn = r0["function_name"]
                        d = r0.get("response") or {}
                        if isinstance(d, str):
                            try:
                                d = json.loads(d)
                            except Exception:
                                d = {"raw": d}
                        if rfn in ("edit_file", "write_file"):
                            if d.get("status") == "ok":
                                has_ok_edit = True
                                checked_since_edit = False
                                last_filepath = d.get("filepath") or last_filepath
                                imm_state = "IMMEDIATE_EDIT_OK"
                            else:
                                imm_state = "IMMEDIATE_EDIT_ERR"
                        elif rfn == "read_file":
                            last_filepath = d.get("filepath") or last_filepath
                            imm_state = "IMMEDIATE_READ_TRUNC" if d.get("is_truncated") else "IMMEDIATE_READ_FULL"
                        elif rfn == "run_command":
                            call_ref = calls_by_id.get(r0.get("tool_call_id"), {})
                            cmd_str = (call_ref.get("arguments") or {}).get("command", "")
                            stdout = str(d.get("stdout", ""))
                            ec = d.get("exit_code", d.get("returncode"))
                            if has_ok_edit and TEST_CMD.search(cmd_str) and ec == 0 and (
                                re.search(r"\b\d+ passed\b", stdout) or "python" in cmd_str
                            ):
                                checked_since_edit = True
                                imm_state = "IMMEDIATE_TEST_PASS"
                            elif has_ok_edit and TEST_CMD.search(cmd_str) and ec not in (0, None):
                                imm_state = "IMMEDIATE_TEST_FAIL"
                            else:
                                imm_state = "IMMEDIATE_CMD_EXPLORE"
                        elif rfn == "submit_patch":
                            imm_state = "AFTER_SUBMIT"
    return out


def parse_gemma4_call(txt: str) -> tuple[str | None, dict]:
    m = re.search(r"<\|tool_call>call:(\w+)\{", txt)
    action = m[1] if m else None
    args = {k: v for k, v in re.findall(r'(\w+):<\|"\|>(.*?)<\|"\|>', txt, re.S)}
    return action, args


def main():
    from adk_submission.server import _round_up_vllm_lora_rank
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    runs = [
        ROOT / "experiments/discovery02",
        ROOT / "experiments/discovery03",
        ROOT / "experiments/discovery04",
    ]
    clean_turns = extract_all_clean_turns(runs)

    # Add the 3 exact verified calibration states from preservation/exact/cases.json as well
    ver_exact = ROOT / "preservation/exact/cases.json"
    if ver_exact.is_file():
        for vc in json.loads(ver_exact.read_text()):
            clean_turns.append({
                "issue": vc["issue"],
                "repo": vc.get("repo", "Textualize/rich"),
                "snapshot": vc.get("snapshot", ""),
                "clean_role": "imm_test_pass",
                "imm_state": "IMMEDIATE_TEST_PASS",
                "has_ok_edit": True,
                "checked_since_edit": True,
                "resolved": True,
                "filepath": vc.get("filepath"),
                "action": "submit_patch",
                "arguments": {},
                "tokens": vc["tokens"],
                "prompt_sha": vc["prompt_sha"],
                "turn_idx": 99,
                "source_run": "exact_verified",
            })

    print("Extracted clean turns by role:", Counter(c["clean_role"] for c in clean_turns), flush=True)

    # Deduplicate by prompt_sha and cap imm_read_full / imm_edit_err to at most 2 per issue so replay takes < 3 mins!
    selected = []
    seen_sha = set()
    per_issue_role = Counter()
    caps = {
        "imm_edit_ok": 3,
        "post_edit_prem_submit": 3,
        "imm_edit_err": 3,
        "imm_read_full": 2,
        "imm_read_trunc": 1,
        "imm_test_pass": 3,
        "turn1_init": 1,
    }
    for c in clean_turns:
        if c["prompt_sha"] in seen_sha:
            continue
        kr = (c["issue"], c["clean_role"])
        if per_issue_role[kr] >= caps.get(c["clean_role"], 2):
            continue
        seen_sha.add(c["prompt_sha"])
        per_issue_role[kr] += 1
        selected.append(c)

    print("Selected balanced clean benchmark/harvest turns:", Counter(c["clean_role"] for c in selected), flush=True)
    out_dir = ROOT / "harvest32/clean_v14"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clean_cases.json").write_text(json.dumps(selected))

    # Run native vLLM champion_r1 with max_tokens=4096 on all selected clean cases!
    prev_nat = {}
    old_nat = ROOT / "harvest32/native/champion.json"
    if old_nat.is_file():
        for r in json.loads(old_nat.read_text()):
            if r.get("action") is not None and r.get("text"):
                p_act, p_args = parse_gemma4_call(r["text"])
                prev_nat[r["prompt_sha"]] = {**r, "action": p_act, "arguments": p_args}

    need = [c for c in selected if c["prompt_sha"] not in prev_nat]
    print(f"Replaying {len(need)}/{len(selected)} clean cases natively at max_tokens=4096 (reusing {len(selected)-len(need)})...", flush=True)

    if need:
        assets = json.loads((ROOT / "assets.json").read_text())
        ad_path = ROOT / "experiments/discovery02/input/adapters/wsp_r1"
        engine = LLM(
            model=assets["model"], dtype="bfloat16", max_model_len=32768,
            max_num_seqs=8, max_num_batched_tokens=2048, enable_lora=True,
            max_loras=1, max_lora_rank=_round_up_vllm_lora_rank(1),
            gpu_memory_utilization=0.82, scheduler_reserve_full_isl=False, tensor_parallel_size=1,
        )
        tok = engine.get_tokenizer()
        params = SamplingParams(
            temperature=0.0, top_p=0.95, max_tokens=4096,
            stop_token_ids=[tok.convert_tokens_to_ids(t) for t in ("<tool_call|>", "<turn|>")],
            skip_special_tokens=False,
        )
        req = LoRARequest("champion_r1", 1, str(ad_path))
        outs = engine.generate([{"prompt_token_ids": c["tokens"]} for c in need], params, lora_request=req, use_tqdm=False)
        for c, o in zip(need, outs, strict=True):
            txt = tok.decode(o.outputs[0].token_ids, skip_special_tokens=False)
            act, args = parse_gemma4_call(txt)
            prev_nat[c["prompt_sha"]] = {
                "issue": c["issue"],
                "role": c["clean_role"],
                "prompt_sha": c["prompt_sha"],
                "prompt_tokens": len(c["tokens"]),
                "action": act,
                "arguments": args,
                "text": txt,
            }

    champ_records = [prev_nat[c["prompt_sha"]] for c in selected]
    (out_dir / "champion_clean.json").write_text(json.dumps(champ_records, indent=2))
    summary_by_role = Counter((c["clean_role"], prev_nat[c["prompt_sha"]]["action"]) for c in selected)
    print("Native champion_r1 actions on clean states:")
    for (rl, act), v in sorted(summary_by_role.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        print(f"  {rl:24s} -> {str(act):14s}: {v}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    main()
