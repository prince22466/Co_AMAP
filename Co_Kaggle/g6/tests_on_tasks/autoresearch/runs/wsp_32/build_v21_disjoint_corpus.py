"""Build 100% disjoint, execution-grounded multi-stage repair corpus (`clean_v21/real_replays.json`) on Colab.

Strictly enforces zero snapshot overlap with `QUICK_30` and `HOLDOUT`.
Separates post-test states into:
  - `true_imm_edit_ok`        : immediate `ok` edit_file response -> needs targeted check (`run_command`)
  - `true_imm_bad_test_path`  : pytest exit_code in {4, 5} (wrong path / 0 items) -> needs corrected `run_command`
  - `true_imm_real_test_fail` : pytest/python exit_code in {1, 2} (real defect exposed!) -> needs `edit_file` / `read_file` repair!
  - `true_imm_test_pass`      : pytest/python exit_code == 0 after edit -> null-space (`submit_patch` preserved!)
  - `true_imm_edit_err`       : `FileEditError` -> needs `read_file`
  - `true_imm_read_full` / `true_imm_read_trunc` / `turn1_init` : null-space (`b3_exact_args_match` preserved!)
"""
from __future__ import annotations

import copy
import difflib
import hashlib
import json
import os
import re
import subprocess
import tarfile
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path("/content/wsp")
DATA = ROOT / "input/competitions/gemma-4-developer-agent"

QUICK_30_IDX = [
    24, 14, 17, 19, 21, 22, 66, 67, 72, 75, 107, 113, 71, 106, 117,
    16, 13, 95, 68, 69, 74, 100, 108, 109, 114, 115, 118, 119, 98, 127,
]
HOLDOUT_IDS = {"rich_3043", "rich_3064", "rich_2725"}
DISCOVERY_19 = {
    "rich_3506", "rich_3296", "rich_3067", "rich_3469", "rich_3130",
    "rich_4079", "rich_3480", "rich_3472", "rich_3105", "rich_3006",
    "fastapi_14786", "fastapi_14479", "requests_7328",
    "fastapi_15661", "fastapi_15588", "requests_7505", "requests_7502",
    "rich_4070", "rich_4006",
}

TR_RE = re.compile(r"<\|tool_response>response:(\w+)\{(.*?)\}<tool_response\|>", re.S)
TC_RE = re.compile(r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>", re.S)
TEST_CMD = re.compile(r"(?:^|\s|&&|;)(?:pytest|python(?:3)?\s+-m\s+pytest|python(?:3)?\s+-c\s+)")


def parse_gemma4_call(txt: str) -> tuple[str | None, dict]:
    m = re.search(r"<\|tool_call>call:(\w+)\{", txt)
    action = m[1] if m else None
    args = {k: v for k, v in re.findall(r'(\w+):<\|"\|>(.*?)<\|"\|>', txt, re.S)}
    return action, args


def _has_ok_status(s: str) -> bool:
    c = s.replace(" ", "")
    return ('"status":"ok"' in c) or ('status:<|"|>ok<|"|>' in c)


def _has_zero_exit(s: str) -> bool:
    c = s.replace(" ", "")
    return ('"exit_code":0' in c) or ("exit_code:0" in c) or ('"returncode":0' in c) or ("returncode:0" in c)


def _has_bad_path(s: str) -> bool:
    c = s.replace(" ", "")
    sl = s.lower()
    return (
        ('"exit_code":4' in c) or ("exit_code:4" in c)
        or ('"exit_code":5' in c) or ("exit_code:5" in c)
        or ("file or directory not found" in sl)
        or ("collected 0 items" in sl)
        or ("no tests ran" in sl)
    )


def classify_v21_role(tok, tks: list[int], source_tag: str = "") -> str:
    if source_tag in ("exact_verified", "exec_disjoint_pass"):
        return "true_imm_test_pass"
    if source_tag == "exec_disjoint_fail":
        return "true_imm_real_test_fail"
    txt = tok.decode(tks, skip_special_tokens=False)
    trs = list(TR_RE.finditer(txt))
    tcs = list(TC_RE.finditer(txt))
    if not trs:
        return "turn1_init" if "<|turn>user" in txt and txt.count("<|turn>model") <= 1 else "non_imm_prose"
    if any(m.group(1) == "submit_patch" for m in trs):
        return "after_submit_ignore"
    after = txt[trs[-1].end() :]
    if "<|turn>user" in after or "<|tool_call>" in after or len(after) > 80:
        return "non_imm_followup"

    has_ok_edit = any(m.group(1) in ("edit_file", "write_file") and _has_ok_status(m.group(2)) for m in trs)
    fn, body = trs[-1].group(1), trs[-1].group(2)
    if fn in ("edit_file", "write_file"):
        return "true_imm_edit_ok" if _has_ok_status(body) else "true_imm_edit_err"
    if fn == "read_file":
        c = body.replace(" ", "").lower()
        return "true_imm_read_trunc" if ('"is_truncated":true' in c or "is_truncated:true" in c) else "true_imm_read_full"
    if fn == "run_command":
        last_cmd = tcs[-1].group(2) if tcs and tcs[-1].group(1) == "run_command" else ""
        is_test = bool(TEST_CMD.search(last_cmd)) or ("pytest" in last_cmd)
        if has_ok_edit and is_test:
            if "CommandTimeoutError" in body or "timed out" in body.lower():
                return "true_imm_test_timeout"
            if _has_bad_path(body):
                return "true_imm_bad_test_path"
            if _has_ok_status(body) and _has_zero_exit(body) and ("passed" in body or "python" in last_cmd):
                return "true_imm_test_pass"
            return "true_imm_real_test_fail"
        return "true_imm_cmd_explore"
    return f"other_{fn}"


def render_chat(tokenizer, template_req: dict, messages: list[dict]) -> list[int]:
    norm = copy.deepcopy(messages)
    for msg in norm:
        if isinstance(msg.get("content"), str):
            msg["content"] = [{"type": "text", "text": msg["content"]}]
        for c in msg.get("tool_calls", []):
            args = c["function"]["arguments"]
            if isinstance(args, str):
                c["function"]["arguments"] = json.loads(args)
    return list(
        tokenizer.apply_chat_template(
            norm,
            tools=template_req.get("tools"),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
            **template_req.get("chat_template_kwargs", {}),
        )
    )


def extend_turn(messages: list[dict], fn_name: str, args: dict, response: dict) -> None:
    ident = f"calv21-{len(messages)}"
    messages.append({
        "role": "assistant",
        "tool_calls": [{"id": ident, "type": "function", "function": {"name": fn_name, "arguments": args}}],
    })
    messages.append({"role": "tool", "tool_call_id": ident, "content": json.dumps(response)})


def find_test_file_for_patch(work: Path, patch_text: str, issue: str) -> tuple[str, list[str]] | None:
    """Pick a fast existing test file inside the disjoint snapshot matching the patched module."""
    py = str(ROOT / "eval_env/bin/python")
    files = [ln[len("+++ b/") :].strip() for ln in patch_text.splitlines() if ln.startswith("+++ b/")]
    stems = [Path(f).stem.lstrip("_") for f in files if f.endswith(".py")]
    test_files = sorted(p.relative_to(work).as_posix() for p in (work / "tests").rglob("test_*.py")) if (work / "tests").is_dir() else []

    ordered = []
    for st in stems:
        for cand in (f"tests/test_{st}.py",):
            if cand in test_files and cand not in ordered:
                ordered.append(cand)
        for tf in test_files:
            if st and st in Path(tf).stem and tf not in ordered:
                ordered.append(tf)

    # Disjoint issue-specific fast test fallbacks if filename differs from test name (e.g. fastapi routing -> security/invalid_sequence/sse)
    fallbacks = {
        "rich_3296": ["tests/test_syntax.py", "tests/test_padding.py"],
        "rich_3506": ["tests/test_segment.py"],
        "rich_3130": ["tests/test_markdown.py"],
        "rich_3469": ["tests/test_markdown.py"],
        "rich_3472": ["tests/test_pretty.py"],
        "rich_4079": ["tests/test_markdown.py"],
        "rich_3006": ["tests/test_repr.py"],
        "rich_3067": ["tests/test_highlighter.py"],
        "fastapi_14479": ["tests/test_invalid_sequence_param.py"],
        "fastapi_14786": ["tests/test_security_http_base.py"],
        "fastapi_15588": ["tests/test_sse.py"],
    }
    for fb in fallbacks.get(issue, []):
        if (work / fb).is_file() and fb not in ordered:
            ordered.append(fb)

    for tf in ordered[:3]:
        cmd_str = f"PYTHONPATH=src pytest -q {tf}" if issue.startswith("requests_") else f"pytest -q {tf}"
        argv = [py, "-m", "pytest", "-q", tf]
        return cmd_str, argv
    return None


def build_executed_disjoint_states(tokenizer, tasks_by_id: dict, runs: list[Path]) -> list[dict]:
    """Execute real pytest checks on champion's returned patches (fail) & published dev fixes (pass) for disjoint snapshots."""
    # Find a verified template request from discovery02 to guarantee 100% native token rendering
    d2_reqs = [json.loads(s) for s in (ROOT / "experiments/discovery02/requests.jsonl").read_text().splitlines() if s.strip()]
    d2_exact = {
        r["request_sha"]: r["tokens"]
        for r in map(json.loads, (ROOT / "experiments/discovery02/replay_tokens.jsonl").read_text().splitlines())
    }
    sample_req = d2_reqs[0]
    sample_sha = hashlib.sha256(json.dumps(sample_req, sort_keys=True).encode()).hexdigest()
    assert render_chat(tokenizer, sample_req, sample_req["messages"]) == d2_exact[sample_sha], "Chat renderer mismatch!"

    # Gather initial user request & returned patch per disjoint issue
    init_reqs = {}
    returned_patches = {}
    for run in runs:
        if not (run / "requests.jsonl").is_file() or not (run / "diagnostics.zip").is_file():
            continue
        rlist = [json.loads(s) for s in (run / "requests.jsonl").read_text().splitlines() if s.strip()]
        for iss in DISCOVERY_19:
            if iss in tasks_by_id and iss not in init_reqs:
                ps = tasks_by_id[iss]["problem_statement"]
                for r in rlist:
                    if len(r.get("messages", [])) == 2 and ps[:120] in str(r["messages"][0].get("content", "")):
                        init_reqs[iss] = r
                        break
        with zipfile.ZipFile(run / "diagnostics.zip") as dz:
            for n in dz.namelist():
                if "/patches/" in n and n.endswith(".patch"):
                    iid = Path(n).stem.removesuffix("_returned")
                    if iid in DISCOVERY_19:
                        p_txt = dz.read(n).decode("utf-8", errors="replace")
                        if p_txt.strip():
                            returned_patches[iid] = p_txt

    extra_items = []
    env_base = dict(os.environ)

    for iss in sorted(DISCOVERY_19):
        snap = DATA / "snapshots" / f"{iss}.tgz"
        if not snap.is_file() or iss not in init_reqs or iss not in tasks_by_id:
            continue
        task = tasks_by_id[iss]
        req0 = init_reqs[iss]

        for mode, patch_txt, src_tag in (
            ("fail", returned_patches.get(iss, ""), "exec_disjoint_fail"),
            ("pass", task.get("patch", ""), "exec_disjoint_pass"),
        ):
            if not patch_txt.strip():
                continue
            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                with tarfile.open(snap) as tar:
                    tar.extractall(work, filter="data")
                files = [ln[len("+++ b/") :].strip() for ln in patch_txt.splitlines() if ln.startswith("+++ b/")]
                files = [f for f in files if (work / f).is_file()]
                if not files:
                    continue
                originals = {f: (work / f).read_text(errors="replace") for f in files}
                ap = subprocess.run(
                    ["git", "apply", "--whitespace=nowarn", "-"],
                    input=patch_txt if patch_txt.endswith("\n") else patch_txt + "\n",
                    text=True, cwd=work, capture_output=True,
                )
                if ap.returncode != 0:
                    continue
                tf_info = find_test_file_for_patch(work, patch_txt, iss)
                if not tf_info:
                    continue
                cmd_str, argv = tf_info
                env = dict(env_base, PYTHONPATH=str(work / "src")) if iss.startswith("requests_") else env_base
                try:
                    pr = subprocess.run(argv, cwd=work, env=env, text=True, capture_output=True, timeout=15)
                except subprocess.TimeoutExpired:
                    continue

                if mode == "fail" and pr.returncode not in (1, 2):
                    continue
                if mode == "pass" and (pr.returncode != 0 or "passed" not in pr.stdout):
                    continue

                msgs = copy.deepcopy(req0["messages"])
                ok_build = True
                last_fp = files[-1]
                for f in files[:2]:
                    a = originals[f].splitlines(keepends=True)
                    b = (work / f).read_text(errors="replace").splitlines(keepends=True)
                    ops = [e for e in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if e[0] != "equal"]
                    if not ops:
                        ok_build = False
                        break
                    s_i = max(0, ops[0][1] - 2)
                    e_i = min(len(a), ops[-1][2] + 2)
                    old_s = "".join(a[s_i:e_i])
                    new_s = "".join(b[ops[0][3] - (ops[0][1] - s_i) : ops[-1][4] + e_i - ops[-1][2]])
                    if not old_s or old_s == new_s:
                        ok_build = False
                        break
                    extend_turn(
                        msgs, "read_file",
                        {"filepath": f, "start_line": s_i + 1, "end_line": e_i},
                        {"status": "ok", "filepath": f, "content": old_s, "is_truncated": False},
                    )
                    delta = "".join(difflib.unified_diff(a, b, fromfile="a/" + f, tofile="b/" + f))
                    extend_turn(
                        msgs, "edit_file",
                        {"filepath": f, "old_string": old_s, "new_string": new_s, "allow_multiple": False},
                        {"status": "ok", "filepath": f, "occurrences": 1, "strategy": "exact", "diff": delta[:1500]},
                    )
                if not ok_build:
                    continue

                if mode == "fail":
                    comb = (pr.stdout + "\n" + pr.stderr).strip()
                    cmd_resp = {
                        "status": "error",
                        "error_type": "CommandError",
                        "error_message": comb[-1200:],
                        "details": {
                            "stdout": pr.stdout[-1200:],
                            "stderr": pr.stderr[-1200:],
                            "exit_code": int(pr.returncode),
                        },
                    }
                    role = "true_imm_real_test_fail"
                else:
                    cmd_resp = {
                        "status": "ok",
                        "stdout": pr.stdout[-1200:],
                        "stderr": pr.stderr[-1200:],
                        "exit_code": 0,
                    }
                    role = "true_imm_test_pass"

                extend_turn(msgs, "run_command", {"command": cmd_str}, cmd_resp)
                tks = render_chat(tokenizer, req0, msgs)
                sha = hashlib.sha256(json.dumps(msgs, sort_keys=True).encode()).hexdigest()
                extra_items.append({
                    "issue": iss,
                    "true_role": role,
                    "filepath": last_fp,
                    "tokens": tks,
                    "prompt_sha": sha,
                    "source_run": src_tag,
                })
                print(f"  Built executed disjoint state: {iss:15s} | {role:24s} | cmd={cmd_str} | ec={pr.returncode} | toks={len(tks)}", flush=True)
    return extra_items


def main():
    from adk_submission.server import _round_up_vllm_lora_rank
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tasks_all = [json.loads(s) for s in (DATA / "tasks.jsonl").read_text().splitlines() if s.strip()]
    tasks_by_id = {t["instance_id"]: t for t in tasks_all}

    # Strict snapshot-disjointness assertion against QUICK_30 and HOLDOUT!
    reserved_snaps = {(tasks_all[i]["repo"], tasks_all[i]["base_commit"]) for i in QUICK_30_IDX}
    reserved_snaps |= {(t["repo"], t["base_commit"]) for t in tasks_all if t["instance_id"] in HOLDOUT_IDS}
    disc_snaps = {(tasks_by_id[i]["repo"], tasks_by_id[i]["base_commit"]) for i in DISCOVERY_19}
    assert not (disc_snaps & reserved_snaps), f"FATAL BENCHMARK LEAKAGE: {disc_snaps & reserved_snaps}"
    print(f"Verified 100% snapshot disjointness: {len(disc_snaps)} discovery snapshots vs {len(reserved_snaps)} reserved QUICK_30+HOLDOUT!", flush=True)

    assets = json.loads((ROOT / "assets.json").read_text())
    tok_hf = AutoTokenizer.from_pretrained(assets["model"], local_files_only=True)

    prev_nat = {}
    for p in (
        ROOT / "harvest32/clean_v20/real_replays.json",
        ROOT / "harvest32/clean_v14/champion_clean.json",
        ROOT / "harvest32/native/champion.json",
    ):
        if p.is_file():
            for r in json.loads(p.read_text()):
                if r.get("text"):
                    act, args = parse_gemma4_call(r["text"])
                    prev_nat[r["prompt_sha"]] = {"action": act, "arguments": args, "text": r["text"]}

    all_items = {}
    v20_p = ROOT / "harvest32/clean_v20/real_replays.json"
    if v20_p.is_file():
        for c in json.loads(v20_p.read_text()):
            assert c["issue"] in DISCOVERY_19 or str(c["issue"]).startswith("disc_"), f"Unexpected issue {c['issue']}"
            sr = classify_v21_role(tok_hf, c["tokens"], c.get("source_run", ""))
            if sr in (
                "true_imm_edit_ok", "true_imm_real_test_fail", "true_imm_bad_test_path",
                "true_imm_test_pass", "true_imm_edit_err", "true_imm_read_full",
                "true_imm_read_trunc", "turn1_init",
            ):
                all_items[c["prompt_sha"]] = {
                    "issue": c["issue"],
                    "true_role": sr,
                    "filepath": c.get("filepath"),
                    "tokens": c["tokens"],
                    "prompt_sha": c["prompt_sha"],
                    "source_run": c.get("source_run", "v20"),
                }

    # Add exact_verified cases from wsp_31 if not already included
    ver_p = ROOT / "preservation/exact/cases.json"
    if ver_p.is_file():
        for vc in json.loads(ver_p.read_text()):
            assert vc["issue"] in DISCOVERY_19
            all_items[vc["prompt_sha"]] = {
                "issue": vc["issue"],
                "true_role": "true_imm_test_pass",
                "filepath": vc.get("filepath"),
                "tokens": vc["tokens"],
                "prompt_sha": vc["prompt_sha"],
                "source_run": "exact_verified",
            }

    # Scan discovery02/03/04 replay_tokens.jsonl for any missed post-edit / post-test turns
    disc_ps = [tasks_by_id[i]["problem_statement"] for i in DISCOVERY_19]
    disc_needles = {}
    for iid in DISCOVERY_19:
        cands = [ln.strip() for ln in tasks_by_id[iid]["problem_statement"].splitlines() if len(ln.strip()) >= 8]
        uniq = [ln for ln in cands if sum(1 for o in disc_ps if ln in o) == 1] or cands
        disc_needles[iid] = max(uniq, key=len)

    runs = [ROOT / "experiments/discovery02", ROOT / "experiments/discovery03", ROOT / "experiments/discovery04"]
    for run in runs:
        rp = run / "replay_tokens.jsonl"
        if not rp.is_file():
            continue
        for ln in rp.read_text().splitlines():
            if not ln.strip():
                continue
            o = json.loads(ln)
            sha = o["request_sha"]
            if sha in all_items:
                continue
            sr = classify_v21_role(tok_hf, o["tokens"])
            if sr in ("true_imm_edit_ok", "true_imm_real_test_fail", "true_imm_bad_test_path", "true_imm_test_pass"):
                txt_full = tok_hf.decode(o["tokens"], skip_special_tokens=False)
                iss = next((iid for iid, ndl in disc_needles.items() if ndl in txt_full), None)
                if not iss:
                    continue
                m_fp = re.findall(r'"?filepath"?\s*:\s*(?:<\|"\|>|")([^"<]+)', txt_full)
                all_items[sha] = {
                    "issue": iss,
                    "true_role": sr,
                    "filepath": m_fp[-1] if m_fp else None,
                    "tokens": o["tokens"],
                    "prompt_sha": sha,
                    "source_run": run.name,
                }

    # Augment with real executed disjoint fail/pass histories across DISCOVERY_19!
    exec_extra = build_executed_disjoint_states(tok_hf, tasks_by_id, runs)
    for item in exec_extra:
        if item["prompt_sha"] not in all_items:
            all_items[item["prompt_sha"]] = item

    need = [v for sha, v in all_items.items() if sha not in prev_nat]
    print(f"V21 Disjoint Corpus: {len(all_items)} total items, {len(need)} new prompts to replay natively with champion_r1...", flush=True)

    if need:
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
            prev_nat[c["prompt_sha"]] = {"action": act, "arguments": args, "text": txt}

    merged = []
    for sha, v in all_items.items():
        nr = prev_nat.get(sha, {})
        merged.append({
            **v,
            "action": nr.get("action"),
            "arguments": nr.get("arguments") or {},
            "text": nr.get("text") or "",
        })

    out_d = ROOT / "harvest32/clean_v21"
    out_d.mkdir(parents=True, exist_ok=True)
    (out_d / "real_replays.json").write_text(json.dumps(merged))

    print("Saved clean_v21/real_replays.json! Role x Champion Action table:", flush=True)
    for (rl, act), cnt in sorted(Counter((x["true_role"], str(x["action"])) for x in merged).items()):
        print(f"  {rl:25s} | {act:15s} : {cnt}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    main()
