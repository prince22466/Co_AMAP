"""Orchestrate V4 downstream python3-c synthesis, native vLLM screen, and HOLDOUT verification on Colab wsp32."""
import json
import subprocess

from provision_and_step2 import CLI, HERE, SESSION, remote_exec, upload_verified
from run_step3_and_step4 import run_remote_stage


def main():
    upload_verified(HERE / "remote_synth_v4.py", "/content/wsp/synth_v4.py")

    remote_exec("""
from pathlib import Path
code = Path('/content/wsp/screen_v3.py').read_text()
code = code.replace('v3_candidates', 'v4_candidates').replace('screen_v3_r1', 'screen_v4_r1')
Path('/content/wsp/screen_v4.py').write_text(code)
""")

    py_base = "python3"
    py_eval = "/content/wsp/eval_env/bin/python"

    print("=== 1. Synthesizing V4 downstream L39/L40 python3-c targeter ===", flush=True)
    run_remote_stage("synth_v4", f"{py_base} -u /content/wsp/synth_v4.py", timeout_sec=600)
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/v4_candidates/manifest.json", str(HERE / "v4_manifest.json")], check=True)
    print((HERE / "v4_manifest.json").read_text(encoding="utf-8"), flush=True)

    print("=== 2. Screening V4 candidates in native vLLM ===", flush=True)
    run_remote_stage(
        "screen_v4",
        f"{py_eval} -u /content/wsp/screen_v4.py --harvest-dir /content/wsp/harvest32 --runs /content/wsp/experiments/discovery02 /content/wsp/experiments/discovery03",
        timeout_sec=1800,
    )
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/screen_v4_r1/summary.json", str(HERE / "screen_v4_summary.json")], check=True)

    remote_exec("""
import json
from pathlib import Path
out = {}
d = Path('/content/wsp/harvest32/screen_v4_r1')
for p in sorted(d.glob('*.json')):
    if p.name == 'summary.json':
        continue
    rows = json.loads(p.read_text())
    unch = [r for r in rows if r.get('role') == 'unchecked']
    if p.stem.endswith('_candidate') or p.stem == 'champion_r1':
        out[p.stem] = [
            {
                'issue': r['issue'],
                'action': r.get('action'),
                'command': r.get('command'),
                'is_bare_pytest': r.get('is_bare_pytest'),
                'defect_catch': r.get('discriminating_defect_catch'),
                'agent_exit': r.get('on_returned_patch_exit'),
                'ref_exit': r.get('on_reference_fix_exit'),
                'out_tail': r.get('agent_out_tail', '')[-180:],
            }
            for r in unch
        ]
Path('/content/wsp/harvest32/screen_v4_r1/unchecked_details.json').write_text(json.dumps(out, indent=2))
""")
    subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/harvest32/screen_v4_r1/unchecked_details.json", str(HERE / "screen_v4_unchecked_details.json")], check=True)
    print("=== V4 SUMMARY ===", flush=True)
    print((HERE / "screen_v4_summary.json").read_text(encoding="utf-8"), flush=True)
    print("=== V4 UNCHECKED DETAILS ===", flush=True)
    print((HERE / "screen_v4_unchecked_details.json").read_text(encoding="utf-8"), flush=True)

    v4_sum = json.loads((HERE / "screen_v4_summary.json").read_text(encoding="utf-8"))
    cands = [r for r in v4_sum if r["adapter"].endswith("_candidate") and r["verified_preserved"] == "3/3" and r["localize_preserved"] == "13/13"]
    cands.sort(
        key=lambda r: (
            r["unchecked_defect_catches"],
            r["edit_preserved_vs_champ"] == "4/4",
            r["unchecked_targeted_flips"],
            -r["unchecked_bare_pytest"],
            -r["unchecked_submits_left"],
        ),
        reverse=True,
    )
    best = cands[0]
    ad_path = f"/content/wsp/harvest32/screen_v4_r1/adapters/{best['adapter']}"
    sha_txt = remote_exec(
        f"import hashlib; from pathlib import Path; print('SHA=' + hashlib.sha256((Path({ad_path!r})/'adapter_model.safetensors').read_bytes()).hexdigest())"
    )
    best_sha = next(l.split("=")[1].strip() for l in sha_txt.splitlines() if l.startswith("SHA="))
    print(f"\n=== Running HOLDOUT on top V4 finalist: {best['adapter']} (sha={best_sha[:12]}) ===", flush=True)
    run_remote_stage(
        "eval_holdout_v4_best",
        f"{py_eval} -u /content/wsp/run_evaluator_v2.py --split holdout --adapter {ad_path} --sha {best_sha} --name holdout_v4_best",
        timeout_sec=1800,
    )
    out_local = HERE / "evals/holdout_v4_best"
    out_local.mkdir(parents=True, exist_ok=True)
    for fn in ("neon_metrics.json", "neon_failures.jsonl"):
        subprocess.run([CLI, "download", "-s", SESSION, f"/content/wsp/experiments/holdout_v4_best/{fn}", str(out_local / fn)], check=True)
    print("HOLDOUT V4 BEST METRICS:", (out_local / "neon_metrics.json").read_text(encoding="utf-8"), flush=True)
    print("HOLDOUT V4 BEST FAILURES:", (out_local / "neon_failures.jsonl").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
