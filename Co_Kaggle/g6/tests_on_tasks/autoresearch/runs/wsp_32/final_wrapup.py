"""Final wrap-up for wsp_32: archive all remote research artifacts locally, append QUICK_15 ledger rows, stop Colab GPU."""
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(r"e:\Projects\Co_AMAP\Co_Kaggle\g6\tests_on_tasks\autoresearch\runs\wsp_32")
sys.path.insert(0, str(HERE))
from provision_and_step2 import CLI, RESEARCH, SESSION, remote_exec

# 1. Bundle remote artifacts (small files only; no base model / wheels / snapshots)
print(remote_exec(r"""
import zipfile
from pathlib import Path
root = Path('/content/wsp')
out = root / 'wsp32_final_handoff.zip'
if out.exists():
    out.unlink()
include_dirs = [
    'harvest32/clean_v21', 'harvest32/v21_fsm', 'harvest32/v22_cauchy', 'harvest32/v23_sweetspot',
    'harvest32/v24_matched39_41', 'harvest32/screen_v21_fsm', 'harvest32/screen_v22_cauchy',
    'harvest32/screen_v23_sweetspot', 'harvest32/screen_v24_matched',
]
include_files = [
    'remote_funnels.json', 'v21_disjoint6_report.json', 'v22_disjoint6_report.json',
    'v23_disjoint6_report.json', 'v24_disjoint6_report.json',
    'build_v21_disjoint_corpus.py', 'remote_synth_v21_fsm.py', 'remote_synth_v22_cauchy.py',
    'remote_synth_v24_matched39_41.py', 'screen_v21_fsm.py', 'screen_v22_cauchy.py',
    'screen_v23_sweetspot.py', 'screen_v24_matched.py', 'event_funnel_analyzer.py',
]
n = 0
with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for d in include_dirs:
        p = root / d
        if p.is_dir():
            for f in p.rglob('*'):
                if f.is_file():
                    z.write(f, f.relative_to(root).as_posix()); n += 1
    for f in include_files:
        p = root / f
        if p.is_file():
            z.write(p, f); n += 1
    for exp in ('discovery02', 'discovery03', 'discovery04', 'wsp32_v23_disjoint6', 'wsp32_v24_disjoint6',
                'wsp32_champ_rep2', 'wsp32_champ_rep3', 'wsp32_iter05', 'wsp32_iter06', 'wsp32_iter07'):
        for fn in ('neon_metrics.json', 'neon_failures.jsonl', 'diagnostics.zip'):
            p = root / 'experiments' / exp / fn
            if p.is_file():
                z.write(p, f'experiments/{exp}/{fn}'); n += 1
print('BUNDLED_FILES', n, 'BYTES', out.stat().st_size)
""", timeout=300))

local_zip = HERE / "checkpoints" / "wsp32_final_handoff.zip"
if local_zip.exists():
    local_zip.unlink()
subprocess.run([CLI, "download", "-s", SESSION, "/content/wsp/wsp32_final_handoff.zip", str(local_zip)], check=True)
with zipfile.ZipFile(local_zip) as z:
    assert z.testzip() is None, "Corrupt handoff zip"
    n_members = len(z.namelist())
sha = hashlib.sha256(local_zip.read_bytes()).hexdigest()
print(f"LOCAL_HANDOFF_ZIP {local_zip} members={n_members} bytes={local_zip.stat().st_size} sha256={sha}")

# 2. Append QUICK_15 ledger rows for wsp32 runs (skip rows already present)
tsv = RESEARCH / "results.tsv"
existing = tsv.read_text(encoding="utf-8")
notes = {
    "wsp32_champ_cal": ("CALIBRATION", "Unmodified champion iter_04 on pinned Colab L4 (repeat 1)."),
    "wsp32_champ_rep2": ("CALIBRATION", "Unmodified champion iter_04 on pinned Colab L4 (repeat 2)."),
    "wsp32_champ_rep3": ("CALIBRATION", "Unmodified champion iter_04 on pinned Colab L4 (repeat 3, solo)."),
    "wsp32_iter01": ("DISCARD", "V4 pure-WSP candidate."),
    "wsp32_iter02": ("DISCARD", "V8 pure-WSP candidate."),
    "wsp32_iter03": ("DISCARD", "V12 pure-WSP candidate."),
    "wsp32_iter04": ("DISCARD", "V16 upper-band o_proj+down_proj candidate."),
    "wsp32_iter05": ("DISCARD", "V19 L39+L41 down_proj (no L40); best funnel s2=8 s3=5 but 4/15."),
    "wsp32_iter06": ("DISCARD", "V19 L39+L40+L41 balanced; 4/15, leaks=2."),
    "wsp32_iter07": ("DISCARD", "V20 cascade L39->L40->L41; 4/15, leaks=3."),
}
rows = []
for run, (decision, hyp) in notes.items():
    if f"\n{run}\t" in existing:
        continue
    mp = HERE / "evals" / run / "neon_metrics.json"
    if not mp.is_file():
        continue
    m = json.loads(mp.read_text(encoding="utf-8"))
    hyp_full = f"{hyp} Solved: {', '.join(m['resolved_ids'])}."
    rows.append("\t".join(str(x) for x in (
        run, "none", m["adapter_sha256"][:12], m.get("rank", 1), m["split"], m["solved"],
        m["file_edit_errors_total"], m["premature_submits"], m["syntax_leaks"], m["avg_duration_seconds"],
        decision, "none", hyp_full,
    )))
if rows:
    with tsv.open("a", encoding="utf-8") as f:
        if not existing.endswith("\n"):
            f.write("\n")
        f.write("\n".join(rows) + "\n")
print(f"APPENDED {len(rows)} ledger rows to {tsv}")

# 3. Stop Colab GPU and verify
subprocess.run([CLI, "stop", "-s", SESSION], check=False)
st = subprocess.run([CLI, "status"], capture_output=True, text=True)
print("COLAB_STATUS_AFTER_STOP:\n", st.stdout, st.stderr)
