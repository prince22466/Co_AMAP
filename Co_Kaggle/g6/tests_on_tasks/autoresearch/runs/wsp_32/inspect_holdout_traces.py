"""Inspect exact tool-call trajectories and patches across holdout_champ, holdout_ml2_35_41_hi, and holdout_v3_best."""
from provision_and_step2 import remote_exec


def main():
    out = remote_exec("""
import json, zipfile
for rname in ('holdout_champ', 'holdout_ml2_35_41_hi', 'holdout_v3_best'):
    print('=== ' + rname + ' ===')
    with zipfile.ZipFile(f'/content/wsp/experiments/{rname}/diagnostics.zip') as z:
        for iss in ('rich_3043', 'rich_3064', 'rich_2725'):
            ev_p = next(n for n in z.namelist() if n.endswith(f'/observer/{iss}/events.jsonl'))
            evs = [json.loads(l) for l in z.read(ev_p).splitlines() if l.strip()]
            seq = []
            for e in evs:
                for c in e.get('calls') or []:
                    fn = c['function_name']
                    ag = c.get('arguments') or {}
                    if fn == 'run_command':
                        seq.append(f"cmd({ag.get('command','')[:70]})")
                    elif fn == 'read_file':
                        seq.append(f"read({ag.get('filepath','')}:{ag.get('start_line')}-{ag.get('end_line')})")
                    elif fn == 'edit_file':
                        seq.append(f"edit({ag.get('filepath','')})")
                    else:
                        seq.append(fn)
                for r in e.get('responses') or []:
                    d = r.get('response') or {}
                    if isinstance(d, str):
                        try:
                            d = json.loads(d)
                        except Exception:
                            d = {}
                    if r['function_name'] == 'edit_file':
                        seq.append('->' + str(d.get('status')))
                    elif r['function_name'] == 'run_command':
                        seq.append('->rc=' + str(d.get('exit_code', d.get('returncode'))))
            print(f'  {iss}:', ' '.join(seq))
""", timeout=90)
    print(out, flush=True)


if __name__ == "__main__":
    main()
