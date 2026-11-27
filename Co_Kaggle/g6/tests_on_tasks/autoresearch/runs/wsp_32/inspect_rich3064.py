"""Inspect exact patch and pytest output on rich_3064 in holdout_ml2_35_41_hi vs reference patch."""
from provision_and_step2 import remote_exec


def main():
    out = remote_exec("""
import json, zipfile
from pathlib import Path
tasks = {r['instance_id']: r for r in map(json.loads, Path('/content/wsp/input/competitions/gemma-4-developer-agent/tasks.jsonl').read_text().splitlines())}
with zipfile.ZipFile('/content/wsp/experiments/holdout_ml2_35_41_hi/diagnostics.zip') as z:
    for iss in ('rich_3064', 'rich_2725'):
        p_name = next((n for n in z.namelist() if n.endswith(f'/patches/{iss}_returned.patch')), None)
        print(f'=== {iss} RETURNED PATCH ===')
        print(z.read(p_name).decode('utf-8', errors='replace') if p_name else 'NONE')
        print(f'=== {iss} REF PATCH ===')
        print(tasks[iss]['patch'][:1000])
    ev_p = next(n for n in z.namelist() if n.endswith('/observer/rich_3064/events.jsonl'))
    for l in z.read(ev_p).splitlines():
        e = json.loads(l)
        for r in e.get('responses') or []:
            if r['function_name'] == 'run_command':
                print('CMD RESP:', str(r.get('response'))[:500])
""", timeout=60)
    print(out, flush=True)


if __name__ == "__main__":
    main()
