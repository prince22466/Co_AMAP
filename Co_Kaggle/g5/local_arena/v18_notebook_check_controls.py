"""Validate that the installed engine reproduces both saved history sets."""
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from evaluate_v17_history import control

if __name__ == '__main__':
    paths = sorted(Path('game_history').glob('v1[67]/*.json'))
    rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(control, str(p)) for p in paths]):
            row = future.result()
            rows.append(row)
            print(row['episode'], row['exact_rewards'], row['observation_mismatches'], flush=True)
    Path('local_arena/v18_notebook_check/controls.json').write_text(json.dumps(rows, indent=2))
