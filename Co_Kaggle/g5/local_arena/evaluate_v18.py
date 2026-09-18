"""Screen policies against both saved loss sets and live v17."""
import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from bench_v16 import run

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidate', default='submission_nb/v18_main.py')
    p.add_argument('--output', required=True)
    p.add_argument('--episodes', default='')
    p.add_argument('--seeds', default='')
    p.add_argument('--diagnostics', action='store_true')
    p.add_argument('--jobs', type=int, default=4)
    a = p.parse_args()
    jobs = []
    if not a.seeds:
        for path in sorted(Path('game_history').glob('v1[67]/*.json')):
            if a.episodes and path.stem not in a.episodes.split(','): continue
            jobs.append((a.candidate, str(path), 0, 0, None, a.diagnostics))
    else:
        for seed in map(int, a.seeds.split(',')):
            for seat in (0, 1):
                jobs.append((a.candidate, None, seat, seed, 'submission_nb/kaggriculture-sub_v17.ipynb', a.diagnostics))
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for f in as_completed([pool.submit(run,j) for j in jobs]):
            r = f.result(); rows.append(r)
            print(r['episode'] or str(r['seed'])+':'+str(r['seat']), r['rewards'], r['margin'], flush=True)
            Path(a.output).write_text(json.dumps({'matches':rows},indent=2))
    print('wins',sum(r['margin']>0 for r in rows),'/',len(rows),'mean',sum(r['margin'] for r in rows)/len(rows),flush=True)

if __name__ == '__main__': main()
