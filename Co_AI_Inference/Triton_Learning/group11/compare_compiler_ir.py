#!/usr/bin/env python3
"""Produce a readable unified diff of two Group11 compiler stages; no GPU needed."""
from __future__ import annotations
import argparse
import difflib
import json
from pathlib import Path

STAGE_PRIORITY = ('amdgcn','sass','ptx','llir','ttgir','ttir')


def compare(root: Path, left: str, right: str, stage: str | None = None) -> str:
    def metadata(stem):
        path = root / f'{stem}.stages.json'
        if not path.exists():
            raise FileNotFoundError(f'Missing {path}; run the benchmark with --dump-asm')
        return json.loads(path.read_text(encoding='utf-8'))
    a=metadata(left); b=metadata(right)
    if stage is None:
        stage=next((s for s in STAGE_PRIORITY if s in a and s in b and
                    a[s].get('kind')=='text' and b[s].get('kind')=='text'),None)
        if stage is None:
            raise ValueError(f'No common text stage (left={list(a)}, right={list(b)})')
    if stage not in a or stage not in b:
        raise ValueError(f'No common {stage} stage')
    if a[stage].get('kind') != 'text' or b[stage].get('kind') != 'text':
        raise ValueError('Cannot text-diff a binary stage; compare its SHA256 in CSV')
    x=root/f'{left}.{stage}.txt'
    y=root/f'{right}.{stage}.txt'
    old=x.read_text(encoding='utf-8').splitlines(keepends=True)
    new=y.read_text(encoding='utf-8').splitlines(keepends=True)
    prefix=(f'# Group11 compiler diff: {left} vs {right}\n'
            f'# Stage: {stage} (textual; not necessarily native hardware assembly)\n'
            f'# SHA-256 A: {a[stage]["sha256"]}\n'
            f'# SHA-256 B: {b[stage]["sha256"]}\n'
            '# Different text does not necessarily mean different machine instructions.\n\n')
    return prefix+''.join(difflib.unified_diff(old,new,fromfile=str(x),tofile=str(y),n=3))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dir',required=True,type=Path,help='Directory supplied to --dump-asm')
    p.add_argument('--a',required=True,help='First kernel stem, e.g. K01_unroll1')
    p.add_argument('--b',required=True,help='Second kernel stem, e.g. K01_unroll4')
    p.add_argument('--stage',help='Text stage: ttir, ttgir, llir, or vendor ISA')
    p.add_argument('--out',help='Write unified diff to a text file')
    args=p.parse_args()
    result=compare(args.dir,args.a,args.b,args.stage)
    if args.out:
        path=Path(args.out); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(result,encoding='utf-8')
        print(f'Wrote {path}')
    else:
        print(result)

if __name__=='__main__': main()
