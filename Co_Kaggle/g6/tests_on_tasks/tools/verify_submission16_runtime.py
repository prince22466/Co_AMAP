"""Run the packaged ZIP through the official compiler, Runner and skill executor."""
import argparse
import asyncio
import json
from pathlib import Path
import tempfile
import zipfile
from build_submission16 import ARCHIVE, G6
from smoke_submission16 import run_smoke


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=ARCHIVE)
    parser.add_argument('--output-dir',type=Path,default=G6/'tests_on_tasks/reviews/submission_16/runtime')
    args=parser.parse_args()
    with tempfile.TemporaryDirectory() as d:
        with zipfile.ZipFile(args.archive) as z:
            assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
            z.extractall(d)
        report=asyncio.run(run_smoke(Path(d),args.output_dir))
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
