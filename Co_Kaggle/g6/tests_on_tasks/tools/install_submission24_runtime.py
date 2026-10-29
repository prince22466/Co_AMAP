"""Install CPU-only test dependencies from the official, hash-checked wheelhouse.

Run in a disposable venv or CI runner. No model weights or GPU runtime are installed.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

WHEELS = {
    'adk_eval_core-0.1.0-py3-none-any.whl': 'e614e1b2fe8d886a634871e4c57f9887e1cc66e8c27300f8503a43c6e3ca82f3',
    'adk_submission-0.2.13-py3-none-any.whl': 'f156f0bd84580ccb9334726d114d7a5481f16d3a6e2887cc4d276ced0ef8d99d',
    'swegemma-0.2.11-py3-none-any.whl': '9ad876581f3acde8ec3c369858f0834e2b9f41a3cffce3ca42f09cb090aea9d3',
    'google_adk-1.36.1-py3-none-any.whl': '1a2f6868c509e3151fb0de3575a7d18b45c338be86f420924dad74e7193631a0',
    'google_genai-2.11.0-py3-none-any.whl': '5bc8186100e1d34d691fbe0cba392b7e04e98d286ca952323a6672d054accf95',
}
BASE = 'https://www.kaggle.com/api/v1/datasets/download/metric/gemma-4-developer-agent-wheelhouse/'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel-dir', type=Path, required=True)
    parser.add_argument('--download-only', action='store_true')
    args = parser.parse_args()
    args.wheel_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, expected in WHEELS.items():
        path = args.wheel_dir / name
        if not path.exists():
            subprocess.run(['curl', '-fLsS', '--retry', '2', '--max-time', '60', BASE + name, '-o', str(path)], check=True)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Official wheel hash mismatch: {name}: {actual}')
        paths.append(str(path))
        print(name, actual, flush=True)
    if args.download_only:
        return
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', *paths[:3]], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', *paths[3:],
        'litellm==1.82.4', 'docker>=7', 'networkx>=3', 'numpy>=2',
        'pandas>=2', 'python-dotenv', 'rich', 'tokenizers>=0.20'], check=True)


if __name__ == '__main__':
    main()
