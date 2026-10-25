"""Build submission 16 reproducibly. Does not run model inference."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import zipfile
import yaml

G6 = Path(__file__).resolve().parents[2]
SOURCE = G6 / 'submission_nb/submission_16'
ARCHIVE = G6 / 'submission_nb/artifacts/submission_16.zip'
MODEL = 'gemma-4-12b-it-qat-w4a16-ct'


def entries():
    result = {}
    for p in sorted(SOURCE.rglob('*')):
        if '__pycache__' in p.parts or p.name == 'README.md' or not p.is_file():
            continue
        if p.is_symlink() or p.suffix not in {'.yaml', '.md', '.py'}:
            raise ValueError(f'Unexpected release file: {p}')
        name, data = p.relative_to(SOURCE).as_posix(), p.read_bytes()
        if p.suffix == '.py':
            ast.parse(data, filename=name)
        if re.search(rb'(?:fastapi|requests|rich|httpx)_\d{3,}', data):
            raise ValueError(f'Task-specific content in release: {name}')
        result[name] = data
    # This is structural validation; official compiler acceptance is checked separately.
    class Loader(yaml.SafeLoader):
        pass
    Loader.add_constructor('!include', lambda loader, node: loader.construct_scalar(node))
    configs = {n: yaml.load(d, Loader=Loader) for n,d in result.items() if n.endswith('.yaml')}
    assert configs['agent.yaml']['agent_class'] == 'SequentialAgent'
    assert configs['sub_agents/repair_loop.yaml']['max_iterations'] == 2
    llms = {n:c for n,c in configs.items() if 'model' in c}
    assert len(llms) == 3 and {c['model'] for c in llms.values()} == {MODEL}
    assert [n for n,c in llms.items() if 'submit_patch' in c['tools']] == ['sub_agents/finalizer.yaml']
    assert all(c['include_contents'] == 'none' for c in llms.values())
    assert all(c['generate_content_config']['thinking_config']['thinking_budget'] == 0 for c in llms.values())
    base = result['skills/repo-navigation/scripts/workbench.py']
    for role, name in [('verifier', 'repo-verification'), ('finalizer', 'repo-finalization')]:
        assert result[f'skills/{name}/scripts/workbench.py'] == base.replace(b"ROLE = 'coder'", f"ROLE = '{role}'".encode())
    assert configs['eval_config.yaml']['evaluation']['max_time_minutes'] == 4
    assert result['skills/repo-navigation/SKILL.md'].startswith(b'---\nname: repo-navigation\n')
    return result


def build(archive=ARCHIVE):
    payload = entries()
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in payload.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, data)
    manifest = {'model': MODEL, 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                'files': {n: hashlib.sha256(d).hexdigest() for n,d in payload.items()},
                'scope': 'Packaging validation only; no model inference or benchmark score.',
                'acceptance_target': 'At least 2 newly resolved tasks on the same first 10; not yet measured.'}
    archive.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ARCHIVE)
    print(json.dumps(build(p.parse_args().output), indent=2))
