"""Build the exact E4B-only S25 package; no benchmark evaluation is implied."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

import yaml

G6 = Path(__file__).resolve().parents[2]
SOURCE = G6 / 'submission_nb/submission_25'
ARCHIVE = G6 / 'submission_nb/artifacts/submission_25.zip'
MODEL = 'gemma-4-e4b-it-qat-w4a16-ct'
FILES = (
    'agent.yaml', 'eval_config.yaml', 'instruction.md',
    'skills/repair/SKILL.md', 'skills/repair/scripts/work.py',
)


def build():
    payload = {}
    for name in FILES:
        path = SOURCE / name
        if path.is_symlink() or not path.resolve().is_relative_to(SOURCE.resolve()):
            raise ValueError('Unsafe package path: ' + name)
        payload[name] = path.read_bytes()
        if path.suffix == '.py':
            ast.parse(payload[name], filename=name)
    class Loader(yaml.SafeLoader):
        pass
    Loader.add_constructor('!include', lambda loader, node: loader.construct_scalar(node))
    configs = {n: yaml.load(b, Loader=Loader) for n, b in payload.items() if n.endswith('.yaml')}
    root = configs['agent.yaml']
    assert root.get('agent_class', 'LlmAgent') == 'LlmAgent'
    assert root['model'] == MODEL and 'adapter' not in root
    assert root['include_contents'] == 'default'
    assert root['tools'] == ['get_status', 'submit_patch']
    assert root['skills'] == ['skills/repair']
    assert root['generate_content_config']['thinking_config']['thinking_budget'] == 0
    for name, config in configs.items():
        for child in config.get('sub_agents', []):
            assert (Path(name).parent / child['config_path']).as_posix() in payload
        if 'instruction' in config:
            assert (Path(name).parent / config['instruction']).as_posix() in payload
    assert configs['eval_config.yaml']['evaluation'] == dict(
        timeout_seconds=30, max_tool_calls=40, max_time_minutes=4, max_turns=48)
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE, 'w') as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    manifest = {
        'model': MODEL, 'archive_sha256': hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        'files': {n: hashlib.sha256(b).hexdigest() for n, b in sorted(payload.items())},
        'scope': 'Unscored candidate; CPU/compiler/scripted tests do not measure E4B task success.',
    }
    ARCHIVE.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
