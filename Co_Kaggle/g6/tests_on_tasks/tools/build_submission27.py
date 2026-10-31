"""Deterministically package S27's declarative E4B prompt experiment."""
import hashlib
import json
from pathlib import Path
import zipfile

import yaml

G6 = Path(__file__).resolve().parents[2]
SOURCE = G6 / 'submission_nb/submission_27'
ARCHIVE = G6 / 'submission_nb/artifacts/submission_27.zip'
MODEL = 'gemma-4-e4b-it-qat-w4a16-ct'
FILES = ('agent.yaml', 'eval_config.yaml', 'instruction.md')
TOOLS = ('run_command', 'read_file', 'edit_file', 'write_file', 'get_status', 'submit_patch')


class IncludeLoader(yaml.SafeLoader):
    pass


IncludeLoader.add_constructor('!include', lambda loader, node: loader.construct_scalar(node))


def build():
    payload = {}
    for name in FILES:
        path = SOURCE / name
        if path.is_symlink() or not path.resolve().is_relative_to(SOURCE.resolve()):
            raise ValueError(f'Unsafe package path: {name}')
        payload[name] = path.read_bytes()
    config = yaml.load(payload['agent.yaml'], Loader=IncludeLoader)
    limits = yaml.safe_load(payload['eval_config.yaml'])
    assert config.get('agent_class', 'LlmAgent') == 'LlmAgent'
    assert config['model'] == MODEL and 'adapter' not in config
    assert config['include_contents'] == 'default'
    assert config['instruction'] == 'instruction.md'
    assert tuple(config['tools']) == TOOLS
    assert not config.get('skills') and not config.get('sub_agents')
    assert config['generate_content_config'] == {
        'temperature': 0.0, 'top_p': 0.95, 'max_output_tokens': 4096,
        'thinking_config': {'thinking_budget': 512, 'include_thoughts': False}}
    assert limits['evaluation'] == {'timeout_seconds': 30, 'max_tool_calls': 40,
                                    'max_time_minutes': 4, 'max_turns': 48}
    instruction = payload['instruction.md'].decode('utf-8')
    assert '{problem_description}' in instruction and '{hints?}' in instruction
    assert len(instruction) < 3800
    assert all(s in instruction for s in ('EVIDENCE & PROGRESS MANAGER',
        'REPETITION DETECTOR', 'PROGRESS CONTROLLER', 'submit_patch'))

    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE, 'w') as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    manifest = {'model': MODEL,
                'archive_sha256': hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
                'files': {p: hashlib.sha256(data).hexdigest() for p, data in sorted(payload.items())},
                'scope': 'Unscored E4B prompt-only experiment; no model task performance measured.'}
    ARCHIVE.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
