"""Deterministic Submission 22 archive builder. Packaging tests are not model evaluation."""
import hashlib
import json
from pathlib import Path
import zipfile
import yaml

G6 = Path(__file__).resolve().parents[2]
SOURCE = G6 / 'submission_nb/submission_22'
ARCHIVE = G6 / 'submission_nb/artifacts/submission_22.zip'
MODEL = 'gemma-4-e4b-it-qat-w4a16-ct'
FILES = (
    'agent.yaml', 'eval_config.yaml', 'instruction.md',
    'skills/patch-audit/SKILL.md',
    'skills/patch-audit/scripts/audit.py',
    'skills/patch-audit/scripts/recover.py',
    'skills/patch-audit/scripts/verify.py',
)

def build():
    payload = {name: (SOURCE / name).read_bytes() for name in FILES}
    class Loader(yaml.SafeLoader):
        pass
    Loader.add_constructor('!include', lambda loader, node: loader.construct_scalar(node))
    agent = yaml.load(payload['agent.yaml'], Loader=Loader)
    assert agent['name'] == 'repair_agent'
    assert agent['model'] == MODEL
    assert agent['tools'] == [
        'get_status', 'read_file', 'edit_file', 'write_file', 'run_command', 'submit_patch'
    ]
    assert agent['skills'] == ['skills/patch-audit']
    assert not agent.get('sub_agents')
    assert agent['generate_content_config']['thinking_config']['thinking_budget'] == 0
    limits = yaml.safe_load(payload['eval_config.yaml'])['evaluation']
    assert limits == {
        'timeout_seconds': 30, 'max_tool_calls': 40,
        'max_time_minutes': 4, 'max_turns': 48,
    }
    assert b'candidate/publish gateway' in payload['instruction.md']
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE, 'w') as z:
        for name, data in payload.items():
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, data)
    manifest = {
        'model': MODEL,
        'archive_sha256': hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        'files': {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()},
        'scope': 'CPU packaging and helper regression checks only; no E4B inference or benchmark score',
    }
    ARCHIVE.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest

if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
