"""Reproducible E4B direct-tool candidate; no inference."""
import hashlib
import json
from pathlib import Path
import zipfile
import yaml

G6 = Path(__file__).resolve().parents[2]
SOURCE = G6/'submission_nb/submission_19'
ARCHIVE = G6/'submission_nb/artifacts/submission_19.zip'
MODEL = 'gemma-4-e4b-it-qat-w4a16-ct'
FILES = ('agent.yaml', 'eval_config.yaml', 'instruction.md', 'skills/patch-audit/SKILL.md', 'skills/patch-audit/scripts/audit.py')


def build():
    payload = {n:(SOURCE/n).read_bytes() for n in FILES}
    class Loader(yaml.SafeLoader): pass
    Loader.add_constructor('!include', lambda l,n:l.construct_scalar(n))
    agent = yaml.load(payload['agent.yaml'], Loader=Loader)
    assert agent['model'] == MODEL
    assert agent['tools'] == ['get_status','read_file','edit_file','write_file','run_command','submit_patch']
    assert agent['skills'] == ['skills/patch-audit'] and 'sub_agents' not in agent
    assert agent['generate_content_config']['thinking_config']['thinking_budget'] == 0
    assert yaml.safe_load(payload['eval_config.yaml'])['evaluation'] == dict(
        timeout_seconds=30,max_tool_calls=40,max_time_minutes=4,max_turns=48)
    ARCHIVE.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(ARCHIVE,'w') as z:
        for name,data in payload.items():
            info=zipfile.ZipInfo(name,(2026,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644 << 16
            z.writestr(info,data)
    manifest={'model':MODEL,'archive_sha256':hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        'files':{n:hashlib.sha256(d).hexdigest() for n,d in payload.items()},
        'scope':'Packaging only. Real-model performance unmeasured.'}
    ARCHIVE.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

if __name__=='__main__':print(json.dumps(build(),indent=2))
