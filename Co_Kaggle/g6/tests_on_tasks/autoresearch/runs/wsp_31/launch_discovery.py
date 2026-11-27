"""Wait for official files and the isolated environment, then run and checkpoint the champion."""
import time
from monitor import monitor
from recover_colab import HERE, execute

deadline = time.monotonic() + 1200
while time.monotonic() < deadline:
    progress = execute("""
from pathlib import Path
r = Path('/content/wsp')
d = r/'input/competitions/gemma-4-developer-agent'
e = r/'env_setup.log'
log = e.read_text(errors='replace') if e.exists() else ''
ready = 'Installed ' in log and ' + vllm==' in log and (d/'tasks.jsonl').is_file() and len(list((d/'snapshots').glob('*.tgz'))) == 8 and len(list((d/'wheels').glob('*.whl'))) == 124
print('ready', ready, 'snapshots', len(list((d/'snapshots').glob('*.tgz'))), 'wheels', len(list((d/'wheels').glob('*.whl'))))
print('WSP_OK')
""")
    print(progress, flush=True)
    if "ready True" in progress:
        break
    time.sleep(18)
else:
    raise TimeoutError("Staging still incomplete")
print(execute((HERE/"start_discovery.py").read_text() + "\nprint('WSP_OK')"), flush=True)
monitor("discovery01", "discovery.log")
