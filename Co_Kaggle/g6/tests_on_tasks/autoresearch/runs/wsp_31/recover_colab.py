"""Restore only the authorized research payload to the owner's replacement Colab runtime."""
import hashlib
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[1]
CLI = str(Path.home() / ".local/bin/colab.exe")
SESSION = "wsp7dev"


def execute(code):
    run = subprocess.run([CLI, "exec", "-s", SESSION, "--timeout", "60"], input=code.encode(),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=100)
    output = run.stdout.decode(errors="replace")
    if run.returncode != 0 or "WSP_OK" not in output.splitlines():
        raise RuntimeError(output[-2800:])
    return output


if __name__ == "__main__":
    files = {"champion.zip": RESEARCH.parents[1] / "submission_nb/artifacts/submission_40.zip",
             "frozen.ipynb": RESEARCH / "kaggle_eval_kernel/gemma-4-developer-agent-testing_v12_wsp.ipynb",
             "recipe.py": RESEARCH / "recipe.py"}
    for name in ("prepare_colab.py", "prepare_evaluator.py", "check_runtime.py", "check_control.py",
                 "run_evaluator.py", "start_discovery.py", "rank_columns.py"):
        files[name] = HERE / name
    with tempfile.TemporaryDirectory() as temp:
        bundle = Path(temp) / "wsp_setup.zip"
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, path in files.items():
                archive.write(path, name)
        sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
        subprocess.run([CLI, "upload", "-s", SESSION, str(bundle), "/content/wsp_setup.zip"], check=True)
    print(execute(f"""
import hashlib, subprocess, sys, zipfile
from pathlib import Path
root = Path('/content/wsp')
root.mkdir(exist_ok=True)
bundle = Path('/content/wsp_setup.zip')
assert hashlib.sha256(bundle.read_bytes()).hexdigest() == '{sha}'
with zipfile.ZipFile(bundle) as z:
    z.extractall(root)
with (root/'prepare.log').open('w') as log:
    p=subprocess.Popen([sys.executable,'-u',str(root/'prepare_colab.py')], stdout=log,stderr=subprocess.STDOUT)
print('WSP_OK')
"""), flush=True)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        progress = execute("from pathlib import Path; print(Path('/content/wsp/prepare.log').read_text(errors='replace')[-600:]); print('WSP_OK')")
        if "Development environment ready" in progress:
            break
        if "Error:" in progress:
            raise RuntimeError(progress)
        time.sleep(10)
    else:
        raise TimeoutError("Asset preparation still running")
    print(execute("from pathlib import Path; exec(Path('/content/wsp/prepare_evaluator.py').read_text()); print('WSP_OK')"), flush=True)
