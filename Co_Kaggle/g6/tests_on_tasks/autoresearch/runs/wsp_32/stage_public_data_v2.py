"""Fast parallel file-scoped staging of published Kaggle competition objects to Colab VM (no credentials on VM)."""
import argparse
import concurrent.futures
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

from kaggle.api.kaggle_api_extended import ApiDownloadDataFileRequest, KaggleApi

COMPETITION = "gemma-4-developer-agent"
HERE = Path(__file__).resolve().parent
CLI = str(Path.home() / ".local/bin/colab.exe")


def exec_colab(session: str, code: str, timeout: int = 90) -> str:
    for attempt in range(4):
        try:
            run = subprocess.run(
                [CLI, "exec", "-s", session, "--timeout", str(timeout)],
                input=(code + "\nprint('WSP_OK')\n").encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout + 30,
            )
            out = run.stdout.decode("utf-8", errors="replace")
            if run.returncode == 0 and "WSP_OK" in out.splitlines():
                return out
            if attempt == 3:
                raise RuntimeError(f"colab exec failed:\n{out[-2500:]}")
        except subprocess.TimeoutExpired:
            if attempt == 3:
                raise
        time.sleep(3)
    raise RuntimeError("Unreachable")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session")
    parser.add_argument("ids", nargs="+")
    parser.add_argument("--with-wheels", action="store_true")
    args = parser.parse_args()

    if not all(re.fullmatch(r"(rich|fastapi|requests|httpx)_[0-9]+", i) for i in args.ids):
        parser.error("Invalid published task ID")

    names = ["tasks.jsonl", *(f"snapshots/{i}.tgz" for i in args.ids)]
    api = KaggleApi()
    api.authenticate()
    if args.with_wheels:
        token = None
        while True:
            page = api.competition_list_files(COMPETITION, page_token=token, page_size=200)
            names.extend(f.name for f in page.files if f.name.startswith("wheels/") and f.name.endswith(".whl"))
            token = page.next_page_token
            if not token:
                break

    # Check which files are ALREADY staged on the remote VM so we skip unnecessary work!
    already_out = exec_colab(args.session, """
import json
from pathlib import Path
dest = Path('/content/wsp/input/competitions/gemma-4-developer-agent')
existing = [str(p.relative_to(dest)).replace('\\\\', '/') for p in dest.rglob('*') if p.is_file() and not p.name.endswith('.part')] if dest.exists() else []
print('EXISTING=' + json.dumps(existing))
""")
    existing_line = next(l for l in already_out.splitlines() if l.startswith("EXISTING="))
    existing = set(json.loads(existing_line[len("EXISTING="):]))
    needed = [n for n in names if n not in existing]
    print(f"Competition files needed: {len(needed)}/{len(names)} ({len(existing)} already staged)", flush=True)
    if not needed:
        return

    t0 = time.monotonic()

    def resolve_one(name: str) -> tuple[str, str]:
        for attempt in range(4):
            try:
                with api.build_kaggle_client() as client:
                    req = ApiDownloadDataFileRequest()
                    req.competition_name = COMPETITION
                    req.file_name = name
                    resp = client.competitions.competition_api_client.download_data_file(req)
                    url = resp.request.url
                    resp.close()
                origin = urlparse(url)
                if origin.scheme != "https" or origin.hostname != "storage.googleapis.com" or not origin.query:
                    raise RuntimeError(f"No file-scoped storage URL returned for {name}")
                return name, url
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Failed resolving {name}")

    links = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for i, (name, url) in enumerate(pool.map(resolve_one, needed), 1):
            links[name] = url
            if i % 25 == 0 or i == len(needed):
                print(f"  Resolved signed URLs: {i}/{len(needed)} ({time.monotonic()-t0:.1f}s)", flush=True)

    with tempfile.TemporaryDirectory() as temp:
        manifest = Path(temp) / "objects.json"
        manifest.write_text(json.dumps(links), encoding="utf8")
        subprocess.run([CLI, "upload", "-s", args.session, str(manifest), "/content/wsp/objects.json"], check=True)

    # Launch fetch_public_data.py detached on remote and poll so colab exec never times out
    exec_colab(args.session, """
import subprocess, sys
from pathlib import Path
r = Path('/content/wsp')
done = r / 'fetch_public.done'
if done.exists():
    done.unlink()
with (r / 'fetch_public.log').open('w') as log:
    subprocess.Popen(
        ['bash', '-c', f'{sys.executable} -u /content/wsp/fetch_public_data.py; echo $? > /content/wsp/fetch_public.done'],
        cwd=r, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
""")
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        st = exec_colab(args.session, """
from pathlib import Path
r = Path('/content/wsp')
log = (r / 'fetch_public.log').read_text(errors='replace')[-600:] if (r / 'fetch_public.log').exists() else ''
rc = (r / 'fetch_public.done').read_text().strip() if (r / 'fetch_public.done').exists() else ''
print('TAIL:', log)
if rc != '':
    print('FETCH_RC=' + rc)
""")
        if "FETCH_RC=" in st:
            rc = int(next(l for l in st.splitlines() if l.startswith("FETCH_RC=")).split("=")[1])
            if rc != 0 or f"Staged {len(links)} competition files" not in st:
                raise RuntimeError(f"fetch_public_data.py failed (rc={rc}):\n{st}")
            print(f"Staged {len(links)} published competition files cleanly!", flush=True)
            return
        time.sleep(10)
    raise TimeoutError("fetch_public_data.py timed out")


if __name__ == "__main__":
    main()
