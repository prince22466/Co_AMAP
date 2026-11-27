"""Fetch the selected public training files without installing account credentials on the VM."""
import concurrent.futures
import json
import shutil
import zipfile
from pathlib import Path

import requests

source = Path("/content/wsp/objects.json")
links = json.loads(source.read_text())
source.unlink()
destination = Path("/content/wsp/input/competitions/gemma-4-developer-agent")


def stage(item):
    name, url = item
    path = destination / name
    path.parent.mkdir(exist_ok=True, parents=True)
    if path.is_file():
        return
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=(15, 120)) as response:
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}")
            with tmp.open("wb") as handle:
                shutil.copyfileobj(response.raw, handle)
        if not name.endswith(".whl") and zipfile.is_zipfile(tmp):
            with zipfile.ZipFile(tmp) as archive:
                members = [n for n in archive.namelist() if Path(n).name == path.name]
                if len(members) != 1:
                    raise ValueError("Unexpected wrapper archive")
                expanded = tmp.with_suffix(".expanded")
                with archive.open(members[0]) as src, expanded.open("wb") as out:
                    shutil.copyfileobj(src, out)
            expanded.replace(tmp)
        tmp.replace(path)
        print(name, path.stat().st_size, flush=True)
    except Exception as exc:
        # Transport exception messages can contain signed URLs.
        raise RuntimeError(f"Download failed for {name} ({type(exc).__name__})") from None


with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    list(pool.map(stage, links.items()))
print(f"Staged {len(links)} competition files", flush=True)
links.clear()
