"""Stage published development objects on the owner's Colab VM via file-scoped Kaggle downloads."""
import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from kaggle.api.kaggle_api_extended import ApiDownloadDataFileRequest, KaggleApi

COMPETITION = "gemma-4-developer-agent"
HERE = Path(__file__).resolve().parent


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
    links = {}
    with api.build_kaggle_client() as client:
        for name in names:
            request = ApiDownloadDataFileRequest()
            request.competition_name = COMPETITION
            request.file_name = name
            response = client.competitions.competition_api_client.download_data_file(request)
            url = response.request.url
            response.close()
            origin = urlparse(url)
            if origin.scheme != "https" or origin.hostname != "storage.googleapis.com" or not origin.query:
                raise RuntimeError(f"No file-scoped storage URL returned for {name}")
            links[name] = url
    cli = str(Path.home() / ".local/bin/colab.exe")
    with tempfile.TemporaryDirectory() as temp:
        manifest = Path(temp) / "objects.json"
        manifest.write_text(json.dumps(links), encoding="utf8")
        subprocess.run([cli, "upload", "-s", args.session, str(manifest), "/content/wsp/objects.json"], check=True)
    run = subprocess.run([cli, "exec", "-s", args.session, "-f", str(HERE / "fetch_public_data.py"),
                          "--timeout", "600"], check=True, capture_output=True)
    if f"Staged {len(links)} competition files".encode() not in run.stdout:
        raise RuntimeError("Colab did not confirm the complete download")
    print(f"Staged {len(links)} published competition files")


if __name__ == "__main__":
    main()
