from pathlib import Path, PurePosixPath

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

competition = "gemma-4-developer-agent"
folders = ("wheels/", "sample_submission/", "docker/", "sandbox/")
download_root = Path("competition_data")
token = None
count = 0

while True:
    page = api.competition_list_files(
        competition, page_token=token, page_size=200
    )

    for file in page.files:
        if file.name.startswith(folders):
            # Kaggle 2.2.4 saves each file's basename in the supplied path.
            # Supply its parent directory to preserve nested folders.
            destination = download_root / PurePosixPath(file.name).parent
            destination.mkdir(parents=True, exist_ok=True)
            print("Downloading:", file.name)
            api.competition_download_file(
                competition,
                file.name,
                path=str(destination),
            )
            count += 1

    token = page.next_page_token
    if not token:
        break

print(f"Downloaded {count} files.")
