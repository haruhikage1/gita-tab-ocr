import os
import zipfile

import requests

from gtrs.simple_logging import eprint


def download_file(url: str, destination: str) -> None:
    eprint(f"Downloading {url}")
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def unzip_file(zip_path: str, destination_dir: str) -> None:
    eprint(f"Unzipping {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(destination_dir)