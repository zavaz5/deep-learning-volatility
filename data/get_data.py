"""Check the authors' six data files in data/ against their SHA-256 values, and download any that is missing.

The files come from the authors' MIT-licensed repository, https://github.com/amuguruza/NN-StochVol-Calibrations,
and are included here with their licence (data/LICENSE): the two training sets (about 47 MB each) and the authors'
own calibration outputs for the test surfaces (four small text files, for `src/final_eval.py --authors-calibration`).
The hash check catches a damaged or changed file before it can change a result.

Usage:  python data/get_data.py
"""
import hashlib
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_DIR, DATASETS, RAW_DATA_URL


def sha256_of(path: Path) -> str:
    """Hash the file in chunks so a 47 MB file never sits in memory twice."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    """Fetch url into target with Python, or with the system curl if Python cannot verify HTTPS.

    The python.org installer for macOS ships without root certificates until the user runs
    "Install Certificates.command", so urllib fails there with CERTIFICATE_VERIFY_FAILED.
    curl uses the operating system's certificates. We never switch verification off; the
    SHA-256 check afterwards guards the content whichever route fetched it.
    """
    try:
        urllib.request.urlretrieve(url, target)
        return
    except urllib.error.URLError as error:
        print(f"  Python could not download ({error.reason}); trying curl")
    if shutil.which("curl") is None:
        raise SystemExit(f"Download failed and curl is not available. Fetch {url} by hand into {target.parent}/ "
                         "and run this script again; it will verify the hash.")
    subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--output", str(target), url],
                   check=True)


def fetch(file_name: str, expected_sha256: str) -> None:
    """Download one file unless a copy with the right hash is already there.

    A wrong hash stops the run: every number in this project depends on these exact files.
    """
    target = DATA_DIR / file_name
    if target.exists() and sha256_of(target) == expected_sha256:
        print(f"ok (already present)  {file_name}")
        return
    print(f"downloading           {file_name}")
    download(RAW_DATA_URL + file_name, target)
    found = sha256_of(target)
    if found != expected_sha256:
        target.unlink()
        raise SystemExit(f"SHA-256 mismatch for {file_name}: expected {expected_sha256}, found {found}")
    print(f"ok (hash verified)    {file_name}")


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    for dataset in DATASETS.values():
        fetch(dataset["file"], dataset["sha256"])
    for dataset in DATASETS.values():
        for file_name, sha256 in dataset["authors_calibration"].values():
            fetch(file_name, sha256)
