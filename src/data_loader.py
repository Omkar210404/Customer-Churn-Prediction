"""Download and load the IBM Telco Customer Churn dataset.

Usage:
    python -m src.data_loader          # download (if missing) and verify checksum
    python -m src.data_loader --force  # re-download
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd

from src import config


class DatasetError(RuntimeError):
    """Raised when the dataset is missing or does not match the expected file."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(dest: Path = config.RAW_CSV, force: bool = False) -> Path:
    """Download the CSV to `dest` and verify its SHA-256. Returns the path."""
    if dest.exists() and not force:
        verify_dataset(dest)
        print(f"Dataset already present: {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {config.DATASET_URL}")
    with tempfile.NamedTemporaryFile(delete=False, dir=dest.parent, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
        with urllib.request.urlopen(config.DATASET_URL, timeout=60) as resp:
            shutil.copyfileobj(resp, tmp)
    try:
        verify_dataset(tmp_path)
    except DatasetError:
        tmp_path.unlink(missing_ok=True)
        raise
    tmp_path.replace(dest)
    print(f"Saved to {dest}")
    return dest


def verify_dataset(path: Path) -> None:
    actual = sha256_of(path)
    if actual != config.DATASET_SHA256:
        raise DatasetError(
            f"Checksum mismatch for {path}.\n"
            f"  expected {config.DATASET_SHA256}\n  got      {actual}\n"
            "The file differs from the version this project was built on. "
            "See data/README.md."
        )


def load_raw(path: Path = config.RAW_CSV) -> pd.DataFrame:
    """Load the raw CSV with every column as text, so cleaning is explicit."""
    if not path.exists():
        raise DatasetError(
            f"Dataset not found at {path}. Run `python -m src.data_loader` "
            "or see data/README.md to place it manually."
        )
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()
    path = download_dataset(force=args.force)
    df = load_raw(path)
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns. Checksum OK.")


if __name__ == "__main__":
    main()
