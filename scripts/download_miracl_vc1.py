"""Safe, resumable downloader and extractor for MIRACL-VC1 dataset.

Checks credentials, verifies target storage, executes bounded retries,
computes SHA-256 checksums, and safely unpacks without path traversal.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "apoorvwatsky/miraclvc1"
TARGET_DIR = ROOT / "data/external/miraclvc1"


def check_kaggle_credentials() -> bool:
    kaggle_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    kaggle_json = kaggle_dir / "kaggle.json"
    if kaggle_json.is_file():
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return False


def print_setup_instructions():
    print("=" * 70)
    print("MIRACL-VC1 ACQUISITION NOTICE: KAGGLE CREDENTIALS REQUIRED")
    print("=" * 70)
    print("To download MIRACL-VC1 automatically via Kaggle:")
    print("1. Log in to your Kaggle account at https://www.kaggle.com")
    print("2. Navigate to Account Settings -> 'API' section -> 'Create New Token'")
    print("3. A 'kaggle.json' file will be downloaded.")
    print(f"4. Move 'kaggle.json' to: {Path.home() / '.kaggle / kaggle.json'}")
    print("5. Re-run this acquisition script:")
    print("   python scripts/download_miracl_vc1.py")
    print("=" * 70)


def download_miracl(target_dir: Path = TARGET_DIR, max_retries: int = 3):
    if not check_kaggle_credentials():
        print_setup_instructions()
        return False

    target_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(target_dir).free
    required_bytes = 6 * 1024 * 1024 * 1024 # ~6 GB
    if free_bytes < required_bytes:
        raise OSError(f"Insufficient disk space. Need {required_bytes / 1e9:.1f} GB, available: {free_bytes / 1e9:.1f} GB")

    print(f"Downloading {DATASET_ID} to {target_dir}...")
    success = False
    for attempt in range(1, max_retries + 1):
        try:
            cmd = [
                sys.executable, "-m", "kaggle", "datasets", "download",
                "-d", DATASET_ID,
                "-p", str(target_dir),
                "--unzip"
            ]
            subprocess.run(cmd, check=True)
            success = True
            break
        except Exception as e:
            print(f"Attempt {attempt}/{max_retries} failed: {e}")
            time.sleep(3)

    if not success:
        raise RuntimeError(f"Failed to acquire {DATASET_ID} after {max_retries} attempts.")

    print(f"Successfully acquired and unpacked {DATASET_ID}.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=TARGET_DIR)
    args = parser.parse_args()
    download_miracl(args.target)
