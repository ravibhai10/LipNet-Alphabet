"""Reuse or acquire the official GRID sources, verifying recorded archive hashes.

No alternate dataset license/access is assumed. Existing official download and
safe ZIP/TAR extraction code is reused rather than inventing URL patterns.
"""
from __future__ import annotations
import subprocess
import sys

from .common import ROOT, atomic_json, load_config, load_json, parser, project_path, run_contract, sha256, timestamp


def prepare_sources(config, run):
    root = project_path(config["corpus_root"])
    archive_manifest = root / "download_manifest.json"
    missing = not archive_manifest.exists()
    if not missing:
        recorded = load_json(archive_manifest)
        missing = not recorded.get("assets") or any(asset.get("status") != "verified_and_extracted" or not (root / asset["path"]).is_file() for asset in recorded["assets"].values())
    if missing:
        subprocess.run([sys.executable, str(ROOT / "scripts/download_grid_official.py"), "--root", str(root),
                        "--parallel", "2", "--max-minutes", str(config["download_max_minutes"])], check=True, cwd=ROOT)
    manifest = load_json(archive_manifest)
    if manifest.get("source") != "https://spandh.dcs.shef.ac.uk/gridcorpus/":
        raise ValueError("Unrecognized corpus provenance.")
    checked = []
    for name, asset in sorted(manifest["assets"].items()):
        path = (root / asset["path"]).resolve()
        if not path.is_relative_to(root) or asset.get("status") != "verified_and_extracted" or not path.is_file():
            raise ValueError(f"Missing/unverified official archive: {name}; rerun the existing official downloader.")
        if path.stat().st_size != asset["bytes"] or sha256(path) != asset["sha256"]:
            raise ValueError(f"Archive fingerprint changed: {name}; existing bytes preserved.")
        checked.append({"asset": name, "bytes": asset["bytes"], "sha256": asset["sha256"]})
        if len(checked) % 10 == 0:
            print(f"Verified {len(checked)} existing official archives", flush=True)
    if not checked:
        raise ValueError("No verified archives.")
    index = project_path(config["source_inventory"])
    if not (index / "video_inventory.jsonl").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts/prepare_grid_dataset.py"), "--root", str(root),
                        "--output", str(index), "--workers", "2"], check=True, cwd=ROOT)
    result = {"checked_at": timestamp(), "source": manifest["source"], "archives": checked,
              "archive_checksums": "Compared to previously recorded local SHA-256; publisher supplied no SHA-256",
              "archive_validation_at_acquisition": "Existing downloader checks CRC/TAR members and safe extraction",
              "inventory_sha256": sha256(index / "video_inventory.jsonl"),
              "inventory_summary_sha256": sha256(index / "inventory.json")}
    atomic_json(run / "acquisition.json", result)
    return result


def main():
    args = parser(__doc__).parse_args()
    config = load_config(args.config)
    run = run_contract(config, args.run)
    result = prepare_sources(config, run)
    print(f"Verified {len(result['archives'])} archives; corpus inventory ready.")


if __name__ == "__main__":
    main()
