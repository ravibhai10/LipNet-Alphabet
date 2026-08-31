#!/usr/bin/env python3
"""Download all 34 GRID speaker zips from Zenodo (parallel, resumable, retried).

Output layout after this script runs:
    data/raw/grid/s1.zip, s2.zip, ..., s34.zip
    data/alignments/alignments/sN/bbaf2n.align, ...   (already extracted if you ran once)
    data/sN/sN/bbaf2n.mpg, ...                        (extracted by this script)
    data/sN/sN/sN.zip                                  (kept on disk for re-runs)

Usage:
    python scripts/download_grid.py                    # download all 34 (skips s21)
    python scripts/download_grid.py --speakers s1 s2   # specific speakers
    python scripts/download_grid.py --parallel 8       # custom parallelism
    python scripts/download_grid.py --skip-download    # only (re)extract zips
    python scripts/download_grid.py --verify           # verify only (no download)
    python scripts/download_grid.py --dry-run          # show plan, do nothing

After this script finishes, run:
    python scripts/prepare_alphabet.py --speakers all \\
        --max-per-letter 40 --out-dir dataset/alphabet_multi
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import logging
import os
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "raw" / "grid"
DEFAULT_DATA = PROJECT_ROOT / "data"
ZENODO_RECORD = "https://zenodo.org/records/3625687/files"
# GRID's official s21 file is missing from the Zenodo record.
MISSING_SPEAKERS = {21}
ALL_SPEAKERS = list(range(1, 35))  # 1..34

USER_AGENT = "Mozilla/5.0 (LipAlphabet/1.0; +https://zenodo.org/records/3625687)"


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _speakers_from_args(arg: List[str]) -> List[int]:
    """Parse ['all'] or ['s1','s2',...] or ['1','2',...] into [1,2,...]."""
    out: List[int] = []
    for tok in arg:
        s = tok.lower().lstrip("s")
        if s == "all":
            out.extend(ALL_SPEAKERS)
        elif s.isdigit():
            out.append(int(s))
        else:
            raise ValueError(f"Cannot parse speaker id: {tok!r}")
    return sorted(set(out))


def _expected_size(speaker: int) -> Optional[int]:
    """Return expected zip size in bytes from Zenodo, or None if HEAD fails."""
    url = f"{ZENODO_RECORD}/s{speaker}.zip"
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=20) as resp:
            cl = resp.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception as exc:  # noqa: BLE001
        logging.debug("HEAD failed for s%s: %s", speaker, exc)
        return None


def _check_zip(path: Path) -> Tuple[bool, str]:
    """Return (is_valid, message). Uses CRC when available, header otherwise."""
    if not path.exists():
        return False, "missing"
    sz = path.stat().st_size
    if sz < 100 * 1024:
        return False, f"suspiciously small ({sz} bytes)"
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            if bad:
                return False, f"corrupt member: {bad}"
        return True, f"ok ({sz/1e6:.1f} MB)"
    except zipfile.BadZipFile as e:
        return False, f"bad zip: {e}"


def _download_with_retry(url: str, dest: Path, max_retries: int = 5) -> bool:
    """Download with retry + exponential backoff.

    Note: Zenodo does not support Range requests, so we cannot resume
    a partial download mid-stream. On each retry we restart from the
    beginning. Set --max-retries higher for slow connections.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(1, max_retries + 1):
        # Remove any partial file from a previous failed attempt.
        if dest.exists():
            try:
                dest.unlink()
            except OSError:
                pass
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                with open(dest, "wb") as fh:
                    n = 0
                    t0 = time.time()
                    while True:
                        chunk = resp.read(64 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)
                        n += len(chunk)
                        if time.time() - t0 > 2:
                            mb = n / 1e6
                            spd = mb / (time.time() - t0)
                            eta = (total - n) / max(spd * 1e6, 1) if total else 0
                            logging.info(
                                "[%s] %5.1f MB / %5.1f MB  (%.1f MB/s, ETA %.0fs)",
                                dest.name, mb, total/1e6, spd, eta)
                            t0 = time.time()
                logging.info("[%s] done -> %s", dest.name, dest)
                return True
        except Exception as exc:  # noqa: BLE001
            wait = min(60, 2 ** attempt)
            logging.warning("[%s] attempt %d/%d failed: %s (retry in %ds)",
                            dest.name, attempt, max_retries, exc, wait)
            time.sleep(wait)
    return False


def _extract_zip(zip_path: Path, out_dir: Path, speaker: int) -> Dict[str, int]:
    """Extract videos + alignments from one GRID zip into data/."""
    dest_vid = out_dir / f"s{speaker}" / f"s{speaker}"
    dest_ali = out_dir / "alignments" / "alignments" / f"s{speaker}"
    dest_vid.mkdir(parents=True, exist_ok=True)
    dest_ali.mkdir(parents=True, exist_ok=True)
    counts = {"mpg": 0, "align": 0, "skipped": 0}
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                name = info.filename
                if name.endswith("/"):
                    continue
                ext = Path(name).suffix.lower()
                dest = None
                if ext in (".mpg", ".mp4"):
                    dest = dest_vid / Path(name).name
                elif ext == ".align":
                    dest = dest_ali / Path(name).name
                else:
                    counts["skipped"] += 1
                    continue
                if dest and not dest.exists():
                    try:
                        dest.write_bytes(zf.read(name))
                        if ext in (".mpg", ".mp4"):
                            counts["mpg"] += 1
                        else:
                            counts["align"] += 1
                    except Exception as exc:
                        logging.warning("  Failed %s: %s", name, exc)
                        counts["skipped"] += 1
                else:
                    counts["skipped"] += 1
    except zipfile.BadZipFile as exc:
        logging.error("  Corrupt zip %s: %s", zip_path.name, exc)
    return counts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--speakers", nargs="+", default=["all"],
                   help="Speaker IDs or 'all'  [default: all]")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                   help="Zip download dir  [default: data/raw/grid]")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA,
                   help="GRID data root  [default: data]")
    p.add_argument("-j", "--parallel", type=int, default=4)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()



def main() -> int:
    args = parse_args()
    _setup_logging(args.verbose)
    speakers = _speakers_from_args(args.speakers)
    speakers = [s for s in speakers if s not in MISSING_SPEAKERS]
    if not speakers:
        logging.error("No speakers to process"); return 1

    if args.verify:
        logging.info("=== Verifying existing zips ===")
        ok, bad = [], []
        for spk in speakers:
            p = args.out_dir / f"s{spk}.zip"
            valid, msg = _check_zip(p)
            logging.info("  s%d: %s  (%s)", spk, "OK" if valid else "BAD", msg)
            (ok if valid else bad).append(spk)
        print(f"\n{len(ok)} OK, {len(bad)} bad")
        if bad:
            print("Need re-download:", [f"s{s}.zip" for s in bad])
        return 0

    if args.dry_run:
        logging.info("=== Dry run ===")
        for spk in speakers:
            p = args.out_dir / f"s{spk}.zip"
            if p.exists():
                valid, msg = _check_zip(p)
                status = f"valid ({msg})" if valid else f"CORRUPT ({msg})"
            else:
                status = "missing (would download)"
            logging.info("  s%d: %s", spk, status)
        print(f"\nWould process {len(speakers)} speakers")
        return 0

    if not args.skip_download:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        logging.info("=== Downloading %d speakers ===", len(speakers))

        def dl_one(spk: int):
            url = f"{ZENODO_RECORD}/s{spk}.zip"
            dest = args.out_dir / f"s{spk}.zip"
            _download_with_retry(url, dest, args.max_retries)
            valid, msg = _check_zip(dest)
            return spk, valid, msg

        with cf.ThreadPoolExecutor(max_workers=args.parallel) as ex:
            for fut in cf.as_completed({ex.submit(dl_one, s): s for s in speakers}):
                spk, valid, msg = fut.result()
                logging.info("s%d: %s  (%s)", spk, "OK" if valid else "FAIL", msg)

    logging.info("=== Extracting to %s ===", args.data_dir)
    t_mpg = t_ali = t_skip = 0
    for spk in speakers:
        p = args.out_dir / f"s{spk}.zip"
        if not p.exists():
            logging.warning("s%d: zip not found, skipping", spk); continue
        valid, msg = _check_zip(p)
        if not valid:
            logging.error("s%d: corrupt zip (%s), skipping", spk, msg); continue
        logging.info("s%d: extracting ...", spk)
        c = _extract_zip(p, args.data_dir, spk)
        t_mpg += c["mpg"]; t_ali += c["align"]; t_skip += c["skipped"]
        logging.info("s%d: +%d mpg +%d align", spk, c["mpg"], c["align"])

    print(f"\n=== Summary ===")
    print(f"  Speakers:       {len(speakers)}")
    print(f"  Zips dir:       {args.out_dir}")
    print(f"  Data dir:       {args.data_dir}")
    print(f"  mpg extracted:  {t_mpg}")
    print(f"  align extracted:{t_ali}")
    print(f"\nNext: python scripts/prepare_alphabet.py --speakers all "
          f"--max-per-letter 40 --out-dir dataset/alphabet_multi")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

