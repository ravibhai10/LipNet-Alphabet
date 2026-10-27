"""Fix grid_manifest.jsonl: replace literal \\n with real newlines, write atomically."""
from __future__ import annotations

import json
import os
import sys
import hashlib
import tempfile
from pathlib import Path

MANIFEST = Path("data/external/grid/s1/grid_manifest.jsonl")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not MANIFEST.exists():
        print(f"ERROR: manifest not found: {MANIFEST}")
        sys.exit(1)

    old_hash = sha256(MANIFEST)
    print(f"Old manifest: {MANIFEST} size={MANIFEST.stat().st_size} sha256={old_hash}")

    raw = MANIFEST.read_text(encoding="utf-8")

    # The manifest contains literal \\n text (backslash-n) instead of real newlines.
    # Replace every occurrence of the literal two-character sequence backslash-n
    # with an actual newline character.
    fixed = raw.replace("\\n", "\n")

    # Split into physical lines; drop any trailing empty line from the replacement
    lines = fixed.split("\n")
    # Remove trailing empty strings caused by a final newline
    while lines and lines[-1].strip() == "":
        lines.pop()

    print(f"Total physical lines after split: {len(lines)}")

    # Validate every line independently with json.loads()
    records = []
    errors = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
            records.append(obj)
        except json.JSONDecodeError as exc:
            errors.append(f"line {i+1}: {exc} | {stripped[:120]}")

    if errors:
        print(f"ERROR: {len(errors)} invalid JSON lines:")
        for e in errors[:10]:
            print(f"  {e}")
        sys.exit(1)

    # Validate: exactly 1000 records
    if len(records) != 1000:
        print(f"ERROR: expected 1000 records, got {len(records)}")
        sys.exit(1)

    # Validate: exactly 1000 unique utterance IDs
    utt_ids = [r["utterance_id"] for r in records]
    if len(set(utt_ids)) != 1000:
        print(f"ERROR: expected 1000 unique utterance IDs, got {len(set(utt_ids))}")
        sys.exit(1)

    # Validate: every record has matching stems
    from pathlib import PureWindowsPath
    mismatch = 0
    for r in records:
        v_stem = PureWindowsPath(r["video_path"]).stem
        a_stem = PureWindowsPath(r["audio_path"]).stem if r["audio_path"] else ""
        al_stem = PureWindowsPath(r["alignment_path"]).stem if r["alignment_path"] else ""
        if not (v_stem == a_stem == al_stem):
            mismatch += 1
            print(f"  STEM MISMATCH: {r['utterance_id']} v={v_stem} a={a_stem} al={al_stem}")
    if mismatch:
        print(f"ERROR: {mismatch} records have mismatched stems")
        sys.exit(1)

    # Validate: all expected files exist on disk
    # Manifest paths use backslashes; corpus root is MANIFEST.parent / "corpus"
    corpus_root = MANIFEST.parent / "corpus"
    missing_files = 0
    for r in records:
        for key in ("video_path", "audio_path", "alignment_path"):
            p = r[key]
            if not p:
                continue
            # Normalize Windows backslashes to Path-compatible form
            p_norm = p.replace("\\", "/")
            full = corpus_root / p_norm
            if not full.exists():
                missing_files += 1
                print(f"  MISSING: {key}={p} for {r['utterance_id']}")
    if missing_files:
        print(f"ERROR: {missing_files} referenced files missing on disk")
        sys.exit(1)

    # Validate: all validation statuses are correct (ok or known error statuses)
    valid_statuses = {"ok", "missing_video", "missing_audio", "missing_alignment", "corrupt_media"}
    bad_status = [r for r in records if r["validation_status"] not in valid_statuses]
    if bad_status:
        print(f"ERROR: {len(bad_status)} records have invalid validation_status")
        sys.exit(1)

    # First sample check
    first = records[0]
    print(f"First sample: {first['utterance_id']} frames={first['frame_count']} fps={first['fps']} transcript='{first['transcript']}'")

    # Write atomically: temp file → validate → replace
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(MANIFEST.parent), prefix=".grid_manifest_tmp", suffix=".jsonl")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8", newline="\n") as f:
            for i, rec in enumerate(records):
                f.write(json.dumps(rec, ensure_ascii=False))
                if i < len(records) - 1:
                    f.write("\n")
        tmp = Path(tmp_path)
        # Re-validate the temp file before replacing
        with tmp.open(encoding="utf-8") as f:
            tmp_lines = [ln for ln in f.read().split("\n") if ln.strip()]
        if len(tmp_lines) != 1000:
            print(f"ERROR: temp file has {len(tmp_lines)} lines, expected 1000")
            sys.exit(1)
        for ln in tmp_lines:
            json.loads(ln)
        os.replace(tmp, MANIFEST)
        print(f"Wrote {MANIFEST} atomically ({len(records)} records, real newlines)")
    finally:
        if Path(tmp_path).exists():
            Path(tmp_path).unlink(missing_ok=True)

    new_hash = sha256(MANIFEST)
    print(f"New manifest: sha256={new_hash}")
    print("MANIFEST FIX COMPLETE")


if __name__ == "__main__":
    main()