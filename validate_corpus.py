"""Run GRID builder + validator against the real T01 corpus."""
from __future__ import annotations

import sys
from pathlib import Path

from pipeline.grid.builder import GridManifestBuilder
from pipeline.grid.validate import GridValidator
from pipeline.grid.schema import ValidationStatus

CORPUS = Path("data/external/grid/s1/corpus")


def main() -> None:
    talk_dirs = sorted(d for d in CORPUS.iterdir() if d.is_dir())
    print(f"Found {len(talk_dirs)} talker dirs: {[d.name for d in talk_dirs]}")

    total_ok = 0
    total_failed = 0
    status_counts = {}
    duplicate_video_stems = 0

    for talk_dir in talk_dirs:
        talk_id = talk_dir.name
        print(f"\n=== Processing {talk_id} ===")
        builder = GridManifestBuilder(talk_dir.parent)  # parent = corpus root
        # But builder iterates root.iterdir() for talk dirs. We need just this talk.
        # Instead, build entries for this talk only:
        entries = builder._process_speaker(talk_id, talk_dir)
        print(f"  {talk_id}: {len(entries)} entries")

        validator = GridValidator(grid_root=talk_dir.parent, entries=entries)
        report = validator.run()
        report.talk_id = talk_id
        print(f"  {report.summarize()}")

        status_counts[talk_id] = {}
        for e in entries:
            vs = e.validation_status
            status_counts[talk_id][vs] = status_counts[talk_id].get(vs, 0) + 1
            if vs == ValidationStatus.OK:
                total_ok += 1
            else:
                total_failed += 1

    print(f"\n{'='*60}")
    print(f"REAL-CORPUS VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total entries: {total_ok + total_failed}")
    print(f"OK: {total_ok}")
    print(f"Failed: {total_failed}")
    print(f"\nPer-talk breakdown:")
    for tid, counts in status_counts.items():
        print(f"  {tid}: {counts}")

    # Check for duplicate video stems across the whole corpus
    all_video_stems = []
    for talk_dir in talk_dirs:
        video_dir = talk_dir / "video" / "s1"
        if video_dir.exists():
            for f in video_dir.iterdir():
                if f.is_file():
                    all_video_stems.append(f.stem)
    unique_stems = set(all_video_stems)
    print(f"\nVideo stems: {len(all_video_stems)} total, {len(unique_stems)} unique")
    if len(all_video_stems) != len(unique_stems):
        dups = [s for s in unique_stems if all_video_stems.count(s) > 1]
        print(f"  DUPLICATE STEMS: {dups[:10]}...")

    # Audio/alignment counts
    audio_count = 0
    align_count = 0
    for talk_dir in talk_dirs:
        a25 = talk_dir / "audio" / "25kHz" / "s1"
        al = talk_dir / "alignment" / "align"
        if a25.exists():
            audio_count += sum(1 for _ in a25.iterdir())
        if al.exists():
            align_count += sum(1 for _ in al.iterdir())
    print(f"Audio files: {audio_count}, Alignment files: {align_count}")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())