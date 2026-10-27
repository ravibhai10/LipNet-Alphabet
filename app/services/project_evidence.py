"""Read local corpus facts without presenting historical metrics as a benchmark."""
from collections import Counter
import json
from pathlib import Path


def corpus_evidence(root: Path) -> dict:
    grid_rows = []
    invalid = 0
    for manifest in (root / "data/external/grid").glob("*/grid_manifest.jsonl"):
        for line in manifest.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            try:
                grid_rows.append(json.loads(line))
            except (ValueError, TypeError):
                invalid += 1
    alphabet_labels = Counter()
    alphabet_speakers = set()
    for metadata in (root / "alphabet_dataset/recordings").glob("*/*/*/*/metadata.json"):
        try:
            row = json.loads(metadata.read_text(encoding="utf-8-sig"))
            label = row.get("letter", "")
            if row.get("data_source") == "real_camera" and label in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") and (metadata.parent / "video.mp4").is_file():
                alphabet_labels[label] += 1
                alphabet_speakers.add(row.get("speaker_id", ""))
        except (ValueError, OSError, TypeError):
            invalid += 1
    lrw_counts = {split: len(list((root / "data/lrw" / split).glob("*/*.mp4"))) for split in ("train", "valid", "test")}
    return {
        "alphabet": {"recordings": sum(alphabet_labels.values()), "speakers": len(alphabet_speakers),
                     "label_counts": dict(sorted(alphabet_labels.items())),
                     "missing_labels": [label for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if not alphabet_labels[label]]},
        "grid": {"utterances": len(grid_rows), "speakers": len({row.get("speaker_id") for row in grid_rows}),
                 "task": "sentence-level speech", "isolated_alphabet_annotations": False},
        "lrw": {"clips": sum(lrw_counts.values()), "split_counts": lrw_counts,
                "provenance": "development fixtures; not an eligible real-data benchmark"},
        "benchmark": {"status": "not_evaluated", "accuracy": None, "per_class": None,
                      "reason": "No audited speaker-independent isolated A-Z test set or trained AV-HuBERT alphabet head is registered."},
        "invalid_metadata_records": invalid,
    }
