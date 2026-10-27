"""Integrity checks for reproducible speaker-independent evaluation splits."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SplitIntegrityReport:
    split_speakers: dict[str, tuple[str, ...]]
    duplicate_sample_ids: tuple[str, ...]
    missing_split_sample_ids: tuple[str, ...]

    @property
    def overlaps(self) -> dict[str, tuple[str, ...]]:
        owners: dict[str, list[str]] = defaultdict(list)
        for split, speakers in self.split_speakers.items():
            for speaker in speakers:
                owners[speaker].append(split)
        return {
            speaker: tuple(splits)
            for speaker, splits in owners.items()
            if len(splits) > 1
        }

    @property
    def valid(self) -> bool:
        return not (
            self.overlaps
            or self.duplicate_sample_ids
            or self.missing_split_sample_ids
        )


def validate_speaker_splits(
    entries: Iterable[object],
    *,
    allowed_splits: tuple[str, ...] = ("train", "val", "test"),
) -> SplitIntegrityReport:
    """Validate split/speaker disjointness without changing entry objects.

    Entries must expose ``recording_id``, ``speaker_id`` and ``split``.
    """
    speakers: dict[str, set[str]] = {split: set() for split in allowed_splits}
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    missing_split_ids: set[str] = set()

    for entry in entries:
        sample_id = str(getattr(entry, "recording_id"))
        speaker_id = str(getattr(entry, "speaker_id"))
        split = getattr(entry, "split", None)
        if sample_id in seen_ids:
            duplicate_ids.add(sample_id)
        seen_ids.add(sample_id)
        if split not in speakers:
            missing_split_ids.add(sample_id)
        else:
            speakers[split].add(speaker_id)

    return SplitIntegrityReport(
        split_speakers={
            split: tuple(sorted(split_speakers))
            for split, split_speakers in speakers.items()
        },
        duplicate_sample_ids=tuple(sorted(duplicate_ids)),
        missing_split_sample_ids=tuple(sorted(missing_split_ids)),
    )
