"""
Dataset manifest creation and management.

Creates manifest.jsonl with one line per recording for fast dataset indexing.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from data.schema import (
    RecordingMetadata, ManifestEntry, validate_manifest_entry,
    Letter, Split, validate_metadata
)

logger = logging.getLogger(__name__)


@dataclass
class ManifestConfig:
    """Configuration for manifest creation."""
    dataset_root: Path
    manifest_path: Path
    schema_version: str = "1.0"
    split_info_path: Optional[Path] = None


class ManifestManager:
    """
    Manage dataset manifest (manifest.jsonl).

    Provides fast indexing without scanning thousands of files.
    """

    def __init__(self, config: ManifestConfig):
        self.config = config
        self.dataset_root = Path(config.dataset_root)
        self.manifest_path = Path(config.manifest_path)
        self.schema_version = config.schema_version

    def build_manifest(self, splits: Optional[Dict[str, str]] = None) -> List[ManifestEntry]:
        """
        Build manifest by scanning dataset directory.

        Args:
            splits: Optional dict mapping recording_id -> split (train/val/test)

        Returns:
            List of ManifestEntry objects
        """
        entries = []
        recordings_root = self.dataset_root / "recordings"

        if not recordings_root.exists():
            logger.warning(f"Recordings directory not found: {recordings_root}")
            return entries

        # Scan all letter directories
        for letter_dir in sorted(recordings_root.iterdir()):
            if not letter_dir.is_dir() or not Letter.is_valid(letter_dir.name):
                continue

            letter = letter_dir.name

            # Scan speaker directories
            for speaker_dir in sorted(letter_dir.iterdir()):
                if not speaker_dir.is_dir():
                    continue

                speaker_id = speaker_dir.name

                # Scan session directories
                for session_dir in sorted(speaker_dir.iterdir()):
                    if not session_dir.is_dir():
                        continue

                    session_id = session_dir.name

                    # Scan recording directories
                    for recording_dir in sorted(session_dir.iterdir()):
                        if not recording_dir.is_dir():
                            continue

                        recording_id = recording_dir.name

                        # Load metadata
                        metadata_path = recording_dir / "metadata.json"
                        if not metadata_path.exists():
                            logger.warning(f"Missing metadata: {metadata_path}")
                            continue

                        try:
                            metadata = RecordingMetadata.load(metadata_path)

                            # Validate metadata
                            errors = validate_metadata(metadata)
                            if errors:
                                logger.warning(f"Metadata validation errors for {recording_id}: {errors}")

                            # Compute relative path
                            rel_path = recording_dir.relative_to(self.dataset_root)

                            # Get split assignment (splits are keyed by relative path)
                            split = None
                            if splits:
                                split = splits.get(str(rel_path))

                            # Create manifest entry
                            entry = ManifestEntry.from_metadata(metadata, str(rel_path), split)
                            entries.append(entry)

                        except Exception as e:
                            logger.error(f"Error loading metadata for {recording_id}: {e}")

        # Save manifest
        self.save_manifest(entries)

        logger.info(f"Built manifest with {len(entries)} recordings")
        return entries

    def save_manifest(self, entries: List[ManifestEntry]) -> None:
        """Save manifest entries to JSONL file."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.manifest_path, 'w') as f:
            for entry in entries:
                # Validate before saving
                errors = validate_manifest_entry(entry)
                if errors:
                    logger.warning(f"Manifest entry validation errors for {entry.recording_id}: {errors}")

                f.write(entry.to_jsonl() + '\n')

        logger.info(f"Saved manifest to {self.manifest_path}")

    def load_manifest(self) -> List[ManifestEntry]:
        """Load manifest from JSONL file."""
        if not self.manifest_path.exists():
            logger.warning(f"Manifest not found: {self.manifest_path}")
            return []

        entries = []
        with open(self.manifest_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = ManifestEntry.from_jsonl(line)
                    entries.append(entry)
                except Exception as e:
                    logger.error(f"Error parsing manifest line {line_num}: {e}")

        logger.info(f"Loaded {len(entries)} entries from manifest")
        return entries

    def get_entries_by_letter(self, letter: str) -> List[ManifestEntry]:
        """Get all entries for a specific letter."""
        all_entries = self.load_manifest()
        return [e for e in all_entries if e.letter == letter.upper()]

    def get_entries_by_speaker(self, speaker_id: str) -> List[ManifestEntry]:
        """Get all entries for a specific speaker."""
        all_entries = self.load_manifest()
        return [e for e in all_entries if e.speaker_id == speaker_id]

    def get_entries_by_session(self, session_id: str) -> List[ManifestEntry]:
        """Get all entries for a specific session."""
        all_entries = self.load_manifest()
        return [e for e in all_entries if e.session_id == session_id]

    def get_entries_by_split(self, split: str) -> List[ManifestEntry]:
        """Get all entries for a specific split."""
        all_entries = self.load_manifest()
        return [e for e in all_entries if e.split == split]

    def get_statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics from manifest."""
        entries = self.load_manifest()

        if not entries:
            return {
                "total_recordings": 0,
                "total_speakers": 0,
                "total_sessions": 0,
                "letters": {},
                "splits": {},
                "frame_stats": {},
                "duration_stats": {},
            }

        speakers = set(e.speaker_id for e in entries)
        sessions = set(f"{e.speaker_id}/{e.session_id}" for e in entries)

        # Per-letter counts
        letter_counts = {}
        for e in entries:
            letter_counts[e.letter] = letter_counts.get(e.letter, 0) + 1

        # Per-split counts
        split_counts = {}
        for e in entries:
            s = e.split or "unassigned"
            split_counts[s] = split_counts.get(s, 0) + 1

        # Frame statistics
        frame_counts = [e.frame_count for e in entries]
        valid_frame_counts = [e.valid_frame_count for e in entries]
        durations = [e.duration_ms for e in entries]

        return {
            "total_recordings": len(entries),
            "total_speakers": len(speakers),
            "total_sessions": len(sessions),
            "letters": letter_counts,
            "splits": split_counts,
            "frame_stats": {
                "mean": sum(frame_counts) / len(frame_counts) if frame_counts else 0,
                "min": min(frame_counts) if frame_counts else 0,
                "max": max(frame_counts) if frame_counts else 0,
                "total": sum(frame_counts),
            },
            "valid_frame_stats": {
                "mean": sum(valid_frame_counts) / len(valid_frame_counts) if valid_frame_counts else 0,
                "min": min(valid_frame_counts) if valid_frame_counts else 0,
                "max": max(valid_frame_counts) if valid_frame_counts else 0,
                "total": sum(valid_frame_counts),
            },
            "duration_stats": {
                "mean_ms": sum(durations) / len(durations) if durations else 0,
                "min_ms": min(durations) if durations else 0,
                "max_ms": max(durations) if durations else 0,
                "total_ms": sum(durations),
            },
        }

    def validate_manifest(self) -> List[str]:
        """
        Validate manifest for consistency issues.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        entries = self.load_manifest()

        if not entries:
            errors.append("Manifest is empty")
            return errors

        # Check for duplicate recordings (scoped to letter/speaker/session)
        scoped = {}
        for entry in entries:
            key = (entry.letter, entry.speaker_id, entry.session_id, entry.recording_id)
            if key in scoped:
                errors.append(
                    f"Duplicate recording (scoped {key}): "
                    f"{scoped[key]} and {entry.path}"
                )
            scoped[key] = entry.path

        # Check for invalid letters
        for entry in entries:
            if not Letter.is_valid(entry.letter):
                errors.append(f"Invalid letter '{entry.letter}' in recording {entry.recording_id}")

        # Check for missing files
        for entry in entries:
            rec_path = self.dataset_root / entry.path
            if not rec_path.exists():
                errors.append(f"Recording directory missing: {entry.path}")

            metadata_path = rec_path / "metadata.json"
            if not metadata_path.exists():
                errors.append(f"Metadata missing: {entry.path}/metadata.json")

            video_path = rec_path / "video.mp4"
            if not video_path.exists():
                errors.append(f"Video missing: {entry.path}/video.mp4")

        # Check split consistency
        speaker_splits = {}
        for entry in entries:
            if entry.split:
                if entry.speaker_id in speaker_splits:
                    if speaker_splits[entry.speaker_id] != entry.split:
                        errors.append(
                            f"Speaker {entry.speaker_id} appears in multiple splits: "
                            f"{speaker_splits[entry.speaker_id]} and {entry.split}"
                        )
                else:
                    speaker_splits[entry.speaker_id] = entry.split

        return errors


def create_splits(
    manifest_path: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    min_speakers_per_split: int = 3
) -> Dict[str, str]:
    """
    Create speaker-independent splits.

    Args:
        manifest_path: Path to manifest.jsonl
        train_ratio: Fraction of speakers for training
        val_ratio: Fraction of speakers for validation
        test_ratio: Fraction of speakers for test
        random_seed: Random seed for reproducibility
        min_speakers_per_split: Minimum speakers per split

    Returns:
        Dict mapping recording_id -> split
    """
    import random

    # Load manifest
    entries = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(ManifestEntry.from_jsonl(line))

    # Group by speaker. We key the returned split map by the recording's
    # relative path (globally unique), because ``recording_id`` is only unique
    # within its session directory.
    speaker_paths = {}
    for entry in entries:
        if entry.speaker_id not in speaker_paths:
            speaker_paths[entry.speaker_id] = []
        speaker_paths[entry.speaker_id].append(entry.path)

    speakers = list(speaker_paths.keys())
    random.seed(random_seed)
    random.shuffle(speakers)

    n_speakers = len(speakers)
    n_train = max(min_speakers_per_split, int(n_speakers * train_ratio))
    n_val = max(min_speakers_per_split, int(n_speakers * val_ratio))
    n_test = max(min_speakers_per_split, n_speakers - n_train - n_val)

    # Adjust if needed
    if n_train + n_val + n_test > n_speakers:
        n_test = n_speakers - n_train - n_val

    train_speakers = set(speakers[:n_train])
    val_speakers = set(speakers[n_train:n_train + n_val])
    test_speakers = set(speakers[n_train + n_val:])

    # Assign splits (one split per speaker, applied to all of that speaker's
    # recordings -> strict speaker independence and no session leakage).
    splits = {}
    for speaker, paths in speaker_paths.items():
        if speaker in train_speakers:
            split = "train"
        elif speaker in val_speakers:
            split = "val"
        else:
            split = "test"

        for path in paths:
            splits[path] = split

    return splits


def save_split_info(
    splits: Dict[str, str],
    manifest_path: Path,
    output_path: Path,
    random_seed: int
) -> None:
    """Save split assignment information."""
    from data.schema import DatasetSplitInfo

    # Load manifest to get sessions
    entries = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(ManifestEntry.from_jsonl(line))

    train_speakers = set()
    val_speakers = set()
    test_speakers = set()
    train_sessions = set()
    val_sessions = set()
    test_sessions = set()

    for entry in entries:
        split = splits.get(entry.path)
        if split == "train":
            train_speakers.add(entry.speaker_id)
            train_sessions.add(f"{entry.speaker_id}/{entry.session_id}")
        elif split == "val":
            val_speakers.add(entry.speaker_id)
            val_sessions.add(f"{entry.speaker_id}/{entry.session_id}")
        elif split == "test":
            test_speakers.add(entry.speaker_id)
            test_sessions.add(f"{entry.speaker_id}/{entry.session_id}")

    split_info = DatasetSplitInfo(
        train_speakers=sorted(train_speakers),
        val_speakers=sorted(val_speakers),
        test_speakers=sorted(test_speakers),
        train_sessions=sorted(train_sessions),
        val_sessions=sorted(val_sessions),
        test_sessions=sorted(test_sessions),
        split_seed=random_seed,
        created_at=datetime.utcnow().isoformat() + "Z"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(split_info.to_dict(), f, indent=2)

    logger.info(f"Saved split info to {output_path}")