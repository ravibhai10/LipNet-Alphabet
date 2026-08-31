"""
Tests for dataset manifest creation and splits.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from data.schema import RecordingMetadata, ManifestEntry, FaceTrackingInfo, MouthInfo, QualityMetrics, Letter, Split, DATA_SOURCE_SYNTHETIC_TEST
from data.manifest import ManifestManager, ManifestConfig, create_splits, save_split_info


class TestManifestManager:
    """Test ManifestManager."""

    def create_test_dataset(self, tmpdir: Path) -> Path:
        """Create a minimal test dataset structure."""
        dataset_root = tmpdir / "test_dataset"
        recordings_root = dataset_root / "recordings"

        # Create directory structure
        for letter in ["A", "B"]:
            for speaker in ["speaker_001", "speaker_002"]:
                for session in ["session_001", "session_002"]:
                    for rec_num in [1, 2]:
                        rec_dir = recordings_root / letter / speaker / session / f"recording_{rec_num:03d}"
                        rec_dir.mkdir(parents=True)

                        # Create metadata
                        metadata = RecordingMetadata(
letter=letter,
                            speaker_id=speaker,
                            session_id=session,
                            recording_id=f"recording_{rec_num:03d}",
                            data_source=DATA_SOURCE_SYNTHETIC_TEST,
                            source="webcam",
                            created_at="2026-01-01T00:00:00Z",
                            fps=30.0,
                            frame_count=60,
                            width=1280,
                            height=720,
                            duration_ms=2000.0,
                            face_tracking=FaceTrackingInfo(
                                frames_with_face=55, frames_without_face=5,
                                max_consecutive_missing=2, mean_confidence=0.9, min_confidence=0.7
                            ),
                            mouth=MouthInfo(
                                frames_with_mouth=50, missing_frames=10,
                                mean_mouth_width=100, mean_mouth_height=50,
                                min_mouth_area=2000, max_mouth_area=8000
                            ),
                            quality=QualityMetrics(
                                mean_brightness=128, std_brightness=20,
                                mean_blur=150, mean_mouth_size=5000,
                                mean_pose_yaw=0, mean_pose_pitch=0, mean_pose_roll=0,
                                duplicate_frame_count=0, quality_flags=[]
                            ),
                            preprocessing_version="1.0"
                        )
                        metadata.save(rec_dir / "metadata.json")

                        # Create dummy video
                        (rec_dir / "video.mp4").write_bytes(b"dummy")

        return dataset_root

    def test_build_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = self.create_test_dataset(Path(tmpdir))
            manifest_path = dataset_root / "manifest.jsonl"

            manager = ManifestManager(ManifestConfig(
                dataset_root=dataset_root,
                manifest_path=manifest_path,
            ))

            entries = manager.build_manifest()

            # Should have 2 letters * 2 speakers * 2 sessions * 2 recordings = 16 entries
            assert len(entries) == 16
            assert manifest_path.exists()

    def test_load_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = self.create_test_dataset(Path(tmpdir))
            manifest_path = dataset_root / "manifest.jsonl"

            manager = ManifestManager(ManifestConfig(
                dataset_root=dataset_root,
                manifest_path=manifest_path,
            ))

            entries = manager.build_manifest()
            loaded = manager.load_manifest()

            assert len(loaded) == len(entries)
            for orig, loaded_entry in zip(entries, loaded):
                assert orig.recording_id == loaded_entry.recording_id
                assert orig.speaker_id == loaded_entry.speaker_id

    def test_get_entries_by_letter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = self.create_test_dataset(Path(tmpdir))

            manager = ManifestManager(ManifestConfig(
                dataset_root=dataset_root,
                manifest_path=dataset_root / "manifest.jsonl",
            ))

            manager.build_manifest()
            a_entries = manager.get_entries_by_letter("A")
            b_entries = manager.get_entries_by_letter("B")

            assert len(a_entries) == 8
            assert len(b_entries) == 8
            assert all(e.letter == "A" for e in a_entries)
            assert all(e.letter == "B" for e in b_entries)

    def test_get_entries_by_speaker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = self.create_test_dataset(Path(tmpdir))

            manager = ManifestManager(ManifestConfig(
                dataset_root=dataset_root,
                manifest_path=dataset_root / "manifest.jsonl",
            ))

            manager.build_manifest()
            speaker_entries = manager.get_entries_by_speaker("speaker_001")

            assert len(speaker_entries) == 8  # 2 letters * 2 sessions * 2 recordings
            assert all(e.speaker_id == "speaker_001" for e in speaker_entries)

    def test_get_statistics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = self.create_test_dataset(Path(tmpdir))

            manager = ManifestManager(ManifestConfig(
                dataset_root=dataset_root,
                manifest_path=dataset_root / "manifest.jsonl",
            ))

            manager.build_manifest()
            stats = manager.get_statistics()

            assert stats["total_recordings"] == 16
            assert stats["total_speakers"] == 2
            assert stats["total_sessions"] == 4  # 2 speakers * 2 sessions
            assert stats["letters"]["A"] == 8
            assert stats["letters"]["B"] == 8

    def test_validate_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = self.create_test_dataset(Path(tmpdir))

            manager = ManifestManager(ManifestConfig(
                dataset_root=dataset_root,
                manifest_path=dataset_root / "manifest.jsonl",
            ))

            manager.build_manifest()
            errors = manager.validate_manifest()

            assert len(errors) == 0


class TestCreateSplits:
    """Test speaker-independent split creation."""

    def test_create_splits_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create manifest with multiple speakers
            manifest_path = Path(tmpdir) / "manifest.jsonl"

            entries = []
            for i, speaker in enumerate(["speaker_001", "speaker_002", "speaker_003", "speaker_004"]):
                for letter in ["A", "B"]:
                    entry = ManifestEntry(
                        recording_id=f"rec_{speaker}_{letter}",
                        speaker_id=speaker,
                        session_id="session_001",
                        letter=letter,
                        path=f"recordings/{letter}/{speaker}/session_001/rec_{speaker}_{letter}",
                        frame_count=60,
                        duration_ms=2000.0,
                        valid_frame_count=55,
                        quality_summary={},
                        preprocessing_version="1.0"
                    )
                    entries.append(entry)

            # Write manifest
            with open(manifest_path, 'w') as f:
                for entry in entries:
                    f.write(entry.to_jsonl() + '\n')

            # Create splits
            splits = create_splits(
                manifest_path,
                train_ratio=0.5,
                val_ratio=0.25,
                test_ratio=0.25,
                random_seed=42,
                min_speakers_per_split=1
            )

            # Check all recordings have splits
            assert len(splits) == 8

            # Check speaker independence (splits are keyed by relative path)
            speaker_splits = {}
            for entry in entries:
                split = splits[entry.path]
                if entry.speaker_id in speaker_splits:
                    assert speaker_splits[entry.speaker_id] == split, "Speaker in multiple splits"
                else:
                    speaker_splits[entry.speaker_id] = split
            assert len(speaker_splits) == 4  # 4 speakers, each in exactly one split

    def test_min_speakers_per_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.jsonl"

            # Only 3 speakers
            entries = []
            for speaker in ["speaker_001", "speaker_002", "speaker_003"]:
                entry = ManifestEntry(
                    recording_id=f"rec_{speaker}",
                    speaker_id=speaker,
                    session_id="session_001",
                    letter="A",
                    path=f"recordings/A/{speaker}/session_001/rec_{speaker}",
                    frame_count=60,
                    duration_ms=2000.0,
                    valid_frame_count=55,
                    quality_summary={},
                    preprocessing_version="1.0"
                )
                entries.append(entry)

            with open(manifest_path, 'w') as f:
                for entry in entries:
                    f.write(entry.to_jsonl() + '\n')

            splits = create_splits(
                manifest_path,
                train_ratio=0.34,
                val_ratio=0.33,
                test_ratio=0.33,
                random_seed=42,
                min_speakers_per_split=1
            )

            # With 3 speakers and min 1 per split, all should get at least 1
            speaker_splits = {}
            for entry in entries:
                split = splits[entry.path]
                speaker_splits[entry.speaker_id] = split

            assert len(set(speaker_splits.values())) == 3  # All 3 splits used


class TestSaveSplitInfo:
    """Test saving split information."""

    def test_save_split_info(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.jsonl"
            output_path = Path(tmpdir) / "splits.json"

            # Create simple manifest
            entries = [
                ManifestEntry(
                    recording_id="rec_1", speaker_id="speaker_001", session_id="session_001",
                    letter="A", path="p1", frame_count=10, duration_ms=100, valid_frame_count=10,
                    quality_summary={}, preprocessing_version="1.0"
                ),
                ManifestEntry(
                    recording_id="rec_2", speaker_id="speaker_002", session_id="session_001",
                    letter="A", path="p2", frame_count=10, duration_ms=100, valid_frame_count=10,
                    quality_summary={}, preprocessing_version="1.0"
                ),
            ]
            with open(manifest_path, 'w') as f:
                for e in entries:
                    f.write(e.to_jsonl() + '\n')

            splits = {"p1": "train", "p2": "test"}
            save_split_info(splits, manifest_path, output_path, 42)

            assert output_path.exists()
            with open(output_path) as f:
                info = json.load(f)

            assert info["split_seed"] == 42
            assert set(info["train_speakers"]) == {"speaker_001"}
            assert set(info["test_speakers"]) == {"speaker_002"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])