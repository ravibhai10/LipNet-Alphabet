"""
Integration tests for the full Phase 2 pipeline.

Tests end-to-end workflows using temporary test datasets.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from data.schema import (
    RecordingMetadata, FaceTrackingInfo, MouthInfo, QualityMetrics,
    ManifestEntry, Letter, Split,
    DATA_SOURCE_SYNTHETIC_TEST, DATA_SOURCE_REAL_CAMERA,
)
import numpy as np

from data.ingestion import VideoIngestor, IngestionConfig, get_video_info, validate_video
from data.manifest import ManifestManager, ManifestConfig, create_splits, save_split_info
from data.validate import DatasetValidator, validate_dataset
from data.inspect import DatasetInspector, inspect_dataset


class TestIntegration:
    """Integration tests for Phase 2 components."""

    def create_full_test_dataset(self, tmpdir: Path) -> Path:
        """Create a complete test dataset with multiple recordings."""
        dataset_root = tmpdir / "integration_test"
        recordings_root = dataset_root / "recordings"

        # Create 2 letters, 3 speakers, 2 sessions, 2 recordings each = 24 recordings
        for letter in ["A", "B"]:
            for speaker_idx in range(3):
                speaker = f"speaker_{speaker_idx+1:03d}"
                for session_idx in range(2):
                    session = f"session_{session_idx+1:03d}"
                    for rec_idx in range(2):
                        rec_id = f"recording_{rec_idx+1:03d}"
                        rec_dir = recordings_root / letter / speaker / session / rec_id
                        rec_dir.mkdir(parents=True)

                        metadata = RecordingMetadata(
                            letter=letter,
                            speaker_id=speaker,
                            session_id=session,
                            recording_id=rec_id,
                            source="webcam",
                            data_source=DATA_SOURCE_REAL_CAMERA,
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
                        (rec_dir / "video.mp4").write_bytes(b"dummy video")

                        # Create frame data
                        frames_dir = rec_dir / "frames"
                        frames_dir.mkdir()
                        frame_data = [
                            {"frame_index": i, "timestamp_ms": i * 33.33,
                             "mouth_bbox": [0.3, 0.4, 0.2, 0.1], "head_pose": [0, 0, 0],
                             "quality_flags": []}
                            for i in range(60)
                        ]
                        with open(frames_dir / "frame_data.json", 'w') as f:
                            json.dump(frame_data, f)

                        # Create landmarks
                        landmarks_dir = rec_dir / "landmarks"
                        landmarks_dir.mkdir()
                        for name in ["mouth_landmarks.npz", "mouth_bboxes.npz", "head_poses.npz"]:
                            np.savez_compressed(landmarks_dir / name,
                                **{f"frame_{i:06d}": np.zeros((20, 2) if "landmarks" in name else (4,) if "bboxes" in name else (3,))
                                   for i in range(60)})

        return dataset_root

    @patch('data.validate.validate_video')
    @patch('data.validate.get_video_info')
    def test_full_validation_pipeline(self, mock_get_info, mock_validate):
        """Test full validation pipeline on test dataset."""
        mock_get_info.return_value = MagicMock(
            fps=30, frame_count=60, width=1280, height=720,
            duration_ms=2000, codec="mp4v", is_valid=True
        )
        mock_validate.return_value = (True, None)

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = self.create_full_test_dataset(Path(tmpdir))

            # Build manifest
            manager = ManifestManager(ManifestConfig(
                dataset_root=dataset_root,
                manifest_path=dataset_root / "manifest.jsonl",
            ))
            entries = manager.build_manifest()

            # Verify manifest
            assert len(entries) == 24  # 2 * 3 * 2 * 2

            # Create splits
            splits = create_splits(
                dataset_root / "manifest.jsonl",
                train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                random_seed=42, min_speakers_per_split=1
            )
            assert len(splits) == 24

            # Rebuild manifest with splits
            entries = manager.build_manifest(splits)
            split_entries = [e for e in entries if e.split]
            assert len(split_entries) == 24

            # Save split info
            save_split_info(splits, dataset_root / "manifest.jsonl",
                          dataset_root / "splits.json", 42)

            # Validate
            validator = DatasetValidator(dataset_root)
            report = validator.validate_all()

            assert not report.has_critical_errors()
            print(f"Validation passed with {len(report.results)} checks")

    def test_inspection_pipeline(self):
        """Test inspection pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = self.create_full_test_dataset(Path(tmpdir))

            # Build manifest
            manager = ManifestManager(ManifestConfig(
                dataset_root=dataset_root,
                manifest_path=dataset_root / "manifest.jsonl",
            ))
            manager.build_manifest()

            # Inspect
            inspector = DatasetInspector(dataset_root)
            stats = inspector.inspect()

            assert stats["total_recordings"] == 24
            assert stats["total_speakers"] == 3
            assert stats["total_sessions"] == 6
            assert stats["recordings_per_letter"]["A"] == 12
            assert stats["recordings_per_letter"]["B"] == 12

    def test_ingestion_to_manifest(self):
        """Test ingestion creates proper directory structure and metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset"
            dataset_root.mkdir()

            config = IngestionConfig(
                dataset_root=dataset_root,
                fps=30,
                max_sequence_length=76,
                min_sequence_length=15,
            )

            # We can't test actual video ingestion without a real video/webcam
            # but we can test the directory structure creation
            ingestor = VideoIngestor(config)

            # Verify directory structure would be created correctly
            recordings_root = dataset_root / "recordings"
            assert ingestor.recordings_root == recordings_root

            # Test recording ID generation
            rec_id1 = ingestor._generate_recording_id("A", "speaker_001", "session_001")
            assert rec_id1 == "recording_001"
            # Real ingestion creates the directory before the next ID is
            # generated; mirror that so the generator sees existing recordings.
            (ingestor.recordings_root / "A" / "speaker_001" / "session_001" / rec_id1).mkdir(parents=True)
            rec_id2 = ingestor._generate_recording_id("A", "speaker_001", "session_001")
            assert rec_id2 == "recording_002"


import json
from unittest.mock import MagicMock


if __name__ == "__main__":
    pytest.main([__file__, "-v"])