"""
Tests for full preprocessing pipeline.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from data.schema import RecordingMetadata, FaceTrackingInfo, MouthInfo, QualityMetrics, DATA_SOURCE_SYNTHETIC_TEST
from data.preprocess import PreprocessingPipeline, PreprocessingConfig


class TestPreprocessingPipeline:
    """Test preprocessing pipeline."""

    def create_test_recording(self, tmpdir: Path) -> Path:
        """Create a test recording with metadata."""
        dataset_root = tmpdir / "test_dataset"
        recordings_root = dataset_root / "recordings"

        rec_dir = recordings_root / "A" / "speaker_001" / "session_001" / "recording_001"
        rec_dir.mkdir(parents=True)

        metadata = RecordingMetadata(
letter="A",
            speaker_id="speaker_001",
            session_id="session_001",
            recording_id="recording_001",
                            data_source=DATA_SOURCE_SYNTHETIC_TEST,
            source="webcam",
            created_at="2026-01-01T00:00:00Z",
            fps=30.0,
            frame_count=10,
            width=1280,
            height=720,
            duration_ms=333.0,
            face_tracking=FaceTrackingInfo(
                frames_with_face=0, frames_without_face=0,
                max_consecutive_missing=0, mean_confidence=0.0, min_confidence=1.0
            ),
            mouth=MouthInfo(
                frames_with_mouth=0, missing_frames=0,
                mean_mouth_width=0, mean_mouth_height=0,
                min_mouth_area=0, max_mouth_area=0
            ),
            quality=QualityMetrics(
                mean_brightness=0, std_brightness=0,
                mean_blur=0, mean_mouth_size=0,
                mean_pose_yaw=0, mean_pose_pitch=0, mean_pose_roll=0,
                duplicate_frame_count=0, quality_flags=[]
            ),
            preprocessing_version="1.0"
        )
        metadata.save(rec_dir / "metadata.json")
        (rec_dir / "video.mp4").write_bytes(b"dummy video")

        return dataset_root

    @patch('data.preprocess.TrackingPipeline.process_recording')
    def test_process_recording(self, mock_process):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = self.create_test_recording(Path(tmpdir))

            config = PreprocessingConfig(dataset_root=dataset_root)
            pipeline = PreprocessingPipeline(config)

            rec_dir = dataset_root / "recordings" / "A" / "speaker_001" / "session_001" / "recording_001"
            metadata = RecordingMetadata.load(rec_dir / "metadata.json")

            # Mock the tracking pipeline to avoid needing MediaPipe
            mock_metadata = metadata
            mock_metadata.face_tracking.frames_with_face = 10
            mock_process.return_value = mock_metadata

            result = pipeline.process_recording(rec_dir, metadata)

            assert result.face_tracking.frames_with_face == 10
            mock_process.assert_called_once()

    def test_is_processed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = self.create_test_recording(Path(tmpdir))

            config = PreprocessingConfig(dataset_root=dataset_root)
            pipeline = PreprocessingPipeline(config)

            rec_dir = dataset_root / "recordings" / "A" / "speaker_001" / "session_001" / "recording_001"

            # Not processed yet
            assert pipeline._is_processed(rec_dir) is False

            # Create required files
            landmarks_dir = rec_dir / "landmarks"
            landmarks_dir.mkdir()
            for name in ["mouth_landmarks.npz", "mouth_bboxes.npz", "head_poses.npz"]:
                (landmarks_dir / name).write_bytes(b"dummy")

            frames_dir = rec_dir / "frames"
            frames_dir.mkdir()
            (frames_dir / "frame_data.json").write_text("[]")

            # Now should be processed
            assert pipeline._is_processed(rec_dir) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])