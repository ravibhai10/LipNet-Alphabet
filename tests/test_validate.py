"""
Tests for dataset validation command.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from data.schema import RecordingMetadata, ManifestEntry, FaceTrackingInfo, MouthInfo, QualityMetrics, DATA_SOURCE_SYNTHETIC_TEST, DATA_SOURCE_REAL_CAMERA
from data.validate import DatasetValidator, ValidationResult, ValidationReport, validate_dataset


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_creation(self):
        result = ValidationResult(
            check_name="test_check",
            passed=True,
            message="All good",
            severity="info"
        )
        assert result.check_name == "test_check"
        assert result.passed is True
        assert result.severity == "info"

    def test_with_details(self):
        result = ValidationResult(
            check_name="test_check",
            passed=False,
            message="Failed",
            severity="error",
            details=["detail1", "detail2"]
        )
        assert len(result.details) == 2


class TestValidationReport:
    """Test ValidationReport."""

    def test_add_results(self):
        report = ValidationReport(results=[])
        report.add_result(ValidationResult("check1", True, "ok", "info"))
        report.add_result(ValidationResult("check2", False, "fail", "error"))
        report.add_result(ValidationResult("check3", False, "warn", "warning"))

        assert report.critical_errors == 1
        assert report.warnings == 1
        assert report.has_critical_errors() is True

    def test_summary(self):
        report = ValidationReport(results=[])
        report.add_result(ValidationResult("check1", True, "ok", "info"))
        summary = report.summary()
        assert "Validation Report" in summary
        assert "PASS" in summary


class TestDatasetValidator:
    """Test DatasetValidator."""

    def create_test_dataset(self, tmpdir: Path) -> Path:
        """Create a minimal test dataset."""
        dataset_root = tmpdir / "test_dataset"
        recordings_root = dataset_root / "recordings"

        # Create A/speaker_001/session_001/recording_001
        for letter in ["A", "B"]:
            for speaker in ["speaker_001", "speaker_002"]:
                for session in ["session_001"]:
                    rec_dir = recordings_root / letter / speaker / session / "recording_001"
                    rec_dir.mkdir(parents=True)

                    metadata = RecordingMetadata(
                        letter=letter,
                        speaker_id=speaker,
                        session_id=session,
                        recording_id="recording_001",
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
                    (rec_dir / "video.mp4").write_bytes(b"dummy")

                    # Create frame_data.json
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
                    import numpy as np
                    for name in ["mouth_landmarks.npz", "mouth_bboxes.npz", "head_poses.npz"]:
                        np.savez_compressed(landmarks_dir / name,
                            **{f"frame_{i:06d}": np.zeros((20, 2) if "landmarks" in name else (4,) if "bboxes" in name else (3,))
                               for i in range(60)})

        return dataset_root

    @patch('data.validate.validate_video')
    @patch('data.validate.get_video_info')
    def test_validate_all_pass(self, mock_get_info, mock_validate):
        mock_get_info.return_value = MagicMock(
            fps=30, frame_count=60, width=1280, height=720,
            duration_ms=2000, codec="mp4v", is_valid=True
        )
        mock_validate.return_value = (True, None)

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = self.create_test_dataset(Path(tmpdir))

            # Create manifest
            from data.manifest import ManifestManager, ManifestConfig
            manager = ManifestManager(ManifestConfig(
                dataset_root=dataset_root,
                manifest_path=dataset_root / "manifest.jsonl",
            ))
            manager.build_manifest()

            validator = DatasetValidator(dataset_root)
            report = validator.validate_all()

            # Should pass all checks
            assert not report.has_critical_errors()
            assert len(report.results) > 0

    def test_validate_missing_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "empty_dataset"
            dataset_root.mkdir()

            validator = DatasetValidator(dataset_root)
            report = validator.validate_all()

            assert report.has_critical_errors()
            assert any("manifest_exists" in r.check_name and not r.passed for r in report.results)

    def test_validate_invalid_letter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "test_dataset"
            recordings_root = dataset_root / "recordings"

            # Create recording with a valid letter directory but malformed metadata
            # (metadata letter invalid) so the validator can surface the error.
            rec_dir = recordings_root / "A" / "speaker_001" / "session_001" / "recording_001"
            rec_dir.mkdir(parents=True)

            metadata = RecordingMetadata(
letter="INVALID",
                speaker_id="speaker_001",
                session_id="session_001",
                recording_id="recording_001",
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
            (rec_dir / "video.mp4").write_bytes(b"dummy")

            from data.manifest import ManifestManager, ManifestConfig
            manager = ManifestManager(ManifestConfig(
                dataset_root=dataset_root,
                manifest_path=dataset_root / "manifest.jsonl",
            ))
            manager.build_manifest()

            validator = DatasetValidator(dataset_root)
            report = validator.validate_all()

            assert report.has_critical_errors()
            assert any("valid_letters" in r.check_name and not r.passed for r in report.results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])