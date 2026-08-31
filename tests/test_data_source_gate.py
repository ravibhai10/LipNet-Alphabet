"""
Tests for the hard data-source production gate (§6 Phase 6).

The validator MUST:
- Reject entries with data_source == DATA_SOURCE_UNSPECIFIED
- Reject entries with data_source == DATA_SOURCE_SYNTHETIC_TEST
- Reject metadata.json files missing the data_source field
- Accept entries with data_source == DATA_SOURCE_REAL_CAMERA
"""
import json
import tempfile
from pathlib import Path

import pytest

from data.schema import (
    ManifestEntry,
    DATA_SOURCE_REAL_CAMERA,
    DATA_SOURCE_SYNTHETIC_TEST,
)
from data.validate import DatasetValidator, validate_dataset


class TestDataSourceGate:

    def _make_entry(
        self,
        recording_id,
        data_source,
        metadata_data_source,
        dataset_root,
    ):
        """Build a minimal ManifestEntry + on-disk metadata.json pair.

        metadata_data_source controls what gets written to the on-disk
        metadata.json: if None, no data_source field is written (simulating
        legacy synthetic recordings).
        """
        rec_dir = dataset_root / recording_id
        rec_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "letter": "A",
            "speaker_id": "speaker_001",
            "session_id": "session_001",
            "recording_id": recording_id,
            "source": "webcam",
            "created_at": "2026-01-01T00:00:00Z",
            "fps": 30.0,
            "frame_count": 60,
            "width": 1280,
            "height": 720,
            "duration_ms": 2000.0,
            "face_tracking": {
                "frames_with_face": 55, "frames_without_face": 5,
                "max_consecutive_missing": 2, "mean_confidence": 0.9,
                "min_confidence": 0.7,
            },
            "mouth": {
                "frames_with_mouth": 50, "missing_frames": 10,
                "mean_mouth_width": 100, "mean_mouth_height": 50,
                "min_mouth_area": 2000, "max_mouth_area": 8000,
            },
            "quality": {
                "mean_brightness": 128, "std_brightness": 20,
                "mean_blur": 150, "mean_mouth_size": 5000,
                "mean_pose_yaw": 0, "mean_pose_pitch": 0, "mean_pose_roll": 0,
                "duplicate_frame_count": 0, "quality_flags": [],
            },
            "preprocessing_version": "1.0",
        }
        if metadata_data_source is not None:
            meta["data_source"] = metadata_data_source
        with open(rec_dir / "metadata.json", "w") as f:
            json.dump(meta, f)
        return ManifestEntry(
            recording_id=recording_id,
            speaker_id="speaker_001",
            session_id="session_001",
            letter="A",
            path=recording_id,
            frame_count=60,
            duration_ms=2000.0,
            valid_frame_count=55,
            quality_summary={},
            preprocessing_version="1.0",
            data_source=data_source,
        )

    def test_real_camera_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "ds"
            (dataset_root / "recordings").mkdir(parents=True)
            entry = self._make_entry(
                recording_id="rec_001",
                data_source=DATA_SOURCE_REAL_CAMERA,
                metadata_data_source=DATA_SOURCE_REAL_CAMERA,
                dataset_root=dataset_root,
            )
            validator = DatasetValidator(dataset_root)
            result = validator._check_data_source([entry])
            assert result.passed, f"Expected pass, got: {result.message}"
            assert "real_camera" in result.message

    def test_synthetic_test_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "ds"
            (dataset_root / "recordings").mkdir(parents=True)
            entry = self._make_entry(
                recording_id="rec_001",
                data_source=DATA_SOURCE_SYNTHETIC_TEST,
                metadata_data_source=DATA_SOURCE_SYNTHETIC_TEST,
                dataset_root=dataset_root,
            )
            validator = DatasetValidator(dataset_root)
            result = validator._check_data_source([entry])
            assert not result.passed
            assert "PRODUCTION DATA GATE FAILED" in result.message
            assert any("synthetic" in d for d in result.details)

    def test_unspecified_sentinel_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "ds"
            (dataset_root / "recordings").mkdir(parents=True)
            entry = self._make_entry(
                recording_id="rec_001",
                data_source="__UNSPECIFIED__",
                metadata_data_source=None,
                dataset_root=dataset_root,
            )
            validator = DatasetValidator(dataset_root)
            result = validator._check_data_source([entry])
            assert not result.passed
            assert "PRODUCTION DATA GATE FAILED" in result.message

    def test_metadata_json_missing_field_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "ds"
            (dataset_root / "recordings").mkdir(parents=True)
            entry = self._make_entry(
                recording_id="rec_001",
                data_source=DATA_SOURCE_REAL_CAMERA,
                metadata_data_source=None,
                dataset_root=dataset_root,
            )
            validator = DatasetValidator(dataset_root)
            result = validator._check_data_source([entry])
            assert not result.passed
            assert any("NO data_source" in d for d in result.details)

    def test_unknown_data_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "ds"
            (dataset_root / "recordings").mkdir(parents=True)
            entry = self._make_entry(
                recording_id="rec_001",
                data_source="something_random",
                metadata_data_source="something_random",
                dataset_root=dataset_root,
            )
            validator = DatasetValidator(dataset_root)
            result = validator._check_data_source([entry])
            assert not result.passed
            assert any("unknown" in d for d in result.details)

    def test_synthetic_dataset_in_isolated_dir(self):
        """The on-disk synthetic_dataset/ must fail validation when scanned."""
        synthetic_root = (
            Path("c:/Users/ravik/OneDrive/Desktop/lip alphabet")
            / "synthetic_dataset"
        )
        if not synthetic_root.exists():
            pytest.skip("synthetic_dataset/ not present on disk")
        rc = validate_dataset(str(synthetic_root))
        assert rc != 0, "synthetic_dataset/ must fail validation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
