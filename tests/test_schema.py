
"""
Tests for dataset schema and metadata validation.
"""

import pytest
import json
import tempfile
from pathlib import Path
import numpy as np
from data.schema import (
    RecordingMetadata, FaceTrackingInfo, MouthInfo, QualityMetrics,
    ManifestEntry, FrameData, Letter, Split, QualityFlag,
    validate_metadata, validate_manifest_entry, validate_frame_ordering,
    SpeakerSplit, DatasetSplitInfo,
    DATA_SOURCE_SYNTHETIC_TEST, DATA_SOURCE_REAL_CAMERA,
)


class TestLetter:
    def test_all_letters_valid(self):
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            assert Letter.is_valid(letter)
    def test_invalid_letters(self):
        assert not Letter.is_valid("a")
        assert not Letter.is_valid("AA")
        assert not Letter.is_valid("1")
        assert not Letter.is_valid("")


class TestRecordingMetadata:
    def create_minimal_metadata(self):
        return RecordingMetadata(
            letter="A", speaker_id="speaker_001", session_id="session_001",
            recording_id="recording_001", source="webcam",
            data_source=DATA_SOURCE_REAL_CAMERA,
            created_at="2026-01-01T00:00:00Z", fps=30.0, frame_count=60,
            width=1280, height=720, duration_ms=2000.0,
            face_tracking=FaceTrackingInfo(
                frames_with_face=55, frames_without_face=5,
                max_consecutive_missing=2, mean_confidence=0.9, min_confidence=0.7),
            mouth=MouthInfo(
                frames_with_mouth=50, missing_frames=10,
                mean_mouth_width=100.0, mean_mouth_height=50.0,
                min_mouth_area=2000.0, max_mouth_area=8000.0),
            quality=QualityMetrics(
                mean_brightness=128.0, std_brightness=20.0, mean_blur=150.0,
                mean_mouth_size=5000.0, mean_pose_yaw=0.0, mean_pose_pitch=0.0,
                mean_pose_roll=0.0, duplicate_frame_count=0, quality_flags=[]),
            preprocessing_version="1.0", schema_version="1.0")
    def test_valid_metadata(self):
        metadata = self.create_minimal_metadata()
        errors = validate_metadata(metadata)
        assert len(errors) == 0
    def test_invalid_letter(self):
        metadata = self.create_minimal_metadata()
        metadata.letter = "INVALID"
        errors = validate_metadata(metadata)
        assert any("Invalid letter" in e for e in errors)
    def test_missing_speaker_id(self):
        metadata = self.create_minimal_metadata()
        metadata.speaker_id = ""
        errors = validate_metadata(metadata)
        assert any("Missing speaker_id" in e for e in errors)
    def test_invalid_fps(self):
        metadata = self.create_minimal_metadata()
        metadata.fps = 0
        errors = validate_metadata(metadata)
        assert any("Invalid fps" in e for e in errors)
    def test_frame_count_mismatch(self):
        metadata = self.create_minimal_metadata()
        metadata.face_tracking.frames_with_face = 100
        errors = validate_metadata(metadata)
        assert any("mismatch" in e for e in errors)
    def test_json_serialization(self):
        metadata = self.create_minimal_metadata()
        d = metadata.to_dict()
        loaded = RecordingMetadata.from_dict(d)
        assert loaded.letter == metadata.letter
        assert loaded.speaker_id == metadata.speaker_id

    def test_data_source_required(self):
        """data_source must be explicitly set; missing/synthetic fails validation."""
        metadata = self.create_minimal_metadata()
        metadata.data_source = "real_camera"
        errors = validate_metadata(metadata)
        assert not any("data_source" in e or "REQUIRED" in e for e in errors)
        metadata.data_source = "synthetic_test"
        errors = validate_metadata(metadata)
        assert any("real_camera" in e for e in errors)

    def test_data_source_unspecified(self):
        """Unspecified sentinel fails validation."""
        metadata = self.create_minimal_metadata()
        metadata.data_source = "__UNSPECIFIED__"
        errors = validate_metadata(metadata)
        assert any("REQUIRED" in e or "UNSPECIFIED" in e for e in errors)


class TestManifestEntry:
    def create_entry(self):
        return ManifestEntry(
            recording_id="rec001", speaker_id="speaker_001", session_id="session_001",
            letter="A", path="recordings/A/speaker_001/session_001/rec001",
            data_source=DATA_SOURCE_REAL_CAMERA, quality_summary={}, landmark_source="mediapipe",
            frame_count=60, duration_ms=2000.0, valid_frame_count=55,
            preprocessing_version="1.0", split="train", schema_version="1.0")
    def test_valid_entry(self):
        entry = self.create_entry()
        errors = validate_manifest_entry(entry)
        assert len(errors) == 0
    def test_invalid_letter(self):
        entry = self.create_entry()
        entry.letter = "INVALID"
        errors = validate_manifest_entry(entry)
        assert any("Invalid letter" in e for e in errors)
    def test_invalid_split(self):
        entry = self.create_entry()
        entry.split = "invalid"
        errors = validate_manifest_entry(entry)
        assert any("Invalid split" in e for e in errors)
    def test_jsonl_serialization(self):
        entry = self.create_entry()
        jsonl = entry.to_jsonl()
        loaded = ManifestEntry.from_jsonl(jsonl)
        assert loaded.recording_id == entry.recording_id
        assert loaded.letter == entry.letter


class TestFrameData:
    def test_valid_ordering(self):
        frames = [
            FrameData(frame_index=0, timestamp_ms=0.0),
            FrameData(frame_index=1, timestamp_ms=33.33),
            FrameData(frame_index=2, timestamp_ms=66.66),
        ]
        errors = validate_frame_ordering(frames)
        assert len(errors) == 0
    def test_frame_index_mismatch(self):
        frames = [
            FrameData(frame_index=0, timestamp_ms=0.0),
            FrameData(frame_index=2, timestamp_ms=33.33),
        ]
        errors = validate_frame_ordering(frames)
        assert any("Frame index mismatch" in e for e in errors)
    def test_non_monotonic_timestamp(self):
        frames = [
            FrameData(frame_index=0, timestamp_ms=100.0),
            FrameData(frame_index=1, timestamp_ms=50.0),
        ]
        errors = validate_frame_ordering(frames)
        assert any("Timestamp not monotonic" in e for e in errors)
    def test_empty_frames(self):
        frames = []
        errors = validate_frame_ordering(frames)
        assert len(errors) == 0


class TestSpeakerSplit:
    def test_serialization(self):
        split = SpeakerSplit(
            speaker_id="speaker_001", split=Split.TRAIN,
            sessions=["session_001", "session_002"])
        d = split.to_dict()
        loaded = SpeakerSplit.from_dict(d)
        assert loaded.speaker_id == split.speaker_id
        assert loaded.split == split.split
        assert loaded.sessions == split.sessions


class TestDatasetSplitInfo:
    def test_serialization(self):
        info = DatasetSplitInfo(
            train_speakers=["speaker_001", "speaker_002"], val_speakers=["speaker_003"],
            test_speakers=["speaker_004"],
            train_sessions=["speaker_001/session_001"],
            val_sessions=["speaker_003/session_001"],
            test_sessions=["speaker_004/session_001"],
            split_seed=42, created_at="2026-01-01T00:00:00Z")
        d = info.to_dict()
        loaded = DatasetSplitInfo.from_dict(d)
        assert loaded.train_speakers == info.train_speakers
        assert loaded.split_seed == info.split_seed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
