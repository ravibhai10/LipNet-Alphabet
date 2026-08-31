"""Tests for the real-dataset gate in TrainingService.

Production training is forbidden unless:
    - data_source is real_camera
    - landmark_source is mediapipe
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from app.services.training_service import (
    TrainingService, PreTrainingCheck,
)


def _write_manifest(path: Path, entries: list) -> None:
    """Write a JSONL manifest from a list of dicts."""
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _make_entry(
    recording_id: str = "r1",
    data_source: str = "real_camera",
    landmark_source: str = "mediapipe",
    split: str = "train",
    letter: str = "A",
) -> dict:
    return {
        "recording_id": recording_id,
        "speaker_id": "spk1",
        "session_id": "sess1",
        "letter": letter,
        "split": split,
        "path": f"recordings/{letter}/spk1/sess1/{recording_id}",
        "frame_count": 30,
        "duration_ms": 1000.0,
        "valid_frame_count": 30,
        "quality_summary": {},
        "preprocessing_version": "1.0",
        "data_source": data_source,
        "landmark_source": landmark_source,
    }


class TestRealDatasetGate:
    """Direct test of the gate logic."""

    def _service(self):
        return TrainingService()

    def test_gate_passes_for_real_data(self, tmp_path):
        manifest = tmp_path / "manifest.jsonl"
        _write_manifest(manifest, [
            _make_entry("r1", "real_camera", "mediapipe"),
            _make_entry("r2", "real_camera", "mediapipe", split="val"),
        ])
        result = self._service()._check_real_dataset_gate(tmp_path)
        assert result["passed"] is True
        assert result["errors"] == []

    def test_gate_rejects_synthetic(self, tmp_path):
        manifest = tmp_path / "manifest.jsonl"
        _write_manifest(manifest, [
            _make_entry("r1", "synthetic_dataset", "mediapipe"),
        ])
        result = self._service()._check_real_dataset_gate(tmp_path)
        assert result["passed"] is False
        assert any("REAL_DATASET_GATE_FAILED" in e for e in result["errors"])

    def test_gate_rejects_non_mediapipe(self, tmp_path):
        manifest = tmp_path / "manifest.jsonl"
        _write_manifest(manifest, [
            _make_entry("r1", "real_camera", "synthetic"),
        ])
        result = self._service()._check_real_dataset_gate(tmp_path)
        assert result["passed"] is False
        assert any("LANDMARK_SOURCE_GATE_FAILED" in e for e in result["errors"])

    def test_gate_rejects_mixed(self, tmp_path):
        manifest = tmp_path / "manifest.jsonl"
        _write_manifest(manifest, [
            _make_entry("r1", "real_camera", "mediapipe"),
            _make_entry("r2", "synthetic_dataset", "mediapipe"),
            _make_entry("r3", "real_camera", "dlib"),
        ])
        result = self._service()._check_real_dataset_gate(tmp_path)
        assert result["passed"] is False
        assert any("REAL_DATASET_GATE_FAILED" in e for e in result["errors"])
        assert any("LANDMARK_SOURCE_GATE_FAILED" in e for e in result["errors"])

    def test_gate_fails_without_manifest(self, tmp_path):
        result = self._service()._check_real_dataset_gate(tmp_path)
        assert result["passed"] is False
        assert any("MANIFEST_MISSING" in e for e in result["errors"])

    def test_gate_fails_with_empty_manifest(self, tmp_path):
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text("", encoding="utf-8")
        result = self._service()._check_real_dataset_gate(tmp_path)
        assert result["passed"] is False
        assert any("MANIFEST_EMPTY" in e for e in result["errors"])

    def test_gate_rejects_missing_data_source_field(self, tmp_path):
        manifest = tmp_path / "manifest.jsonl"
        e = _make_entry("r1", "real_camera", "mediapipe")
        del e["data_source"]
        _write_manifest(manifest, [e])
        result = self._service()._check_real_dataset_gate(tmp_path)
        assert result["passed"] is False