"""Unit tests for the GRID ingestion + manifest tooling (offline, no corpus)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.grid.builder import _find_media, _guess_fps_and_frames, _infer_utterance_id
from pipeline.grid.manifest import GRIDAlignmentParser, AlignmentToken
from pipeline.grid.schema import (
    AlignmentStatus,
    GridManifestEntry,
    GridValidationReport,
    ValidationStatus,
)
from pipeline.grid.validate import GridValidator


def test_alignment_token_roundtrip():
    t = AlignmentToken(word="red", start_ms=100, end_ms=1050)
    assert t.duration_ms == 950
    assert t.word == "red"


def test_alignment_parser():
    import tempfile
    from pathlib import Path as _P
    data = "0\t1000\tput\n1100\t2000\tred\n2100\t3000\tat"
    with tempfile.NamedTemporaryFile(suffix=".align", delete=False) as f:
        f.write(data.encode())
        f.flush()
        p = GRIDAlignmentParser(_P(f.name))
    parsed = p.parse()
    assert len(parsed.tokens) == 3
    assert parsed.tokens[0].word == "put"
    assert parsed.tokens[0].start_ms == 0
    assert parsed.tokens[2].end_ms == 3000


def test_alignment_parser_missing():
    p = GRIDAlignmentParser(Path("/tmp/does_not_exist.align"))
    parsed = p.parse()
    assert parsed.valid() is False
    assert parsed.errors


def test_find_media(tmp_path):
    (tmp_path / "audio").mkdir()
    (tmp_path / "video").mkdir()
    (tmp_path / "alignment").mkdir()
    (tmp_path / "audio" / "a.wav").write_bytes(b"x")
    (tmp_path / "video" / "v.mp4").write_bytes(b"x")
    (tmp_path / "alignment" / "a.align").write_bytes(b"x")
    v, a, al = _find_media(tmp_path)
    assert v.name == "v.mp4"
    assert a.name == "a.wav"
    assert al.name == "a.align"


def test_infer_utterance_id():
    from pathlib import Path
    # name with digits -> zero-pad
    assert _infer_utterance_id("T01", Path("/tmp/T01_U0001.mp4")) == "T01_U0001"
    # name without digits: zero-pad the name to 4 + readdir suffix
    assert _infer_utterance_id("T02", Path("/tmp/rec.mp4")) == "T02_U0rec"


def test_guess_fps_and_frames(tmp_path):
    v = tmp_path / "v.mp4"
    v.write_bytes(b"")
    fps, frames = _guess_fps_and_frames(v)
    # synthetic file cannot be opened by cv2
    assert fps == 0.0
    assert frames == 0


def test_manifest_entry_serialization():
    e = GridManifestEntry(
        talk_id="T01", speaker_id="grid_T01", utterance_id="T01_U0001",
        video_path="T01_v.mp4", audio_path="audio/25kHz/T01_a.wav",
        alignment_path="T01.align", frame_count=0, fps=0.0,
    )
    d = e.to_dict()
    assert d["utterance_id"] == "T01_U0001"
    e2 = GridManifestEntry.from_dict(d)
    assert e2.talk_id == "T01"


def test_grid_manifest_entry_roundtrip_json():
    e = GridManifestEntry(
        talk_id="T01", speaker_id="grid_T01", utterance_id="T01_U0001",
        video_path="T01_v.mp4", audio_path="audio/25kHz/T01_a.wav",
        alignment_path="T01.align", frame_count=0, fps=0.0,
        transcript="put red at 9 now", alignment_status=AlignmentStatus.OK,
        validation_status=ValidationStatus.OK, value="put red at 9 now",
    )
    s = e.to_json()
    e2 = GridManifestEntry.from_json(s)
    assert e2.utterance_id == "T01_U0001"


def test_grid_validation_report():
    base = dict(
        talk_id="T01", speaker_id="grid_T01", video_path="v.mp4", audio_path="a.wav", alignment_path="a.align",
        frame_count=0, fps=0.0,
    )
    rep = GridValidationReport(talk_id="T01")
    rep.record(GridManifestEntry(**base, utterance_id="U1", validation_status=ValidationStatus.OK))
    rep.record(GridManifestEntry(**base, utterance_id="U2", validation_status=ValidationStatus.MISSING_VIDEO))
    assert rep.total == 2
    assert rep.ok == 1
    assert rep.failed == 1
    assert rep.has_critical_errors() is True
