"""Unit tests for the GRID ingestion + manifest tooling (offline, no corpus)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.grid.builder import (
    GridManifestBuilder,
    _find_media,
    _guess_fps_and_frames,
    _infer_utterance_id,
)
from pipeline.grid.manifest import (
    GRIDAlignmentParser,
    AlignmentToken,
    write_manifest_jsonl,
)
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


def test_find_media_supports_mpg(tmp_path):
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    video = video_dir / "utterance.mpg"
    video.write_bytes(b"x")

    found, _, _ = _find_media(tmp_path)

    assert found == video


def _write_utterance(root: Path, stem: str, *, video=True, audio=True, alignment=True):
    for folder, suffix, enabled in (
        ("video", ".mpg", video),
        ("audio", ".wav", audio),
        ("alignment", ".align", alignment),
    ):
        if enabled:
            directory = root / folder
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{stem}{suffix}"
            path.write_text("0 100 word\n", encoding="utf-8")


def test_builder_pairs_multiple_mpg_utterances_by_stem(tmp_path, monkeypatch):
    talk = tmp_path / "T01"
    _write_utterance(talk, "first")
    _write_utterance(talk, "second")
    monkeypatch.setattr("pipeline.grid.builder._guess_fps_and_frames", lambda _: (25.0, 75))

    entries = GridManifestBuilder(tmp_path).build()

    assert [entry.utterance_id for entry in entries] == ["T01_Ufirst", "T01_Usecond"]
    assert all(entry.is_ok for entry in entries)
    assert all(entry.video_path.endswith(".mpg") for entry in entries)
    assert all(Path(entry.audio_path).stem == Path(entry.video_path).stem for entry in entries)
    assert all(Path(entry.alignment_path).stem == Path(entry.video_path).stem for entry in entries)


def test_builder_marks_missing_modality_invalid(tmp_path, monkeypatch):
    talk = tmp_path / "T01"
    _write_utterance(talk, "missing_audio", audio=False)
    monkeypatch.setattr("pipeline.grid.builder._guess_fps_and_frames", lambda _: (25.0, 75))

    entries = GridManifestBuilder(tmp_path).build()

    assert len(entries) == 1
    assert entries[0].validation_status == ValidationStatus.MISSING_AUDIO
    assert entries[0].is_ok is False
    assert "missing audio" in entries[0].errors[0]


def test_builder_marks_duplicate_modality_invalid(tmp_path, monkeypatch):
    talk = tmp_path / "T01"
    _write_utterance(talk, "duplicate")
    duplicate = talk / "audio" / "duplicate.flac"
    duplicate.write_text("duplicate", encoding="utf-8")
    monkeypatch.setattr("pipeline.grid.builder._guess_fps_and_frames", lambda _: (25.0, 75))

    entries = GridManifestBuilder(tmp_path).build()

    assert len(entries) == 1
    assert entries[0].validation_status == ValidationStatus.CORRUPT_MEDIA
    assert entries[0].is_ok is False
    assert any("duplicate audio files" in error for error in entries[0].errors)


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


def test_manifest_writer_uses_physical_lines_and_omits_alignment_object(tmp_path):
    alignment_path = tmp_path / "sample.align"
    alignment_path.write_text("0 100 word\n", encoding="utf-8")
    alignment = GRIDAlignmentParser(alignment_path).parse()
    entry = GridManifestEntry(
        talk_id="T01",
        speaker_id="grid_T01",
        utterance_id="T01_U0001",
        video_path="T01\\video\\sample.mpg",
        audio_path="T01\\audio\\sample.wav",
        alignment_path="T01\\alignment\\sample.align",
        frame_count=75,
        fps=25.0,
        transcript="word",
        alignment=alignment,
    )
    output = write_manifest_jsonl([entry], tmp_path / "manifest.jsonl")

    lines = output.read_text(encoding="utf-8").splitlines(keepends=True)
    assert len(lines) == 1
    assert lines[0].endswith("\n")
    record = json.loads(lines[0])
    assert record["utterance_id"] == "T01_U0001"
    assert "alignment" not in record


def test_manifest_writer_rejects_invalid_entries_without_replacing_output(tmp_path):
    output = tmp_path / "manifest.jsonl"
    output.write_text("original\n", encoding="utf-8")
    entry = GridManifestEntry(
        talk_id="T01",
        speaker_id="grid_T01",
        utterance_id="T01_U0001",
        video_path="",
        audio_path="",
        alignment_path="",
        validation_status=ValidationStatus.MISSING_VIDEO,
    )

    with pytest.raises(ValueError, match="invalid entries"):
        write_manifest_jsonl([entry], output)

    assert output.read_text(encoding="utf-8") == "original\n"


def test_persisted_manifest_uses_real_newlines():
    """Regression test: persisted grid_manifest.jsonl must use real newlines,
    not literal backslash-n text, and must be parseable line-by-line."""
    manifest = Path("data/external/grid/s1/grid_manifest.jsonl")
    assert manifest.exists(), "persisted manifest must exist on disk"

    raw = manifest.read_text(encoding="utf-8")

    # No literal backslash-n sequences allowed anywhere
    assert "\\n" not in raw, "manifest must not contain literal \\n text"

    # Every physical line must parse as independent JSON
    lines = [ln for ln in raw.split("\n") if ln.strip()]
    assert len(lines) == 1000, f"expected 1000 lines, got {len(lines)}"

    import json
    utt_ids = []
    for ln in lines:
        obj = json.loads(ln)
        utt_ids.append(obj["utterance_id"])

    assert len(set(utt_ids)) == 1000, "expected 1000 unique utterance IDs"

    # First sample must match the expected GRID utterance
    first = json.loads(lines[0])
    assert first["utterance_id"] == "T01_Ubbaf2n"
    assert first["frame_count"] == 75
    assert first["fps"] == 25.0
    assert first["transcript"] == "bin blue at f two now"


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
