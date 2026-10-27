"""Only temporary annotation fixtures; no camera or model is used."""
import json
import pytest
from scripts.prepare_alphabet_annotations import prepare
from pipeline.avhubert.classification import load_annotations


def test_preparation_requires_explicit_disjoint_speakers(tmp_path):
    with pytest.raises(ValueError, match="more than once"):
        prepare(tmp_path, {"train": ["s1"], "valid": ["s1"], "test": ["s3"]}, tmp_path / "clips.jsonl")
    assert not (tmp_path / "clips.jsonl").exists()


def test_preparation_preserves_videos_and_creates_valid_relative_manifest(tmp_path):
    assignments = {"train": ["s1"], "valid": ["s2"], "test": ["s3"]}
    for speaker in ("s1", "s2", "s3"):
        for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            folder = tmp_path / "recordings" / label / speaker / "session" / "recording001"
            folder.mkdir(parents=True)
            (folder / "video.mp4").write_bytes(f"annotation-unit-fixture-{speaker}-{label}".encode())
            (folder / "metadata.json").write_text(json.dumps({"speaker_id": speaker, "letter": label, "data_source": "real_camera"}))
    output = tmp_path / "annotations/clips.jsonl"
    report = prepare(tmp_path, assignments, output)
    assert report["clips"] == 78 and report["speakers"] == 3
    assert len(load_annotations(output)) == 78
    assert len(list((tmp_path / "recordings").rglob("video.mp4"))) == 78
    with pytest.raises(FileExistsError):
        prepare(tmp_path, assignments, output)
