"""Software fixtures only. These checks never establish real LRW recognition accuracy."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import tarfile
import threading
from urllib.request import build_opener

import cv2
import numpy as np
import pytest
import torch

from pipeline.lrw import whole_word as words
from scripts import acquire_lrw as acquire


@contextmanager
def server(body, *, html=False, invalid_range=False, changed_etag=False, ignore_range=False):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            offset = int(self.headers.get("Range", "bytes=0-").split("=")[1].rstrip("-"))
            resumed = offset > 0 and not ignore_range
            self.send_response(206 if resumed else 200)
            self.send_header("Content-Type", "text/html" if html else "application/octet-stream")
            self.send_header("Content-Length", str(len(body) - offset if resumed else len(body)))
            self.send_header("ETag", '"new"' if changed_etag else '"stable"')
            if resumed:
                self.send_header("Content-Range", f"bytes {offset + int(invalid_range)}-{len(body)-1}/{len(body)}")
            self.end_headers()
            self.wfile.write(body[offset:] if resumed else body)

        def log_message(self, *args):
            pass
    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}/archive"
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


def partial(target, url, body):
    target.with_name(target.name + ".part").write_bytes(body[:10])
    acquire.atomic_json(target.with_name(target.name + ".part.json"), {
        "source_id": acquire.hashlib.sha256(url.encode()).hexdigest(),
        "expected_bytes": len(body), "etag": '"stable"'})


@pytest.mark.parametrize("ignore", [False, True])
def test_download_resumes_or_restarts_without_appending(tmp_path, ignore):
    body = b"BZh" + bytes(range(100))
    target = tmp_path / "lrw.tar.bz2"
    with server(body, ignore_range=ignore) as url:
        partial(target, url, body)
        path = acquire.download(url, target, build_opener(), reserve_bytes=0, attempts=1)
    assert path.read_bytes() == body
    assert not target.exists()  # Promotion requires publisher MD5 in the CLI.


@pytest.mark.parametrize("failure", ["html", "range", "etag"])
def test_download_rejects_bad_responses_preserving_partial(tmp_path, failure):
    body = b"BZh" + b"x" * 100
    with server(body, html=failure == "html", invalid_range=failure == "range", changed_etag=failure == "etag") as url:
        target = tmp_path / "archive"
        partial(target, url, body)
        with pytest.raises(ValueError):
            acquire.download(url, target, build_opener(), reserve_bytes=0, attempts=1)
        assert target.with_name("archive.part").read_bytes() == body[:10]


@pytest.mark.parametrize("url", ["http://thor.robots.ox.ac.uk/archive", "https://random.example/archive", "https://secret:password@thor.robots.ox.ac.uk/archive"])
def test_authorized_source_gate(url):
    with pytest.raises(ValueError):
        acquire.authorized_opener(url)


def archive(path, name="word/train/clip.mp4", contents=b"video", link=False):
    with tarfile.open(path, "w:bz2") as stream:
        item = tarfile.TarInfo(name)
        item.size = len(contents)
        if link:
            item.type = tarfile.SYMTYPE
            item.linkname = "outside"
        stream.addfile(item, io.BytesIO(contents) if not link else None)


@pytest.mark.parametrize("name", ["../outside", "C:/outside", "word:stream", "word\\outside", "/outside"])
def test_extract_rejects_traversal_and_windows_paths(tmp_path, name):
    source = tmp_path / "archive.bz2"
    archive(source, name)
    with pytest.raises(ValueError):
        acquire.extract_archive(source, tmp_path / "output", reserve_bytes=0)


def test_extract_preserves_matching_files_and_refuses_different_or_link(tmp_path):
    source, output = tmp_path / "archive.bz2", tmp_path / "output"
    archive(source)
    assert acquire.extract_archive(source, output, reserve_bytes=0)["written_files"] == 1
    assert acquire.extract_archive(source, output, reserve_bytes=0)["reused_verified_files"] == 1
    target = output / "word/train/clip.mp4"
    target.write_bytes(b"original user file")
    with pytest.raises(ValueError, match="preserved"):
        acquire.extract_archive(source, output, reserve_bytes=0)
    assert target.read_bytes() == b"original user file"
    archive(source, link=True)
    with pytest.raises(ValueError, match="links"):
        acquire.extract_archive(source, output, reserve_bytes=0)


def metadata(path, reference="123"):
    path.write_text(f"Disk reference: {reference}\nChannel: BBC One HD\nProgram start: 2015-11-26 13:00:00 +0000\nClip start: 1428.12 seconds\nDuration: 0.45 seconds\n")


def make_video(path, count=29, width=128):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25., (width, 128))
    assert writer.isOpened()
    for i in range(count):
        writer.write(np.full((128, width, 3), i, dtype=np.uint8))
    writer.release()


def test_official_metadata_parser_and_invalid_duration(tmp_path):
    path = tmp_path / "clip.txt"
    metadata(path)
    assert words.read_metadata(path)["word_duration"] == .45
    path.write_text(path.read_text().replace("0.45", "2.0"))
    with pytest.raises(ValueError, match="duration"):
        words.read_metadata(path)
    path.write_text("synthetic label ABOUT")
    with pytest.raises(ValueError, match="annotation"):
        words.read_metadata(path)


@pytest.mark.parametrize("count,width", [(28, 128), (30, 128), (29, 46)])
def test_decode_rejects_truncated_long_and_fixture_roi(tmp_path, count, width):
    path = tmp_path / "clip.mp4"
    make_video(path, count, width)
    with pytest.raises(ValueError):
        words.decode_check(path)


def test_real_decoder_and_shared_preprocessing_shape(tmp_path, monkeypatch):
    path = tmp_path / "clip.mp4"
    make_video(path)
    assert words.decode_check(path)["frames"] == 29
    # Landmark geometry mocked ONLY here; no claim generated video contains a face.
    monkeypatch.setattr("tracking.mouth_canonicalization.read_aligned_mouth_video",
                        lambda *a, **k: (np.full((29, 96, 96), 128, dtype=np.uint8), 25.))
    tensor = words.preprocess_video(path)
    assert tensor.dtype == torch.uint8 and tensor.shape == (29, 1, 140, 46)
    assert tensor.float().div(255).mean() == pytest.approx(128 / 255)


def test_ingest_requires_approval_and_official_layout(tmp_path):
    with pytest.raises(ValueError, match="authorized"):
        words.ingest(tmp_path, tmp_path / "out")
    path = tmp_path / "root/train/ABOUT/about_00001.mp4"
    make_video(path)
    with pytest.raises(ValueError, match="Unknown/mismatched"):
        words.ingest(tmp_path / "root", tmp_path / "out", agreement_confirmed=True)


def test_ingest_rejects_missing_annotation_and_duplicate_content(tmp_path):
    root = tmp_path / "root"
    one, two = root / "ABOUT/train/ABOUT_00001.mp4", root / "ABOUT/val/ABOUT_00002.mp4"
    make_video(one)
    with pytest.raises(FileNotFoundError, match="annotation"):
        words.ingest(root, tmp_path / "out", agreement_confirmed=True)
    metadata(one.with_suffix(".txt"))
    two.parent.mkdir(parents=True)
    two.write_bytes(one.read_bytes())
    metadata(two.with_suffix(".txt"), "456")
    with pytest.raises(ValueError, match="Duplicate"):
        words.ingest(root, tmp_path / "out", agreement_confirmed=True)
    assert not (tmp_path / "out/dataset.json").exists()


def test_distribution_requires_every_class_and_actual_official_counts():
    rows = [{"word": "ABOUT", "split": split} for split in words.SPLITS]
    with pytest.raises(ValueError, match="classes"):
        words.validate_distribution(rows)
    with pytest.raises(ValueError, match="distribution"):
        words.validate_distribution(rows, ["ABOUT"])
    rows = [{"word": "ABOUT", "split": split} for split, n in (("train", 800), ("valid", 50), ("test", 50)) for _ in range(n)]
    assert words.validate_distribution(rows, ["ABOUT"])["test"]["ABOUT"] == 50


def test_word_topk_per_class_denominators_and_confusion(tmp_path):
    labels = ["A", "B", "C", "D", "E", "F"]
    report = words.word_metrics([0, 1, 1], [[0, 2, 3, 4, 5], [0, 1, 2, 3, 4], [5, 4, 3, 2, 0]], labels)
    assert report["top1_accuracy"] == pytest.approx(1 / 3)
    assert report["top5_accuracy"] == pytest.approx(2 / 3)
    assert report["per_word"]["B"] == {"correct": 0, "total": 2, "top5_correct": 1, "accuracy": 0.}
    assert report["per_word"]["F"]["accuracy"] is None
    assert report["confusion_matrix"][1][5] == 1
    words.write_metrics(tmp_path, report, [{"id": "execution-fixture"}])
    assert len((tmp_path / "per_word.csv").read_text().splitlines()) == 7
    assert (tmp_path / "confusion_matrix.csv").exists()
    with pytest.raises(ValueError):
        words.word_metrics([0], [[0, 0, 1, 2, 3]], labels)


def fixture_artifact(state):
    return {"schema_version": 1, "task": words.TASK, "architecture": words.ARCHITECTURE,
            "vocabulary": words.VOCABULARY, "vocabulary_sha256": words.signature(words.VOCABULARY),
            "preprocessing": words.PREPROCESSING, "protocol": words.PROTOCOL,
            "manifest_sha256": "0" * 64, "cache_contract": words.cache_contract("0" * 64),
            "epoch": 1, "validation_loss": 6., "model_state_dict": state}


def test_legacy_word_weights_rejected_and_mapping_contract_enforced(tmp_path):
    model = words.AlphabetCNNBiLSTM(num_classes=500)
    path = tmp_path / "fixture.pt"
    torch.save({"model_state_dict": model.state_dict(), "num_classes": 500}, path)
    with pytest.raises(ValueError, match="legacy"):
        words.load_word_model(path)
    artifact = fixture_artifact(model.state_dict())
    torch.save(artifact, path)
    loaded, _ = words.load_word_model(path)
    assert loaded.fc2.out_features == 500
    artifact["vocabulary"] = list(reversed(words.VOCABULARY))
    torch.save(artifact, path)
    with pytest.raises(ValueError):
        words.load_word_model(path)


def test_cpu_training_selection_test_report_and_prediction_roundtrip(tmp_path, monkeypatch):
    """Bypasses acquisition with explicit TEMPORARY fixtures to exercise optimizer/artifact plumbing."""
    torch.set_num_threads(2)
    cache = tmp_path / "cache"
    cache.mkdir()
    rows = []
    for split in words.SPLITS:
        for label in (0, 1):
            key = f"fixture-{split}-{label}"
            torch.save(torch.full((29, 1, 140, 46), 50 + label * 100, dtype=torch.uint8), cache / (key + ".pt"))
            rows.append({"id": key, "word": words.VOCABULARY[label], "label": label, "split": split, "source_sha256": key})
    monkeypatch.setattr(words, "verified_cache", lambda *a: ({"manifest_sha256": "0" * 64}, rows, words.cache_contract("0" * 64)))
    output = tmp_path / "run"
    result = words.train("unused", cache, output, epochs=1, batch_size=2)
    assert result["samples"] == 2  # Fixture count; never a real recognition result.
    assert (output / "held_out_test/per_word.csv").exists()
    model, artifact = words.load_word_model(output / "word_model.pt")
    assert artifact["epoch"] == 1 and model.fc2.out_features == 500
    monkeypatch.setattr(words, "preprocess_video", lambda *a: torch.zeros(29, 1, 140, 46, dtype=torch.uint8))
    result = words.predict(output / "word_model.pt", "unused")
    assert len(result["top5"]) == 5 and result["confidence_calibrated"] is False
    # An audited artifact from different partitions cannot initialize a new test protocol.
    artifact["manifest_sha256"] = "1" * 64
    artifact["cache_contract"] = words.cache_contract("1" * 64)
    torch.save(artifact, tmp_path / "other-partitions.pt")
    with pytest.raises(ValueError, match="frozen splits"):
        words.train("unused", cache, tmp_path / "fine-tune", epochs=1, init_word=tmp_path / "other-partitions.pt")
    with pytest.raises(FileExistsError):
        words.train("unused", cache, output, epochs=1)


def test_extracted_member_partial_resumes_only_after_prefix_verification(tmp_path):
    source, output = tmp_path / "archive.bz2", tmp_path / "output"
    archive(source, contents=b"abcdefghij")
    target = output / "word/train/clip.mp4"
    target.parent.mkdir(parents=True)
    part = target.with_name(target.name + ".lrw-part")
    part.write_bytes(b"abc")
    acquire.extract_archive(source, output, reserve_bytes=0)
    assert target.read_bytes() == b"abcdefghij"
    target.unlink()
    part.write_bytes(b"bad")
    with pytest.raises(ValueError, match="prefix"):
        acquire.extract_archive(source, output, reserve_bytes=0)
    assert part.read_bytes() == b"bad"


def test_cache_resume_fingerprints_sources_and_training_rejects_corruption(tmp_path, monkeypatch):
    source, annotation = tmp_path / "source.mp4", tmp_path / "source.txt"
    source.write_bytes(b"explicit execution fixture")
    metadata(annotation)
    row = {"id": "fixture", "video": str(source), "annotation": str(annotation),
           "source_sha256": words.digest(source), "annotation_sha256": words.digest(annotation)}
    monkeypatch.setattr(words, "load_dataset", lambda *a: ({"manifest_sha256": "0" * 64}, [row]))
    calls = []
    def preprocess(*args):
        calls.append(1)
        return torch.zeros(29, 1, 140, 46, dtype=torch.uint8)
    monkeypatch.setattr(words, "preprocess_video", preprocess)
    output = tmp_path / "cache"
    assert words.cache_clips("unused", output, limit=1, reserve_gib=0)["new_clips"] == 1
    assert words.cache_clips("unused", output, reserve_gib=0)["new_clips"] == 0
    assert len(calls) == 1
    words.verified_cache("unused", output)
    tensor = output / (row["source_sha256"] + ".pt")
    tensor.write_bytes(b"corrupt cache")
    with pytest.raises(ValueError, match="fingerprint"):
        words.verified_cache("unused", output)
    with pytest.raises(ValueError, match="fingerprint"):
        words.cache_clips("unused", output, reserve_gib=0)


def test_legacy_preprocessing_fails_decode_instead_of_substituting_zeros(tmp_path):
    from scripts.preprocess_lrw_dataset import preprocess_video_clip, run_preprocessing
    invalid = tmp_path / "invalid.mp4"
    invalid.write_bytes(b"not video")
    with pytest.raises(ValueError, match="no substitute"):
        preprocess_video_clip(invalid)
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "user-file").write_text("preserve")
    with pytest.raises(FileExistsError):
        run_preprocessing(tmp_path, existing, tmp_path / "manifest.jsonl")


def test_legacy_real_training_entrypoint_cannot_consume_letter_or_fixture_manifests():
    from scripts.train_lrw_500words import run_training
    with pytest.raises(ValueError, match="whole-word"):
        run_training(data_manifest=Path("untrusted.jsonl"))
