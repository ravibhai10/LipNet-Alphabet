"""Optional isolated-environment checks; synthetic TEMP fixtures never enter corpus data."""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pytest
pytest.importorskip("sklearn", reason="Use .venv-word-recognition for the independent word workflow tests")
import torch

from pipeline.word_recognition import common, preprocess, train
from pipeline.word_recognition import download
from pipeline.word_recognition.data import WordDataset, augment
from pipeline.word_recognition.model import WordVideoNet
from pipeline.word_recognition.smoke import gradient_report
from pipeline.word_recognition.test import evaluate, export_matrix, metrics


@pytest.fixture
def config(tmp_path):
    value = common.load_config(common.ROOT / "pipeline/word_recognition/config.cpu.json")
    value["words"] = ["bin", "lay"]
    value["quotas"] = dict.fromkeys(common.SPLITS, 1)
    value["minimum_speakers"] = dict.fromkeys(common.SPLITS, 1)
    value["processed_root"] = str(tmp_path / "processed")
    value["training"].update(epochs=3, batch_size=2, patience=1, max_minutes=1)
    return value


def rows_for(config):
    rows = []
    for fold, split in enumerate(common.SPLITS):
        for index, word in enumerate(config["words"]):
            rows.append({"id": f"fixture-{split}-{word}", "word": word, "label": index, "speaker": f"s{fold+1}", "split": split,
                         "source_sha256": common.fingerprint([split, word]), "pixels_sha256": common.fingerprint([split, word, "pixels"]),
                         "start_frame": 2, "end_frame": 10, "data_source": "official_grid_video", "landmark_source": "mediapipe"})
    return rows


def make_dataset(config, run):
    import cv2
    common.run_contract(config, run)
    rows = rows_for(config)
    for index, row in enumerate(rows):
        clip = np.full((24, 64, 64), 30 + index * 30, dtype=np.uint8)
        row["pixels_sha256"] = hashlib.sha256(clip.tobytes()).hexdigest()
        row["frames_path"] = f"{row['speaker']}/{row['word']}/{row['id']}"
        directory = common.prepared_root(config) / row["frames_path"]
        directory.mkdir(parents=True)
        for i, frame in enumerate(clip):
            assert cv2.imwrite(str(directory / f"frame_{i:04d}.png"), frame)
    common.write_rows(run / "words.jsonl", rows)
    info = {"task": common.TASK, "config_sha256": common.fingerprint(config), "vocabulary": config["words"],
            "preprocessing": config["preprocessing"], "manifest_sha256": common.sha256(run / "words.jsonl"),
            "landmarker_sha256": common.sha256(common.ROOT / "frontend/public/models/face_landmarker.task"),
            "samples": len(rows), "protocol": "explicit temporary software fixture", "task_limits": "not a recognition benchmark"}
    common.atomic_json(run / "dataset.json", info)
    common.atomic_json(run / "smoke.json", {"manifest_sha256": info["manifest_sha256"], "config_sha256": common.fingerprint(config)})
    return rows


@pytest.mark.parametrize("change", ["letter", "quota", "lr", "geometry"])
def test_config_rejects_incompatible_task_or_unsafe_shape(tmp_path, config, change):
    if change == "letter": config["words"] = ["a", "b"]
    if change == "quota": config["quotas"]["test"] = 0
    if change == "lr": config["training"]["lr"] = float("inf")
    if change == "geometry": config["preprocessing"]["size"] = 112
    path = tmp_path / "config.json"
    path.write_text(__import__("json").dumps(config))
    with pytest.raises(ValueError):
        common.load_config(path)


@pytest.mark.parametrize("change", ["speaker", "source", "pixels", "bounds", "quota", "landmarks"])
def test_dataset_gate_enforces_independent_words(config, change):
    rows = rows_for(config)
    if change == "speaker": rows[2]["speaker"] = rows[0]["speaker"]
    if change == "source": rows[2]["source_sha256"] = rows[0]["source_sha256"]
    if change == "pixels": rows[2]["pixels_sha256"] = rows[0]["pixels_sha256"]
    if change == "bounds": rows[2]["end_frame"] = rows[2]["start_frame"]
    if change == "quota": rows.pop()
    if change == "landmarks": rows[0]["landmark_source"] = "synthetic"
    with pytest.raises(ValueError):
        common.validate_rows(rows, config)


def test_same_source_different_word_segments_allowed_only_within_fold(config):
    rows = rows_for(config)
    rows[1]["source_sha256"] = rows[0]["source_sha256"]
    rows[1]["start_frame"], rows[1]["end_frame"] = 11, 20
    assert common.validate_rows(rows, config)["counts"]["train"] == {"bin": 1, "lay": 1}


def test_model_real_shapes_and_nonzero_finite_gradient_matrix():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = WordVideoNet(4)
    logits, shapes = model(torch.rand(2, 1, 24, 64, 64), return_shapes=True)
    assert shapes["frame_features"] == [2, 24, 32]
    assert shapes["bidirectional_gru"] == [2, 24, 96]
    assert logits.shape == (2, 4)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert gradient_report(model)["total_l2_norm"] > 0
    with pytest.raises(ValueError):
        model(torch.rand(2, 24, 1, 64, 64))


def test_augmentation_preserves_dimensions_range_and_input():
    source = torch.rand(1, 24, 64, 64)
    original = source.clone()
    result = augment(source)
    assert result.shape == source.shape and result.min() >= 0 and result.max() <= 1
    assert torch.equal(source, original)


def test_canonical_loader_rejects_missing_or_changed_frames(tmp_path, config):
    run = tmp_path / "run"
    rows = make_dataset(config, run)
    _, loaded = common.load_dataset(run, config, verify_pixels=True)
    dataset = WordDataset(run, loaded[:2], config)
    assert dataset[0][0].shape == (1, 24, 64, 64)
    assert torch.equal(dataset[0][0], dataset[0][0])
    with pytest.raises(ValueError, match="training partition"):
        WordDataset(run, loaded[-2:], config, training=True)
    target = common.prepared_root(config) / rows[0]["frames_path"] / "frame_0000.png"
    import cv2
    cv2.imwrite(str(target), np.ones((64, 64), dtype=np.uint8))
    with pytest.raises(ValueError, match="pixels changed"):
        preprocess.read_frames(run, rows[0], config)
    target.unlink()
    with pytest.raises(ValueError, match="Missing"):
        preprocess.read_frames(run, rows[0], config)


def test_word_metrics_correct_wer_scope_and_confusion_html(tmp_path):
    result = metrics([0, 0, 1], [[0, 1], [1, 0], [1, 0]], ["bin", "lay"])
    assert result["word_accuracy"] == pytest.approx(2 / 3)
    assert result["wer"] == pytest.approx(1 / 3)
    assert result["per_word"]["bin"]["correct"] == 1
    assert result["per_word"]["bin"]["total"] == 2
    assert result["confusion_matrix"] == [[1, 1], [0, 1]]
    assert result["classification_report"]["bin"]["recall"] == .5
    result["protocol"] = "temporary fixture"
    export_matrix(tmp_path / "matrix.html", result)
    assert "Row percentages" in (tmp_path / "matrix.html").read_text()
    with pytest.raises(ValueError):
        metrics([0], [[0, 0]], ["bin", "lay"])


def test_training_early_stop_checkpoint_evaluation_and_resume(tmp_path, config, monkeypatch):
    """Temporary plumbing metrics are not dataset/model performance claims."""
    run = tmp_path / "run"
    make_dataset(config, run)
    monkeypatch.setattr(train, "evaluate_loader", lambda *args: (1., .5))
    result = train.train(config, run)
    assert result["epochs_completed"] == 2 and result["early_stopped"] is True
    artifact = torch.load(run / "best.pt", weights_only=True)
    assert artifact["epoch"] == 1 and artifact["vocabulary"] == config["words"]
    assert (run / "last.pt").exists()
    with pytest.raises(FileExistsError):
        train.train(config, run)
    assert train.train(config, run, resume=True)["epochs_completed"] == 2
    evaluation = evaluate(config, run)
    assert evaluation["samples"] == 2 and evaluation["split"] == "test"
    output = run / "evaluation" / evaluation["checkpoint_sha256"][:16]
    assert (output / "confusion_matrix.html").exists()
    assert (output / "predictions.jsonl").exists()
    assert evaluate(config, run)["confusion_matrix"] == evaluation["confusion_matrix"]
    artifact["vocabulary"] = list(reversed(config["words"]))
    with pytest.raises(ValueError, match="mismatch"):
        train.validate_checkpoint(artifact, config, common.load_json(run / "dataset.json"))


def test_run_contract_preserves_existing_unrelated_output(tmp_path, config):
    run = tmp_path / "run"
    run.mkdir()
    (run / "user-file").write_text("unchanged")
    with pytest.raises(FileExistsError):
        common.run_contract(config, run)
    assert (run / "user-file").read_text() == "unchanged"


def test_archive_verification_detects_changed_bytes_without_overwriting(tmp_path, config):
    corpus, inventory, run = tmp_path / "corpus", tmp_path / "inventory", tmp_path / "run"
    corpus.mkdir()
    inventory.mkdir()
    run.mkdir()
    config["corpus_root"], config["source_inventory"] = str(corpus), str(inventory)
    asset = corpus / "official-archive.zip"
    asset.write_bytes(b"temporary integrity fixture")
    common.atomic_json(corpus / "download_manifest.json", {"source": "https://spandh.dcs.shef.ac.uk/gridcorpus/", "assets": {
        "s1/video": {"path": asset.name, "status": "verified_and_extracted", "bytes": asset.stat().st_size, "sha256": common.sha256(asset)}}})
    common.atomic_json(inventory / "inventory.json", {})
    (inventory / "video_inventory.jsonl").write_text("{}\n")
    assert len(download.prepare_sources(config, run)["archives"]) == 1
    asset.write_bytes(b"changed user bytes")
    with pytest.raises(ValueError, match="fingerprint changed"):
        download.prepare_sources(config, run)
    assert asset.read_bytes() == b"changed user bytes"


def test_training_timeout_stops_before_held_out_evaluation(tmp_path, config, monkeypatch):
    from types import SimpleNamespace
    run = tmp_path / "run"
    make_dataset(config, run)
    ticks = iter([0., 0., 1000., 1000.])
    monkeypatch.setattr(train, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    with pytest.raises(TimeoutError, match="budget"):
        train.train(config, run)
    assert not (run / "best.pt").exists()
    assert not (run / "evaluation_summary.json").exists()
    assert common.load_json(run / "training_summary.json")["epochs_completed"] == 0


def test_interrupted_epoch_resume_matches_uninterrupted_optimizer_and_rng(tmp_path, config, monkeypatch):
    import shutil
    config["training"]["patience"] = 5
    baseline, interrupted = tmp_path / "baseline", tmp_path / "interrupted"
    make_dataset(config, baseline)
    common.run_contract(config, interrupted)
    for name in ("dataset.json", "words.jsonl", "smoke.json"):
        shutil.copyfile(baseline / name, interrupted / name)
    monkeypatch.setattr(train, "evaluate_loader", lambda *args: (1., .5))
    train.train(config, baseline)
    original_model = train.WordVideoNet
    class InterruptModel(original_model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = 0
        def forward(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("deliberate interrupted software fixture")
            return super().forward(*args, **kwargs)
    monkeypatch.setattr(train, "WordVideoNet", InterruptModel)
    with pytest.raises(RuntimeError, match="deliberate interrupted"):
        train.train(config, interrupted)
    assert torch.load(interrupted / "last.pt", weights_only=True)["epoch"] == 1
    monkeypatch.setattr(train, "WordVideoNet", original_model)
    train.train(config, interrupted, resume=True)
    expected = torch.load(baseline / "last.pt", weights_only=True)["model_state_dict"]
    actual = torch.load(interrupted / "last.pt", weights_only=True)["model_state_dict"]
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-6, atol=1e-7)


def test_quality_diagnostics_distinguish_finite_execution_from_word_learning():
    from pipeline.word_recognition.diagnostics import analyze
    result = metrics([0, 0, 1, 1], [[1, 0]] * 4, ["bin", "lay"])
    history = [{"gradient_norm_min": .3, "gradient_norm_max": 2., "train_accuracy": .5, "validation_accuracy": .5}]
    report = analyze(history, result)
    assert report["deployment_ready"] is False
    assert "single_class_collapse" in report["flags"]
    assert report["observations"]["zero_recall_words"] == ["bin"]


def test_loader_bounded_cache_reuses_verified_pixels_without_mutating(tmp_path, config, monkeypatch):
    run = tmp_path / "run"
    rows = make_dataset(config, run)
    dataset = WordDataset(run, rows, config)
    first = dataset[0][0]
    monkeypatch.setattr("pipeline.word_recognition.data.read_frames", lambda *a: (_ for _ in ()).throw(AssertionError("repeat read")))
    second = dataset[0][0]
    assert torch.equal(first, second)
    assert next(iter(dataset.cache.values())).dtype == torch.uint8
