"""Small synthetic integrity fixtures; no figures here measure speech accuracy.

No downloaded model is loaded, and no acquisition or encoder extraction runs.
"""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from pipeline.word_recognition import expanded as experiment
from pipeline.word_recognition import expanded_data as data
from pipeline.word_recognition.common import sha256, write_rows


def source_rows(word="hello", count=20):
    return [
        {
            "id": f"allenye66:{word}_{i}",
            "dataset": "allenye66",
            "word_label": word,
            "pixels_sha256": hashlib.sha256(f"{word}:{i}".encode()).hexdigest(),
            "speaker_id": None,
            "session_id": None,
        }
        for i in range(count)
    ]


def test_twenty_word_union_preserves_every_original_and_publisher_label():
    assert len(data.BASE_WORDS) == 8
    assert len(data.UPSTREAM_WORDS) == 13
    assert set(data.BASE_WORDS) & set(data.UPSTREAM_WORDS) == {"hello"}
    assert len(data.WORDS) == 20
    assert data.WORDS == tuple(sorted(set(data.BASE_WORDS) | set(data.UPSTREAM_WORDS)))
    assert experiment.FORMAT != "lipnet_avhubert_word_linear_v1"


def test_source_json_is_integer_geometry_not_executable_python(tmp_path):
    source = tmp_path / "data.txt"
    value = np.zeros((22, 80, 112, 3), dtype=np.uint8)
    value[0, 0, 0] = [1, 17, 255]
    source.write_text(json.dumps(value.tolist()), encoding="utf-8")
    actual = data.parse_sequence(source)
    assert actual.dtype == np.uint8 and np.array_equal(actual, value)
    marker = tmp_path / "executed.txt"
    source.write_text(f"__import__('pathlib').Path({str(marker)!r}).write_text('bad')")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        data.parse_sequence(source)
    assert not marker.exists()


@pytest.mark.parametrize("bad_value", [-1, 256, 0.5, True])
def test_source_parser_rejects_wrapped_or_noninteger_pixels(tmp_path, bad_value):
    source = tmp_path / "data.txt"
    # A uniform value makes dtype inference unambiguous, including boolean input.
    raw = np.full((22, 80, 112, 3), bad_value).tolist()
    source.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="integer BGR"):
        data.parse_sequence(source)


def test_adapter_keeps_bgr_convention_and_real_temporal_endpoints():
    source = np.zeros((22, 80, 112, 3), dtype=np.uint8)
    source[0, :, :, 0] = 255  # OpenCV BGR blue: grayscale 29.
    source[-1, :, :, 2] = 255  # OpenCV BGR red: grayscale 76.
    result = data.adapt_sequence(source)
    assert result.shape == (29, 96, 96) and result.dtype == np.uint8
    assert np.all(result[0] == 29) and np.all(result[-1] == 76)
    assert "unavailable" in data.ADAPTER["alignment"]


@pytest.mark.parametrize("shape,dtype", [((22, 112, 80, 3), np.uint8), ((22, 80, 112, 3), np.float32)])
def test_adapter_rejects_implicit_transposition_and_float_conversion(shape, dtype):
    with pytest.raises(ValueError, match="geometry"):
        data.adapt_sequence(np.zeros(shape, dtype=dtype))


def test_blocked_recording_split_is_numeric_and_input_order_independent():
    rows = source_rows()
    chosen, duplicates = experiment.split_groups(rows[::-1])
    assert not duplicates
    assert [r["id"] for r in chosen] == [r["id"] for r in rows]
    assert [r["split"] for r in chosen] == ["train"] * 14 + ["valid"] * 3 + ["test"] * 3
    assert all(r["speaker_id"] is None for r in chosen)


def test_exact_duplicate_alias_never_creates_another_partition_example():
    rows = source_rows()
    rows.append({**rows[0], "id": "allenye66:hello_99"})
    rows.append({**rows[0], "id": "allenye66:test_1", "word_label": None})
    chosen, duplicates = experiment.split_groups(rows)
    assert len(chosen) == 20 and len(duplicates) == 1
    assert duplicates[0]["aliases"] == ["allenye66:hello_99"]
    assert len({r["pixels_sha256"] for r in chosen}) == len(chosen)
    assert all(r["word_label"] == "hello" for r in chosen)


def test_conflicting_duplicate_labels_and_small_classes_fail_before_training():
    rows = source_rows()
    with pytest.raises(ValueError, match="Conflicting labels"):
        experiment.split_groups(rows + [{**rows[0], "id": "allenye66:bye_99", "word_label": "bye"}])
    with pytest.raises(ValueError, match="Insufficient"):
        experiment.split_groups(rows[:19])


def test_duplicate_source_ids_cannot_represent_two_different_utterances():
    rows = source_rows()
    rows[-1]["id"] = rows[0]["id"]
    with pytest.raises(ValueError, match="(?i)(duplicate|repeated|unique).*id"):
        experiment.split_groups(rows)


def feature_fixture(tmp_path, monkeypatch):
    path = tmp_path / "features.npy"
    np.save(path, np.ones((3, 29, 768), dtype=np.float32), allow_pickle=False)
    original = {
        **source_rows()[0],
        "split": "test",
        "usable": True,
        "role": "known_word",
        "source_sha256": "a" * 64,
        "array_sha256": "b" * 64,
        "input_kind": "publisher_bgr_mouth_array",
        "start_frame": 0,
        "end_frame": 22,
    }
    derived = {**original, "feature": str(path), "feature_sha256": sha256(path)}
    monkeypatch.setattr(experiment, "contract_for", lambda run: ({}, [deepcopy(original)]))
    return original, derived


@pytest.mark.parametrize(
    "field,value",
    [
        ("split", "train"),
        ("word_label", "bye"),
        ("dataset", "miracl"),
        ("usable", False),
        ("source_sha256", "c" * 64),
        ("speaker_id", "invented-speaker"),
    ],
)
def test_derived_features_cannot_change_frozen_sample_identity(tmp_path, monkeypatch, field, value):
    _, row = feature_fixture(tmp_path, monkeypatch)
    row[field] = value
    write_rows(tmp_path / "features.jsonl", [row])
    with pytest.raises(ValueError):
        experiment.load_features(tmp_path)


def test_duplicate_feature_rows_cannot_inflate_evaluation_denominator(tmp_path, monkeypatch):
    _, row = feature_fixture(tmp_path, monkeypatch)
    write_rows(tmp_path / "features.jsonl", [row, row])
    with pytest.raises(ValueError):
        experiment.load_features(tmp_path)


def test_derived_manifest_validator_allows_added_artifact_fields_only():
    original = {**source_rows()[0], "split": "valid", "usable": True}
    derived = {**original, "feature": "features/example.npy", "feature_sha256": "a" * 64}
    experiment.validate_derived([original], [derived])
    with pytest.raises(ValueError):
        experiment.validate_derived([original], [{**derived, "split": "train"}])
    with pytest.raises(ValueError):
        experiment.validate_derived([original], [derived, derived])
    with pytest.raises(ValueError):
        experiment.validate_derived([original], [])


def test_unchanged_features_load_and_mutated_bytes_are_rejected(tmp_path, monkeypatch):
    _, row = feature_fixture(tmp_path, monkeypatch)
    write_rows(tmp_path / "features.jsonl", [row])
    _, rows, features = experiment.load_features(tmp_path)
    assert len(rows) == 1 and features[row["id"]].shape == (3, 29, 768)
    np.save(row["feature"], np.zeros((3, 29, 768), dtype=np.float32), allow_pickle=False)
    with pytest.raises(ValueError, match="bytes changed"):
        experiment.load_features(tmp_path)


def test_feature_key_binds_encoder_preprocessing_as_well_as_source_adapter(monkeypatch):
    original = experiment.feature_key("roi-sha", "encoder-sha")
    monkeypatch.setattr(experiment, "PREPROCESS", {**experiment.PREPROCESS, "mean": 0.5})
    assert experiment.feature_key("roi-sha", "encoder-sha") != original


def test_completed_array_without_receipt_can_resume_after_interruption(tmp_path):
    directory = tmp_path / "original" / "hello_0"
    directory.mkdir(parents=True)
    source = directory / "data.txt"
    value = np.zeros((22, 80, 112, 3), dtype=np.uint8)
    source.write_text(json.dumps(value.tolist()), encoding="utf-8")
    np.save(directory / "frames.npy", value, allow_pickle=False)
    item = {"path": "collected_data/hello_0/data.txt", "size": source.stat().st_size, "sha": data.blob_sha(source)}
    # The publisher bytes already exist, so this must not need the network.
    receipt = data.download_one(item, tmp_path)
    assert receipt["array_sha256"] == sha256(directory / "frames.npy")
    assert receipt["word_label"] == "hello" and (directory / "receipt.json").exists()


def test_cached_receipt_cannot_relabel_identical_verified_publisher_bytes(tmp_path):
    directory = tmp_path / "original" / "hello_0"
    directory.mkdir(parents=True)
    source = directory / "data.txt"
    value = np.zeros((22, 80, 112, 3), dtype=np.uint8)
    source.write_text(json.dumps(value.tolist()), encoding="utf-8")
    item = {"path": "collected_data/hello_0/data.txt", "size": source.stat().st_size, "sha": data.blob_sha(source)}
    receipt = data.download_one(item, tmp_path)
    receipt["word_label"] = "bye"
    (directory / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError):
        data.download_one(item, tmp_path)


@pytest.mark.parametrize("protected", ["head.pt", "selection.json", "test_access.json", "evaluation.json"])
def test_completed_or_test_opened_runs_cannot_be_retrained(tmp_path, protected):
    (tmp_path / protected).write_text("protected")
    with pytest.raises(FileExistsError, match="protected"):
        experiment.train(tmp_path)


def classifier_fixture(tmp_path):
    import torch
    from pipeline.word_recognition import pretrained_inference as runtime

    encoder = tmp_path / "encoder.pt"
    encoder.write_bytes(b"fixture fingerprint only; not an encoder")
    landmark = tmp_path / "frontend/public/models/face_landmarker.task"
    landmark.parent.mkdir(parents=True)
    landmark.write_bytes(b"fixture landmark fingerprint only")
    head = {
        "format": experiment.FORMAT,
        "contract": {
            "preprocessing": runtime.PREPROCESS,
            "vocabulary": list(data.WORDS),
            "source_adapters": {"allenye66": data.ADAPTER, "miracl": runtime.PREPROCESS},
            "upstream_commit": data.COMMIT,
            "encoder_sha256": sha256(encoder),
            "landmarker_sha256": sha256(landmark),
        },
        "vocabulary": list(data.WORDS),
        "audio_inference": False,
        "feature_layer": 9,
        "pooling": "mean",
        "mean": torch.zeros(768),
        "scale": torch.ones(768),
        "weight": torch.zeros((20, 768)),
        "bias": torch.zeros(20),
    }
    path = tmp_path / "head.pt"
    torch.save(head, path)
    (tmp_path / "rejection_policy.json").write_text(json.dumps({"head_sha256": sha256(path), "threshold": 0.9, "margin": 0.2}))

    class Encoder:
        calls = 0

        def preprocessing(self):
            return {"image_crop_size": 88, "image_mean": 0.421, "image_std": 0.165}

        def encode_frames(self, images, **kwargs):
            assert kwargs == {"source_fps": None, "target_frames": 29, "output_layer": 9}
            self.calls += 1
            return np.zeros((29, 768), dtype=np.float32)

    return runtime, path, encoder, Encoder()


def test_expanded_head_cannot_enter_original_eight_word_loader(tmp_path, monkeypatch):
    runtime, path, encoder_path, fake = classifier_fixture(tmp_path)
    monkeypatch.setattr(runtime, "ENCODER", encoder_path)
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="contract mismatch"):
        runtime.AVHubertWordClassifier(path, encoder=fake).load()
    assert fake.calls == 0


def test_twenty_word_loader_reuses_encoder_and_preserves_abstention(tmp_path, monkeypatch):
    runtime, path, encoder_path, fake = classifier_fixture(tmp_path)
    monkeypatch.setattr(runtime, "ENCODER", encoder_path)
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    model = runtime.AVHubertWordClassifier(path, expected_vocabulary_size=20, encoder=fake).load()
    assert fake.calls == 0  # Sharing must not perform a second warm-up or allocate weights.
    prediction = model.predict_rois(np.zeros((8, 96, 96), dtype=np.uint8))
    assert fake.calls == 1 and len(prediction["probabilities"]) == 20
    assert prediction["vocabulary"] == list(data.WORDS)
    assert prediction["word"] is None and prediction["uncertain"] is True
    assert prediction["candidate"] in data.WORDS
    assert sum(prediction["probabilities"]) == pytest.approx(1)
    with pytest.raises(ValueError, match="96px"):
        model.predict_rois(np.zeros((22, 80, 112), dtype=np.uint8))
    assert fake.calls == 1

