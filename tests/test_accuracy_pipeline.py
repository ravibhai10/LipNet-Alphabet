"""Synthetic unit fixtures verify plumbing, never speech recognition accuracy."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
from torch import nn
from models.feature_classifier import TemporalClassificationHead
from pipeline.avhubert.classification import AVHubertClassifier, audit_records, sha256_file
from pipeline.avhubert.loader import AVHubertLoader, ModelUnavailableError
from scripts.avhubert_classifier import train
from src.video_training import augment_video, resample_temporal, warmup_cosine_scheduler
from tracking.mouth_canonicalization import mouth_transform


def records():
    return [dict(id=f"{split}-{label}", speaker_id=f"speaker-{split}", label=label,
                 split=split, data_source="real_camera", video_sha256=f"{index:064x}")
            for index, (split, label) in enumerate((s, c) for s in ("train", "valid", "test") for c in ("A", "B"))]


def test_temporal_interpolation_includes_intermediate_values():
    result = resample_temporal(np.array([[0.], [10.]], dtype=np.uint8), 5)
    np.testing.assert_allclose(result[:, 0], [0, 2.5, 5, 7.5, 10])
    assert result.dtype == np.float32
    with pytest.raises(ValueError):
        resample_temporal([], 29)


def test_clip_augmentation_preserves_temporal_coherence_and_input():
    clip = torch.linspace(0, 1, 80).reshape(1, 1, 8, 10).repeat(9, 1, 1, 1)
    before = clip.clone()
    torch.manual_seed(7)
    result = augment_video(clip, mask_probability=0)
    assert torch.equal(clip, before)
    assert result.shape == clip.shape
    assert torch.equal(result[0], result[-1])
    assert 0 <= result.min() <= result.max() <= 1
    # Replicate padding keeps a monotonic spatial ramp; wrapping would not.
    shifted = augment_video(clip, mask_probability=0, photometric_probability=0)
    assert torch.all(shifted[..., 1:] >= shifted[..., :-1])


def test_temporal_mask_is_a_short_contiguous_span():
    result = augment_video(torch.ones(29, 1, 8, 8), spatial_jitter=0,
                           mask_probability=1, photometric_probability=0)
    masked = torch.where(result.flatten(1).sum(1) == 0)[0]
    assert 2 <= len(masked) <= 3
    assert torch.all(masked[1:] - masked[:-1] == 1)


def test_warmup_then_cosine_reaches_minimum():
    parameter = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=.001)
    scheduler = warmup_cosine_scheduler(optimizer, epochs=12, warmup_epochs=5)
    rates = []
    for _ in range(12):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step(); scheduler.step()
    assert rates[:5] == pytest.approx([.0002, .0004, .0006, .0008, .001])
    assert all(left > right for left, right in zip(rates[4:], rates[5:]))
    assert rates[-1] == pytest.approx(1e-6)


def test_single_epoch_keeps_usable_learning_rate():
    optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=.001)
    warmup_cosine_scheduler(optimizer, epochs=1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(.001)


def test_alignment_removes_head_tilt_and_centers_mouth():
    points = np.zeros((468, 3))
    points[33, :2], points[263, :2] = [.25, .2], [.75, .4]
    points[61, :2], points[291, :2] = [.4, .66], [.6, .74]
    points[13, :2], points[14, :2] = [.5, .69], [.5, .71]
    transform = mouth_transform(points, (200, 200, 3))
    mouth = np.c_[points[[61, 291], :2] * 200, np.ones(2)] @ transform.T
    np.testing.assert_allclose(mouth[0, 1], mouth[1, 1], atol=1e-5)
    center = transform @ np.array([100, 140, 1])
    np.testing.assert_allclose(center, [48, 48], atol=1e-5)
    points[61, 0] = np.nan
    with pytest.raises(ValueError):
        mouth_transform(points, (200, 200))


def test_official_video_only_contract_and_normalization():
    class Encoder:
        def extract_finetune(self, *, source, padding_mask, mask):
            assert source["audio"] is None and padding_mask is None and mask is False
            assert tuple(source["video"].shape) == (1, 1, 29, 88, 88)
            expected = (128 / 255 - .421) / .165
            assert source["video"].mean().item() == pytest.approx(expected, abs=1e-5)
            return torch.ones(1, 29, 768), None
    loader = AVHubertLoader()
    loader.available = True
    loader._task = SimpleNamespace(cfg=SimpleNamespace(image_crop_size=88, image_mean=.421, image_std=.165))
    loader._models = [Encoder()]
    features = loader.encode_frames(np.full((8, 96, 96), 128, dtype=np.uint8), target_frames=29)
    assert features.shape == (29, 768) and np.isfinite(features).all()


def test_explicit_missing_encoder_and_untrained_head_fail_closed(tmp_path):
    with pytest.raises(ModelUnavailableError, match="Explicit"):
        AVHubertLoader(tmp_path / "missing.pt").load()
    with pytest.raises(ModelUnavailableError, match="trained classifier head"):
        AVHubertClassifier(tmp_path / "head.pt").load()


def test_pooling_ignores_padding_and_auxiliary_head_receives_gradients():
    head = TemporalClassificationHead(input_dim=4, num_classes=2, viseme_classes=3)
    values = torch.ones(2, 5, 4)
    values[0, 2:] = 1000
    np.testing.assert_allclose(head.pooled(values, [2, 5]).detach().numpy(), np.ones((2, 4)))
    output = head.forward_with_aux(values, [2, 5])
    criterion = nn.CrossEntropyLoss(label_smoothing=.1)
    loss = criterion(output["logits"], torch.tensor([0, 1])) + .2 * criterion(output["viseme_logits"], torch.tensor([1, 2]))
    loss.backward()
    assert head.classifier.weight.grad.abs().sum() > 0
    assert head.viseme_head.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("corruption, message", [
    ("speaker", "Speaker leakage"), ("hash", "Duplicate video"),
    ("provenance", "real_camera"), ("id", "Duplicate clip"), ("class", "missing from")])
def test_dataset_audit_rejects_leakage_and_fixtures(corruption, message):
    rows = records()
    if corruption == "speaker": rows[2]["speaker_id"] = rows[0]["speaker_id"]
    if corruption == "hash": rows[2]["video_sha256"] = rows[0]["video_sha256"]
    if corruption == "provenance": rows[0]["data_source"] = "synthetic_fixture"
    if corruption == "id": rows[1]["id"] = rows[0]["id"]
    if corruption == "class": rows[-1]["label"] = "C"
    with pytest.raises(ValueError, match=message):
        audit_records(rows)


def test_head_training_writes_loadable_best_checkpoint_without_encoder(tmp_path, monkeypatch):
    """Only a temporary plumbing fixture; no real-data result is produced."""
    rows = records()
    cache = tmp_path / "cache"; cache.mkdir()
    for index, row in enumerate(rows):
        row["feature"] = f"{index}.npy"
        np.save(cache / row["feature"], np.full((29, 4), index / 10, dtype=np.float32))
    (cache / "features.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    encoder_file = tmp_path / "encoder.pt"; encoder_file.write_bytes(b"unit-test-encoder-fixture")
    encoder = dict(input_dim=4, data_source="real_camera", encoder_sha256=sha256_file(encoder_file),
                   preprocessing={}, target_frames=29)
    (cache / "encoder.json").write_text(json.dumps(encoder))
    args = SimpleNamespace(features=cache, output=tmp_path / "output", epochs=2, batch_size=2,
                           viseme_classes=0, viseme_weight=.2, seed=42, hidden_dim=0, lr=.001, warmup_epochs=1)
    train(args)
    checkpoint = torch.load(args.output / "head.pt", weights_only=True)
    metadata = checkpoint["metadata"]
    model = TemporalClassificationHead(**metadata["head_config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    assert metadata["vocabulary"] == ["A", "B"]
    assert metadata["test_samples"] == metadata["validation_samples"] == 2
    summary = json.loads((args.output / "training_summary.json").read_text())
    assert len(summary["history"]) == 2 and np.isfinite(summary["test_loss"])
    report = json.loads((args.output / "classification_report.json").read_text())
    assert report["samples"] == 2 and set(report["per_class"]) == {"A", "B"}
    assert (args.output / "confusion_matrix.csv").is_file()
    class TestEncoder:
        d_out = 4
        def __init__(self, path): self.path = path
        def _resolve_ckpt(self): return self.path
        def load(self): return self
        def preprocessing(self): return {}
    monkeypatch.setattr("pipeline.avhubert.classification.AVHubertLoader", TestEncoder)
    classifier = AVHubertClassifier(args.output / "head.pt", encoder_file).load()
    prediction = classifier.predict_features(np.ones((29, 4), dtype=np.float32))
    assert prediction["label"] in ["A", "B"]
    assert sum(item["prob"] for item in prediction["probabilities"]) == pytest.approx(1)
    checkpoint["metadata"]["test_samples"] = 0
    torch.save(checkpoint, tmp_path / "unevaluated.pt")
    with pytest.raises(ModelUnavailableError, match="completed training"):
        AVHubertClassifier(tmp_path / "unevaluated.pt", encoder_file).load()
    with pytest.raises(FileExistsError):
        train(args)


def test_feature_shape_and_fingerprint_are_validated(tmp_path):
    from scripts.avhubert_classifier import CachedFeatures
    path = tmp_path / "features.npy"
    np.save(path, np.zeros((28, 4), dtype=np.float32))
    row = dict(feature=path.name, feature_sha256=sha256_file(path), label="A")
    metadata = dict(cache_format="avhubert-features-v1", target_frames=29, input_dim=4)
    with pytest.raises(ValueError, match="dimensions"):
        CachedFeatures(tmp_path, [row], ["A"], metadata)[0]
    np.save(path, np.zeros((29, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="fingerprint"):
        CachedFeatures(tmp_path, [row], ["A"], metadata)[0]


def test_evaluation_and_whitespace_cannot_bypass_audit():
    rows = records(); rows[2]["augmentation"] = True
    with pytest.raises(ValueError, match="training only"):
        audit_records(rows)
    rows = records(); rows[2]["speaker_id"] += " "
    with pytest.raises(ValueError, match="whitespace"):
        audit_records(rows)


def test_classification_metrics_include_failures_and_axis_order():
    from pipeline.eval.classification import classification_report
    report = classification_report([0, 0, 1, 1], [0, 1, 1, 1], ["A", "B"])
    assert report["confusion_matrix"] == [[1, 1], [0, 2]]
    assert report["accuracy"] == .75
    assert report["per_class"]["A"]["precision"] == 1
    assert report["per_class"]["A"]["recall"] == .5
    assert report["per_class"]["B"]["f1"] == pytest.approx(.8)
