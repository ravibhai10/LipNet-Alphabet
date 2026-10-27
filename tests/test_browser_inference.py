"""CPU browser-inference contract tests. Synthetic crops test plumbing, not speech accuracy."""
import base64
import numpy as np
import cv2
import pytest
from app.services import browser_inference as module

def jpeg_frame():
    ok, encoded = cv2.imencode(".jpg", np.full((64, 64), 96, dtype=np.uint8))
    assert ok
    return base64.b64encode(encoded).decode("ascii")

def test_temporal_input_is_resampled_to_checkpoint_shape():
    result = module.BrowserInference.decode_frames([jpeg_frame()] * 15)
    assert tuple(result.shape) == (1, 30, 1, 140, 46)
    assert 0 <= result.min() <= result.max() <= 1

@pytest.mark.parametrize("frames", [[], ["x"] * 7, ["x"] * 77, ["invalid-base64"] * 15])
def test_invalid_temporal_inputs_are_rejected(frames):
    with pytest.raises(ValueError):
        module.BrowserInference.decode_frames(frames)

def test_non_jpeg_payload_is_rejected():
    frame = base64.b64encode(b"not an image").decode("ascii")
    with pytest.raises(ValueError, match="JPEG"):
        module.BrowserInference.decode_frames([frame] * 15)

def test_missing_checkpoint_reports_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "CHECKPOINT_PATH", tmp_path / "missing.pt")
    result = module.BrowserInference().health()
    assert result["inference_ready"] is False
    assert result["model_loaded"] is False
    assert "missing" in result["inference_reason"].lower()

@pytest.mark.skipif(not module.CHECKPOINT_PATH.exists(), reason="Local trained alphabet checkpoint is not included")
def test_real_cpu_checkpoint_returns_valid_probabilities():
    service = module.BrowserInference()
    assert service.health()["inference_ready"]
    result = service.predict([jpeg_frame()] * 15)
    assert result["letter"] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    assert 0 <= result["confidence"] <= 1
    assert len(result["top3"]) == 3
    assert result["top3"][0]["letter"] == result["letter"]
    values = [item["prob"] for item in result["top3"]]
    assert all(np.isfinite(value) and 0 <= value <= 1 for value in values)
    assert values == sorted(values, reverse=True)
    assert sum(values) <= 1.00001
    assert len(result["probabilities"]) == 26
    assert sum(item["prob"] for item in result["probabilities"]) == pytest.approx(1, abs=1e-5)
    assert result["temporal"]["model_frames"] == 30
    assert result["uncertainty"]["state"] == "UNCALIBRATED"
    assert result["latency_ms"] >= 0
    service._predict_lock.acquire()
    try:
        with pytest.raises(BlockingIOError):
            service.predict([jpeg_frame()] * 15)
    finally:
        service._predict_lock.release()


def test_minimum_live_window_is_supported():
    result = module.BrowserInference.decode_frames([jpeg_frame()] * 8)
    assert tuple(result.shape) == (1, 30, 1, 140, 46)

def test_http_endpoint_contract():
    from app.main import PredictRequest, predict
    from fastapi import HTTPException
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        PredictRequest(frames=[])
    with pytest.raises(HTTPException) as caught:
        predict(PredictRequest(frames=["bad"] * 15))
    assert caught.value.status_code == 422

