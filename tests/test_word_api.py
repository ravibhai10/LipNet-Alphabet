"""
Unit and integration tests for v1 Neural Word Inference REST API.
Verifies /api/v1/word/health and /api/v1/predict/word.
"""
from __future__ import annotations

import base64
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.services.neural_word_service import neural_word_service


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def sample_frames():
    """Generate 29 mock base64 JPEG mouth crops."""
    buf = io.BytesIO()
    img = Image.new("L", (46, 140), 128)
    img.save(buf, format="JPEG")
    b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return [b64] * 29


def test_word_v1_health(client):
    res = client.get("/api/v1/word/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert data["vocabulary_size"] == 500
    assert "engine_mode" in data


def test_word_v1_predict(client, sample_frames):
    res = client.post("/api/v1/predict/word", json={
        "frames": sample_frames,
        "return_attention": True,
    })
    assert res.status_code == 200
    data = res.json()
    assert data["task"] == "isolated_word"
    assert isinstance(data["predicted_word"], str)
    assert len(data["predicted_word"]) > 0
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["top5"], list)
    assert len(data["top5"]) == 5
    assert len(data["temporal_attention"]) == 29
    assert isinstance(data["inference_ms"], float)
    assert data["inference_ms"] >= 0.0


def test_models_registry_includes_neural_v1(client):
    res = client.get("/api/models")
    assert res.status_code == 200
    tracks = res.json().get("tracks", {})
    assert "neural_v1" in tracks
    assert tracks["neural_v1"]["endpoint"] == "/api/v1/predict/word"
    assert tracks["neural_v1"]["vocabulary_size"] == 500
