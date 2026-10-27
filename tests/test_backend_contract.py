"""Real ASGI routes; synthetic JPEG inputs check contracts, not accuracy."""
import base64
from types import SimpleNamespace
from fastapi.testclient import TestClient
from app import main


def test_health_distinguishes_service_and_selected_model(monkeypatch):
    monkeypatch.setattr(main.browser_inference, "health", lambda: {
        "model_loaded": False, "inference_ready": False, "inference_reason": "Head required",
        "model_kind": "test", "device": "cpu", "streaming_ready": False})
    monkeypatch.setattr(main, "get_inference_service", lambda: SimpleNamespace(has_loaded_model=lambda: True))
    with TestClient(main.app) as client:
        result = client.get("/health").json()
        assert result["status"] == "ok" and result["backend"] == "ready"
        assert result["model_loaded"] is False and result["research_model_loaded"] is True
        assert result["capabilities"]["sentence_recognition"]["ready"] is False
        assert client.get("/api/model/availability").json() == result["capabilities"]


def test_cors_and_invalid_http_contract():
    with TestClient(main.app) as client:
        response = client.options("/api/predict", headers={"Origin": "http://127.0.0.1:5173", "Access-Control-Request-Method": "POST"})
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
        response = client.options("/api/predict", headers={"Origin": "https://unconfigured.invalid", "Access-Control-Request-Method": "POST"})
        assert response.status_code == 400
        response = client.post("/api/predict", json={"frames": []})
        assert response.status_code == 422 and isinstance(response.json()["detail"], list)
        assert client.get("/api/does-not-exist").status_code == 404


def test_prediction_unavailable_does_not_expose_paths(monkeypatch):
    def unavailable(_frames):
        raise RuntimeError("C:\\private-user\\secret-model.pt failed")
    monkeypatch.setattr(main.browser_inference, "predict", unavailable)
    with TestClient(main.app) as client:
        response = client.post("/api/predict", json={"frames": ["unused"] * 8})
        assert response.status_code == 503
        assert "private-user" not in response.text


def test_websocket_is_explicitly_transport_only_and_bounded():
    main._WS_LAST_ACCEPTED.clear()
    with TestClient(main.app) as client:
        with client.websocket_connect("/ws/inference") as socket:
            ready = socket.receive_json()
            assert ready["inference_ready"] is False and ready["model_loaded"] is False
            socket.send_json({"type": "ping", "ts": 123})
            assert socket.receive_json() == {"type": "pong", "ts": 123}
            socket.send_text("[]")
            assert socket.receive_json()["code"] == "invalid_frame"
            for _ in range(34):
                socket.send_json({"type": "frame", "frame": base64.b64encode(b"transport fixture").decode()})
                response = socket.receive_json()
                assert response["status"] == "model_unavailable" and response["frames"] <= 30
    # A clean disconnect releases the shared capacity slot.
    main._WS_LAST_ACCEPTED.clear()
    with TestClient(main.app) as client:
        with client.websocket_connect("/ws/inference") as socket:
            assert socket.receive_json()["type"] == "ready"


def test_corpus_inventory_does_not_promote_fixtures(tmp_path):
    from app.services.project_evidence import corpus_evidence
    report = corpus_evidence(tmp_path)
    assert report["alphabet"]["recordings"] == 0
    assert len(report["alphabet"]["missing_labels"]) == 26
    assert report["benchmark"]["accuracy"] is None


def test_missing_avhubert_head_is_unavailable_while_backend_stays_online(monkeypatch, tmp_path):
    from app.services.browser_inference import BrowserInference
    monkeypatch.setenv("LIPNET_MODEL_BACKEND", "avhubert")
    monkeypatch.setenv("LIPNET_AVHUBERT_HEAD", str(tmp_path / "private" / "missing-head.pt"))
    monkeypatch.setattr(main, "browser_inference", BrowserInference())
    with TestClient(main.app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        result = response.json()
        assert result["backend"] == "ready" and not result["inference_ready"]
        assert result["capabilities"]["avhubert_head"]["ready"] is False
        assert str(tmp_path) not in response.text
        response = client.post("/api/predict", json={"frames": ["fixture"] * 8})
        assert response.status_code == 503 and "letter" not in response.json()
