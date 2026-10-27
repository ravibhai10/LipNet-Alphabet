"""Bounded Windows CPU/HTTP smoke checks. JPEG fixtures are not accuracy evidence."""
from __future__ import annotations
import argparse
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys
import threading
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.avhubert.classification import sha256_file
from app.services.project_evidence import corpus_evidence


def request(base, path, body=None, method=None, headers=None):
    payload = json.dumps(body).encode() if body is not None else None
    req = Request(base + path, data=payload, method=method,
                  headers={"Content-Type": "application/json", **(headers or {})})
    try:
        response = urlopen(req, timeout=30)
    except HTTPError as error:
        response = error
    with response:
        raw = response.read().decode()
        headers = {key.lower(): value for key, value in response.headers.items()}
        body = json.loads(raw) if raw and "application/json" in headers.get("content-type", "") else raw or None
        return response.status, headers, body


async def check_socket(base, frame):
    import websockets
    address = base.replace("https://", "wss://").replace("http://", "ws://") + "/ws/inference"
    async with websockets.connect(address, open_timeout=10) as socket:
        ready = json.loads(await asyncio.wait_for(socket.recv(), 10))
        assert ready["type"] == "ready" and ready["inference_ready"] is False
        await socket.send(json.dumps({"type": "ping", "ts": 123}))
        pong = json.loads(await asyncio.wait_for(socket.recv(), 10))
        assert pong == {"type": "pong", "ts": 123}
        await socket.send("[]")
        invalid = json.loads(await asyncio.wait_for(socket.recv(), 10))
        assert invalid["code"] == "invalid_frame"
        await socket.send(json.dumps({"type": "frame", "frame": frame}))
        result = json.loads(await asyncio.wait_for(socket.recv(), 10))
        assert result["status"] == "model_unavailable" and "prediction" not in result
    return {"ready": ready, "ping_pong": True, "invalid_message_rejected": True,
            "unavailable_is_explicit": True, "clean_disconnect": True}


def check_http(base):
    code, _, health = request(base, "/health")
    assert code == 200 and health["status"] == "ok"
    code, _, availability = request(base, "/api/model/availability")
    assert code == 200 and availability == health["capabilities"]
    code, _, evidence = request(base, "/api/evidence")
    assert code == 200 and evidence["benchmark"]["accuracy"] is None
    code, headers, _ = request(base, "/api/predict", method="OPTIONS", headers={
        "Origin": "http://127.0.0.1:5173", "Access-Control-Request-Method": "POST"})
    assert code == 200 and headers.get("access-control-allow-origin") == "http://127.0.0.1:5173"
    assert request(base, "/api/not-a-route")[0] == 404
    assert request(base, "/api/predict", {"frames": []})[0] == 422
    assert request(base, "/api/predict", {"frames": ["bad"] * 8})[0] == (422 if health["inference_ready"] else 503)
    ok, encoded = cv2.imencode(".jpg", np.full((96, 96), 96, dtype=np.uint8))
    assert ok
    frame = base64.b64encode(encoded).decode()
    began = time.perf_counter()
    code, _, prediction = request(base, "/api/predict", {"frames": [frame] * 8})
    transport_ms = round((time.perf_counter() - began) * 1000, 2)
    if health["inference_ready"]:
        assert code == 200 and prediction["letter"] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        assert len(prediction["probabilities"]) == 26
        assert abs(sum(row["prob"] for row in prediction["probabilities"]) - 1) < 1e-5
        barrier = threading.Barrier(4)
        def concurrent_prediction(_):
            barrier.wait(timeout=5)
            return request(base, "/api/predict", {"frames": [frame] * 29})[0]
        with ThreadPoolExecutor(max_workers=4) as pool:
            statuses = list(pool.map(concurrent_prediction, range(4)))
        assert all(status in (200, 429) for status in statuses) and 200 in statuses and 429 in statuses
    else:
        assert code == 503 and "letter" not in prediction
        statuses = []
    return {"base_url": base, "health": health, "availability": availability, "evidence": evidence,
            "cors": "passed", "invalid_input": "422", "unknown_api": "404", "prediction_http_status": code,
            "prediction_transport_ms": transport_ms, "prediction": prediction,
            "concurrent_request_statuses": statuses, "websocket": asyncio.run(check_socket(base, frame)),
            "input_kind": "synthetic JPEG transport fixture; not a recognition accuracy test"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--encoder-smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = {"alphabet_classifier.pt": "7c00d5117b7fae9a76c05980ca70edd680e79be4c8a152ea2595c98ded90c4c3",
                "external/avhubert/base_vox_iter5.pt": "dee8b0651dbea3a21cda0f2c52b956e3a878d5f6ee503d427d6a4044bb95c9cc"}
    hashes = {name: sha256_file(ROOT / "models" / name) for name in expected}
    assert hashes == expected, "A protected checkpoint changed. Inspect before proceeding."
    report = {"time": datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat(),
              "checkpoint_sha256": hashes, "hashes_match_recorded_originals": True, "corpora": corpus_evidence(ROOT)}
    if args.http:
        report["http"] = check_http(args.base_url.rstrip("/"))
    if args.encoder_smoke:
        from pipeline.avhubert.loader import AVHubertLoader
        loader = AVHubertLoader().load()
        began = time.perf_counter()
        features = loader.encode(ROOT / "data/external/grid/s1/corpus/T01/video/s1/bbaf2n.mpg", target_frames=29)
        assert features.shape == (29, 768) and np.isfinite(features).all()
        report["real_encoder_smoke"] = {"features": list(features.shape), "finite": True, "device": "cpu",
                                        "seconds_after_encoder_load": round(time.perf_counter() - began, 2),
                                        "preprocessing": loader.preprocessing()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
