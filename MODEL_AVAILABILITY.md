# Model availability and API contract

Checked on 2026-10-27 on Windows CPU. File presence, runtime loading, and prediction readiness are distinct.

| Capability | Current state | Meaning |
| --- | --- | --- |
| FastAPI process | Online at `http://127.0.0.1:8000` | Service health only |
| Selected live classifier | Alphabet CNN-BiLSTM, CPU | Strictly loaded from `models/alphabet_classifier.pt` |
| Alphabet labels | Exactly A–Z, indices 0–25 | 26-class softmax; uncalibrated |
| Legacy model input | `[B,30,1,140,46]` grayscale | Browser clips are resized and linearly interpolated to 30 frames |
| AV-HuBERT encoder | Official weights present; real CPU extraction verified | 29 × 768 features; not class predictions |
| AV-HuBERT alphabet head | Not present / not trained | AV-HuBERT live classification is unavailable |
| Research Conformer | Source modules remain available | Separate loader and experimental path; not the active browser classifier |
| Sentence recognition | Unavailable | No compatible trained sentence model/decoder is exposed |
| WebSocket classification | Unavailable | Legacy lifecycle-only transport; live predictions use HTTP |

The trained CNN contains three per-frame 2D convolution blocks (16/32/64 channels), a one-layer bidirectional LSTM with 96 hidden units, and a 26-class output head. It is not a 3D-CNN or an active multistream Conformer. Loading checks dimensions and state keys strictly. Original weights were preserved.

## Health and readiness

`GET /health` and `GET /api/health` return the same fields:

```json
{
  "status": "ok",
  "backend": "ready",
  "model_loaded": true,
  "inference_ready": true,
  "model_kind": "Alphabet CNN-BiLSTM (experimental)",
  "device": "cpu",
  "streaming_ready": false,
  "research_model_loaded": false,
  "capabilities": {
    "active_backend": "alphabet",
    "alphabet_classifier": {"present": true, "ready": true},
    "avhubert_encoder": {"present": true, "loaded": false},
    "avhubert_head": {"present": false, "ready": false, "compatibility": "not_verified"},
    "sentence_recognition": {"ready": false},
    "websocket_classification": {"ready": false, "transport": "lifecycle_only"}
  }
}
```

`inference_reason` provides a public diagnostic. Full model-load errors stay in the local backend log. A service can return HTTP 200 health while `inference_ready` is false. `research_model_loaded` no longer causes a different browser classifier to appear loaded.

`GET /api/model/availability` returns the `capabilities` object. Unselected head files are reported as present with compatibility `not_verified`; existence is never described as a verified classifier.

`GET /api/evidence` returns current registered alphabet recording/class counts, GRID manifest utterance/speaker counts, LRW development clip counts, and the independent benchmark state. No benchmark accuracy or per-class results are registered. This inventory is not an automatic evaluation of newly supplied datasets.

## Alphabet prediction

`POST /api/predict`, JSON body:

```json
{"frames":["<base64 JPEG mouth crop>","<7–75 additional JPEG crops>"]}
```

The list must contain 8–76 JPEG crops; data-URL JPEG prefixes are accepted. Each base64 string is at most 200,000 characters. Decoded crop dimensions must not exceed 1024 pixels per side. The live frontend sends aligned 96 × 96 crops in windows of at most 29 frames. Audio is never captured. The selected model adapter controls its own final spatial normalization and frame count.

A successful response contains:

- `letter`: the actual predicted A–Z string.
- `confidence`: genuine model softmax score in `[0,1]`; not calibrated probability of correctness.
- `top3`: three `{letter, prob}` objects, sorted by score.
- `probabilities`: the full 26-class vector used for frontend EMA.
- `uncertainty`: `state: UNCALIBRATED`; legacy CNN also reports entropy and margin.
- `temporal`: received `frames` and the model's `model_frames`.
- `model`: selected model description.
- `latency_ms`: backend inference elapsed time, excluding network transit.

The UI requires two consistent windows before displaying an EMA result. Request cancellation aborts delivery and prevents stale publication; an already executing CPU forward pass may finish in the backend. The service's nonblocking inference lock prevents concurrent forward passes, including while cancelled client requests wind down.

| HTTP result | Response |
| --- | --- |
| 422 invalid schema | Standard FastAPI `{"detail":[...]}` validation errors |
| 422 invalid JPEG/sequence | `{"detail":"..."}` with actionable input error |
| 429 inference busy | `{"detail":"..."}`; no automatic prediction retry |
| 503 selected model unavailable | `{"detail":"..."}`; no fabricated result or private model path |
| 404 unknown API | `{"detail":"API endpoint not found."}` rather than SPA HTML |

## Connection behavior

- Development frontend: `http://127.0.0.1:5173`; API: `http://127.0.0.1:8000`.
- `VITE_API_BASE_URL` configures the frontend API. Restart Vite after editing `.env.local`. Same-origin built hosting uses relative API paths.
- CORS allows the documented local 5173/5174/3000 and backend origins. `LIPNET_CORS_ORIGINS` overrides the comma-separated allowlist for another setup.
- Health requests time out after 10 seconds per supported health URL. The client polls every 10 seconds online; offline polling backs off to a maximum of 30 seconds. HTTP prediction timeout is 20 seconds; evidence timeout is 8 seconds.
- Components cancel active health, evidence, and prediction requests on cleanup. The camera pipeline has one pending latest window and one in-flight request. The backend returns 429 for competing requests.
- An offline backend, missing endpoint, invalid payload, model unavailability, and camera permission error have separate behavior. Preview remains usable offline.

## Legacy WebSocket transport

The main story and live workspace do not use WebSockets. `/ws/inference` remains available for lifecycle compatibility:

1. Server sends `{"type":"ready","model_loaded":false,"inference_ready":false,"transport":"lifecycle_only","prediction_endpoint":"/api/predict"}`.
2. Client may send `{"type":"ping","ts":123}`; server echoes `{"type":"pong","ts":123}`.
3. Client may send `{"type":"frame","frame":"<base64 bytes>"}` or a binary payload. This transport validates encoding/size, not visual recognition.
4. Valid payloads return `{"type":"prediction","status":"model_unavailable","frames":1,"latency_ms":0.0}`; the frame count is bounded at 30. Latency is measured, not fixed to this illustrative value.
5. Invalid input returns `{"type":"error","code":"invalid_frame","message":"..."}`. Idle timeout is 30 seconds; maximum connections are four, with a per-IP one-second connection rate window and close code 1013 when limited.

The optional client rejects duplicate connections and excessive outgoing buffering, retries at most five times with exponential backoff capped at 15 seconds, and cancels reconnect/ping timers on intentional close or unmount. This prevents the earlier reconnect-after-cleanup defect. Classification readiness stays false even if another HTTP model is loaded.

## Activate AV-HuBERT after real training

Only a verified complete A–Z head can serve the browser endpoint. In a new PowerShell backend terminal:

```powershell
$env:LIPNET_MODEL_BACKEND = 'avhubert'
$env:LIPNET_AVHUBERT_HEAD = (Resolve-Path 'runs/avhubert/my_head_v1/head.pt').Path
.\.venv-avhubert-cpu\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Stop the existing backend before binding that port. The loader verifies vocabulary, training/evaluation metadata, input dimensions, encoder SHA-256, and preprocessing. A missing/incompatible head remains unavailable. Set `LIPNET_MODEL_BACKEND` back to `alphabet` and restart to use the preserved model.

See [README.md](README.md) for Windows startup, [ACCURACY_PIPELINE.md](ACCURACY_PIPELINE.md) for authentic data and head training/evaluation, and [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for executed results.
