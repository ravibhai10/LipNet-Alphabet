# Workstream D — Application integration review

Owner scope: inspection only; no edits to existing `app/` files.

## Existing routes (app/main.py)

| Route | Method | Current behavior |
|---|---|---|
| `/health`, `/api/health` | GET | `{status: ok}` |
| `/api/dataset/stats` | GET | counts recordings in `alphabet_dataset/recordings` |
| `/api/predict` | POST | **returns 503** — “Model backend not connected” |
| `/api/dataset/validate` | GET | runs `DatasetValidator`, returns report |
| `/api/dataset/build_manifest` | POST | builds manifest.jsonl |
| `/api/dataset/create_splits` | POST | train/val/test by speaker |
| `/api/live/start/stop/state` | POST/GET | live webcam → Phase 2→5 |
| `/api/live/save_recording` | POST | saves test recording |
| `/api/live/cameras` | GET | probes camera indices (best-effort) |
| STR `/{full_path:path}` | GET/HEAD | SPA fallback |

## Integration points (smallest)

1. **`/api/predict` (POST)**: currently raises 503. Replace with a thin wrapper that:
   - optionally accepts `speaker_id`, `audio_path`, `video_path` (or `frame:bgr` base64),
   - runs the AV-HuBERT adapter → feature `[T, D]`,
   - converts `[T, D]` → `[T, 192]` via a linear projection to match Phase 3/4 `input_dim=192`,
   - falls back to the existing `AlphabetCNNBiLSTM` classifier (already loads `alphabet_classifier.pt`) when AV-HuBERT is unavailable.
   - **Keep the 503/503 fallback + auto-offline UI behavior** until the AV-HuBERT weights are present.

2. **Model loading**: `InferenceService` (canonical) loads `alphabet_classifier.pt`. Add a second service/`pipeline/avhubert/adapter.py` that loads `models/external/avhubert/avhubert_base.pt` in `models/external/avhubert/loader.py` and exposes `encode(video_path) -> np.ndarray [T, 1024]` (large variant) or `[T, 768]` (base). **Do not rewrite `inference_service.py`.**

3. **Request/response schema** (new):
   ```json
   // POST /api/predict
   {"kind": "avhubert", "video_path": "...", "speaker_id": "grid_t01"}
   // or  {"kind": "alphabet", "frame_b64": "..."}
   ```
   Reponse:
   ```json
   {"letter": "A", "confidence": 0.93, "top_k": [["A",0.93],["B",0.04],...], "model": "avhubert"}
   ```

## Mocked integration tests (drafts)

- Mock `pipeline.avhubert.adapter.AVHuBERTAdapter.encode` to return a fixed `[T, 192]` array; assert `/api/predict` returns `200` with canonical `PredictionResult` shape and correct MIME/`application/json`.
- Mock the model loader to raise `FileNotFoundError`; assert `/api/predict` returns `503` with `detail="Model backend not connected"`.
- Assert `verify_phase2.py` still passes after changes.

## Guardrails
- No unverified model wired into production routes (documented as a blocker until the checksummed AV-HuBERT checkpoint loads and serves a real prediction).
