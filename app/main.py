"""
FastAPI backend for the Visual Speech AI frontend.
Serves the React SPA and provides dataset/research API endpoints.
"""
from __future__ import annotations

import json
import sys
import asyncio
import base64
import binascii
import time
import threading
import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, WebSocketException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from app.services.browser_inference import browser_inference, MODEL_KIND, CHECKPOINT_PATH
from app.services.project_evidence import corpus_evidence

from app.services.dataset_service import get_dataset_service
from app.services.training_service import get_training_service
from app.services.job_manager import get_job_manager
from app.services.inference_service import (
    get_inference_service,
    IncompatibleCheckpointError,
)
from app.services.live_pipeline import (
    get_live_pipeline,
    shutdown_live_pipeline,
)
from app.services.camera_service import (
    get_camera_service,
    CameraUnavailableError,
)

app = FastAPI(
    title="Visual Speech AI",
    description="Research system for visual speech decoding of spoken alphabet letters",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.environ.get("LIPNET_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174,http://127.0.0.1:8000,http://localhost:8000").split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "alphabet_dataset"
RECORDINGS_ROOT = DATASET_ROOT / "recordings"
REACT_BUILD = PROJECT_ROOT / "frontend" / "dist"
LEGACY_FRONTEND = PROJECT_ROOT / "app" / "frontend"

def _is_safe_subpath(target_path: Path, base_dir: Path) -> bool:
    try:
        resolved_target = target_path.resolve()
        resolved_base = base_dir.resolve()
        return resolved_base in resolved_target.parents or resolved_target == resolved_base
    except Exception:
        return False

_WS_CONNECTIONS = asyncio.Semaphore(4)
_WS_MAX_MESSAGE_BYTES = 1_500_000
_WS_MAX_BUFFERED_FRAMES = 30
_WS_LAST_ACCEPTED: dict[str, float] = {}
_WS_RATE_WINDOW_SECONDS = 1.0


def _decode_ws_frame(payload: object) -> bytes:
    if not isinstance(payload, str) or not payload:
        raise ValueError("frame must be a non-empty base64 string")
    encoded = payload.split(",", 1)[-1]
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("frame is not valid base64") from exc
    if not raw or len(raw) > _WS_MAX_MESSAGE_BYTES:
        raise ValueError("frame size is invalid")
    return raw


# ── API endpoints (must be defined BEFORE SPA catch-all) ─────────────────────

def model_availability(browser):
    selected = browser_inference.backend
    head = Path(os.environ.get("LIPNET_AVHUBERT_HEAD", str(PROJECT_ROOT / "models/avhubert_head.pt")))
    ready = browser["inference_ready"]
    return {
        "active_backend": selected,
        "alphabet_classifier": {"present": CHECKPOINT_PATH.is_file(), "ready": selected == "alphabet" and ready},
        "avhubert_encoder": {"present": (PROJECT_ROOT / "models/external/avhubert/base_vox_iter5.pt").is_file(),
                             "loaded": selected == "avhubert" and ready},
        "avhubert_head": {"present": head.is_file(), "ready": selected == "avhubert" and ready,
                          "compatibility": "verified" if selected == "avhubert" and ready else "not_verified"},
        "sentence_recognition": {"ready": False},
        "websocket_classification": {"ready": False, "transport": "lifecycle_only"},
    }

@app.get("/health")
async def health():
    browser = await asyncio.to_thread(browser_inference.health)
    return {
        "status": "ok",
        "backend": "ready",
        **browser,
        "model_loaded": browser["model_loaded"],
        "research_model_loaded": get_inference_service().has_loaded_model(),
        "capabilities": model_availability(browser),
    }


@app.get("/api/health")
async def api_health():
    return await health()


@app.get("/api/model/availability")
async def availability():
    browser = await asyncio.to_thread(browser_inference.health)
    return model_availability(browser)


@app.get("/api/evidence")
async def project_evidence():
    return await asyncio.to_thread(corpus_evidence, PROJECT_ROOT)


@app.get("/api/dataset/stats")
async def dataset_stats():
    """Return dataset statistics from the real recording directory."""
    total_recordings = 0
    speakers: set = set()
    letter_counts: dict = {}

    if RECORDINGS_ROOT.exists():
        for letter_dir in sorted(RECORDINGS_ROOT.iterdir()):
            if not letter_dir.is_dir() or not letter_dir.name.isalpha():
                continue
            letter = letter_dir.name.upper()
            count = 0
            for spk_dir in letter_dir.iterdir():
                if not spk_dir.is_dir():
                    continue
                speakers.add(spk_dir.name)
                for sess_dir in spk_dir.iterdir():
                    if not sess_dir.is_dir():
                        continue
                    for rec_dir in sess_dir.iterdir():
                        if rec_dir.is_dir():
                            count += 1
            if count:
                letter_counts[letter] = count
                total_recordings += count

    return {
        "recordings": total_recordings,
        "speakers": len(speakers),
        "letters": len(letter_counts),
    }



@app.get("/api/overview")
async def dataset_overview():
    """Return overview statistics matching index.html format."""
    total_recordings = 0
    speakers = set()
    sessions = set()
    letter_stats = []

    if RECORDINGS_ROOT.exists():
        for letter_char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            letter_dir = RECORDINGS_ROOT / letter_char
            count = 0
            if letter_dir.exists() and letter_dir.is_dir():
                for spk_dir in letter_dir.iterdir():
                    if not spk_dir.is_dir():
                        continue
                    speakers.add(spk_dir.name)
                    for sess_dir in spk_dir.iterdir():
                        if not sess_dir.is_dir():
                            continue
                        sessions.add(sess_dir.name)
                        for rec_dir in sess_dir.iterdir():
                            if rec_dir.is_dir():
                                count += 1
            letter_stats.append({"letter": letter_char, "total_recordings": count})
            total_recordings += count

    return {
        "total_recordings": total_recordings,
        "total_speakers": len(speakers),
        "total_sessions": len(sessions),
        "letters": letter_stats,
    }


@app.get("/api/recordings/{letter}")
async def list_recordings_for_letter(letter: str):
    """List all recordings available for a specific letter."""
    letter_upper = letter.upper()
    if len(letter_upper) != 1 or not "A" <= letter_upper <= "Z":
        raise HTTPException(status_code=400, detail="Choose a letter from A to Z.")
    letter_dir = RECORDINGS_ROOT / letter_upper
    recordings_list = []

    if letter_dir.exists() and letter_dir.is_dir():
        for spk_dir in sorted(letter_dir.iterdir()):
            if not spk_dir.is_dir():
                continue
            for sess_dir in sorted(spk_dir.iterdir()):
                if not sess_dir.is_dir():
                    continue
                for rec_dir in sorted(sess_dir.iterdir()):
                    if not rec_dir.is_dir():
                        continue
                    vid_file = rec_dir / "video.mp4"
                    meta_file = rec_dir / "metadata.json"
                    duration_ms = 0
                    if meta_file.exists():
                        try:
                            meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
                            duration_ms = meta_data.get("duration_ms", 0)
                        except Exception:
                            pass
                    rel_path = f"{letter_upper}/{spk_dir.name}/{sess_dir.name}/{rec_dir.name}/video.mp4"
                    recordings_list.append({
                        "path": rel_path,
                        "speaker_id": spk_dir.name,
                        "session_id": sess_dir.name,
                        "recording_id": rec_dir.name,
                        "duration_ms": duration_ms,
                    })

    return {"letter": letter_upper, "recordings": recordings_list}


@app.get("/api/video/{video_path:path}")
async def serve_video(video_path: str):
    """Safely stream a recording video to the frontend."""
    candidate = (RECORDINGS_ROOT / video_path).resolve()
    if not _is_safe_subpath(candidate, RECORDINGS_ROOT) or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(str(candidate), media_type="video/mp4")


@app.get("/api/recording/{letter}/{speaker}/{session}/{recording_id}")
async def get_recording_meta(letter: str, speaker: str, session: str, recording_id: str):
    """Retrieve full metadata for a specific recording."""
    meta_path = (RECORDINGS_ROOT / letter.upper() / speaker / session / recording_id / "metadata.json").resolve()
    if not _is_safe_subpath(meta_path, RECORDINGS_ROOT) or not meta_path.is_file():
        raise HTTPException(status_code=404, detail="Recording metadata not found")
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load metadata: {exc}")


class PredictRequest(BaseModel):
    frames: list[str] = Field(min_length=8, max_length=76)


@app.post("/api/predict")
def predict(req: PredictRequest):
    """Run the saved experimental alphabet classifier on browser mouth crops."""
    try:
        return browser_inference.predict(req.frames)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except BlockingIOError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except RuntimeError as exc:
        logging.getLogger(__name__).warning("Browser inference unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="The selected classifier is unavailable. Check model readiness and the local backend log.") from exc


@app.websocket("/ws/inference")
async def websocket_inference(websocket: WebSocket):
    """Bounded streaming protocol for browser inference clients.

    This legacy transport does not run classification. The live product uses
    bounded HTTP windows at /api/predict. Keep readiness explicit on this
    lifecycle-only endpoint and never imply that a loaded HTTP model streams.
    """
    client_key = websocket.client.host if websocket.client else "unknown"
    now = time.monotonic()
    if len(_WS_LAST_ACCEPTED) > 1024:
        cutoff = now - _WS_RATE_WINDOW_SECONDS
        for key, accepted_at in list(_WS_LAST_ACCEPTED.items()):
            if accepted_at < cutoff:
                del _WS_LAST_ACCEPTED[key]
    if now - _WS_LAST_ACCEPTED.get(client_key, 0.0) < _WS_RATE_WINDOW_SECONDS:
        await websocket.close(code=1013, reason="connection rate limit")
        return
    _WS_LAST_ACCEPTED[client_key] = now
    if not await _try_acquire_ws_slot():
        await websocket.close(code=1013, reason="inference capacity reached")
        return
    await websocket.accept()
    buffered_frames: list[bytes] = []
    try:
        await websocket.send_json({"type": "ready", "model_loaded": False, "inference_ready": False,
                                   "transport": "lifecycle_only", "prediction_endpoint": "/api/predict"})
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "error", "code": "idle_timeout", "message": "No frame received for 30 seconds."})
                break
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                raw = message["bytes"]
                if not isinstance(raw, bytes) or not raw or len(raw) > _WS_MAX_MESSAGE_BYTES:
                    await websocket.send_json({"type": "error", "code": "invalid_frame", "message": "Binary frame size is invalid."})
                    continue
            else:
                text = message.get("text")
                try:
                    payload = json.loads(text) if isinstance(text, str) else {}
                    if not isinstance(payload, dict):
                        raise ValueError("WebSocket messages must be JSON objects.")
                    if payload.get("type") == "ping":
                        await websocket.send_json({"type": "pong", "ts": payload.get("ts")})
                        continue
                    raw = _decode_ws_frame(payload.get("frame"))
                except (ValueError, json.JSONDecodeError, TypeError) as exc:
                    await websocket.send_json({"type": "error", "code": "invalid_frame", "message": str(exc)})
                    continue
            buffered_frames.append(raw)
            if len(buffered_frames) > _WS_MAX_BUFFERED_FRAMES:
                del buffered_frames[:-_WS_MAX_BUFFERED_FRAMES]
            started = time.perf_counter()
            await websocket.send_json({
                "type": "prediction", "status": "model_unavailable",
                "frames": len(buffered_frames),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            })
    except WebSocketDisconnect:
        pass
    except (RuntimeError, WebSocketException, ValueError):
        pass
    finally:
        _WS_CONNECTIONS.release()


async def _try_acquire_ws_slot() -> bool:
    try:
        await asyncio.wait_for(_WS_CONNECTIONS.acquire(), timeout=0.05)
        return True
    except asyncio.TimeoutError:
        return False


@app.get("/api/dataset/validate")
async def dataset_validate():
    ds = get_dataset_service()
    report = ds.validate()
    return {
        "passed": not report.has_critical_errors(),
        "critical_errors": report.critical_errors,
        "results": [
            {"check": r.check_name, "passed": r.passed, "message": r.message, "details": r.details}
            for r in report.results
        ],
    }


@app.post("/api/dataset/build-manifest")
async def dataset_build_manifest():
    ds = get_dataset_service()
    n = ds.build_manifest()
    return {"entries": n}


@app.post("/api/dataset/create-splits")
async def dataset_create_splits():
    ds = get_dataset_service()
    counts = ds.create_splits()
    return counts


@app.get("/api/training/check")
async def training_precheck():
    ts = get_training_service()
    check = ts.pre_training_check()
    return {
        "passed": check.passed,
        "recordings": check.dataset_recordings,
        "speakers": check.dataset_speakers,
        "splits": check.dataset_splits,
        "data_source_ok": check.data_source_ok,
        "errors": check.errors,
        "warnings": check.warnings,
    }


class TrainingStartRequest(BaseModel):
    epochs: int = Field(default=30, ge=1, le=500)
    batch_size: int = Field(default=16, ge=1, le=256)
    lr: float = Field(default=0.001, gt=0, le=1)


@app.post("/api/training/start")
def training_start(req: TrainingStartRequest):
    jm = get_job_manager()
    ts = get_training_service()
    if jm.is_running():
        raise HTTPException(status_code=409, detail="A training job is already running.")
    check = ts.pre_training_check()
    if not check.passed:
        raise HTTPException(status_code=409, detail="; ".join(check.errors))
    cfg = ts.get_config()
    ready = threading.Event()
    holder = {}

    def run_training():
        ready.wait()
        return ts.train(
            job_id=holder["job_id"], config=cfg, max_epochs=req.epochs,
            batch_size=req.batch_size, lr=req.lr, device="cpu",
        )

    try:
        job_id = jm.submit(name="training", target=run_training)
        holder["job_id"] = job_id
        jm.get(job_id).progress["total_epochs"] = req.epochs
        ready.set()
        return {"job_id": job_id}
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/training/status/{job_id}")
async def training_status(job_id: str):
    jm = get_job_manager()
    job = jm.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id": job.job_id,
        "state": job.state.value,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "progress": job.progress,
        "error": job.error,
    }


@app.post("/api/training/cancel/{job_id}")
@app.get("/api/training/cancel/{job_id}")
async def training_cancel(job_id: str):
    jm = get_job_manager()
    job = jm.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found.")
    if job.state.value == "RUNNING":
        raise HTTPException(
            status_code=409,
            detail="Running training cannot be interrupted safely by this backend. The job is still running.",
        )
    if jm.cancel(job_id):
        return {"cancelled": True}
    raise HTTPException(status_code=409, detail="The training job has already finished.")


def _saved_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _checkpoint_records():
    directories = list(get_training_service().list_checkpoints())
    if CHECKPOINT_PATH.is_file():
        directories.append(CHECKPOINT_PATH.parent)
    records = []
    for directory in directories:
        summary = _saved_json(directory / "summary.json") or _saved_json(directory / "training_summary.json")
        files = [item for item in directory.iterdir() if item.is_file() and item.suffix in (".pt", ".ckpt")]
        records.append({
            "name": "alphabet_classifier.pt" if directory == CHECKPOINT_PATH.parent else directory.name,
            "path": str(directory),
            "created": datetime.fromtimestamp(directory.stat().st_mtime, timezone.utc).isoformat(),
            "epoch": summary.get("epochs_run", summary.get("epochs")),
            "best_val": summary.get("best_val_accuracy", summary.get("best_val_loss")),
            "config_hash": summary.get("config_hash"),
            "repr_version": summary.get("repr_version"),
            "size_bytes": sum(item.stat().st_size for item in files),
            "model_kind": MODEL_KIND if directory == CHECKPOINT_PATH.parent else "Temporal letter decoder",
        })
    return records


@app.get("/api/training/checkpoints")
async def training_checkpoints():
    return _checkpoint_records()


@app.get("/api/inference/checkpoints")
async def inference_checkpoints():
    service = get_inference_service()
    browser = await asyncio.to_thread(browser_inference.health)
    checkpoints = [
        {"path": str(directory), "name": directory.name,
         "compatible": service.is_loaded(directory), "loaded": service.is_loaded(directory),
         "note": "Loaded temporal checkpoint." if service.is_loaded(directory) else "Compatibility is checked when loaded by the local pipeline."}
        for directory in service.list_checkpoints()
    ]
    if CHECKPOINT_PATH.is_file():
        checkpoints.insert(0, {
            "path": str(CHECKPOINT_PATH), "name": CHECKPOINT_PATH.name,
            "compatible": browser["inference_ready"] and browser_inference.backend == "alphabet",
            "loaded": browser["model_loaded"] and browser_inference.backend == "alphabet",
            "model_kind": MODEL_KIND,
            "note": browser["inference_reason"] if browser_inference.backend == "alphabet" else "Protected CNN-BiLSTM checkpoint. Compatibility is checked when selected.",
        })
    avhubert_path = PROJECT_ROOT / "models/external/avhubert/base_vox_iter5.pt"
    if avhubert_path.is_file():
        selected = browser_inference.backend == "avhubert"
        checkpoints.append({"path": str(avhubert_path), "name": avhubert_path.name,
                            "compatible": selected and browser["inference_ready"],
                            "loaded": selected and browser["model_loaded"], "model_kind": "AV-HuBERT Base encoder",
                            "note": browser["inference_reason"] if selected else "Pretrained encoder weights are available. Train a supervised head before alphabet or word classification."})
    return {"checkpoints": checkpoints, "model_loaded": browser["model_loaded"] or service.has_loaded_model()}


@app.get("/api/experiments")
async def experiments():
    directories = [PROJECT_ROOT / "models"]
    runs_root = PROJECT_ROOT / "runs"
    if runs_root.exists():
        for pattern in ("training_summary.json", "summary.json"):
            directories.extend(item.parent for item in list(runs_root.rglob(pattern))[:100])
    runs = []
    for directory in dict.fromkeys(directories):
        summary = _saved_json(directory / "training_summary.json") or _saved_json(directory / "summary.json")
        if not summary:
            continue
        curve = _saved_json(directory / "training_curve.json")
        train = curve.get("train_loss", [])
        val = curve.get("val_loss", [])
        accuracy = curve.get("val_acc", [])
        history = []
        for index in range(max(len(train), len(val), len(accuracy))):
            row = {"epoch": index + 1}
            for key, values in (("train_loss", train), ("val_loss", val), ("val_accuracy", accuracy)):
                if index < len(values) and isinstance(values[index], (int, float)):
                    row[key] = values[index]
            history.append(row)
        runs.append({
            "name": "Alphabet CNN-BiLSTM" if directory == PROJECT_ROOT / "models" else directory.name,
            "path": str(directory.relative_to(PROJECT_ROOT)),
            "epochs": summary.get("epochs_run", summary.get("epochs", len(history))),
            "best_val_accuracy": summary.get("best_val_accuracy"),
            "test_accuracy": summary.get("test_accuracy"),
            "device": summary.get("device"),
            "history": history,
        })
    return {"runs": runs}


from pydantic import BaseModel, Field


class LiveStartRequest(BaseModel):
    ckpt_dir: str
    camera_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 25


class LiveStopRequest(BaseModel):
    pass


class LiveSaveRequest(BaseModel):
    out_dir: Optional[str] = None


@app.post("/api/live/start")
async def live_start(req: LiveStartRequest):
    isvc = get_inference_service()
    pipeline = get_live_pipeline()
    ckpt = Path(req.ckpt_dir).resolve()
    allowed_bases = [PROJECT_ROOT / "runs", PROJECT_ROOT / "models"]
    if not any(_is_safe_subpath(ckpt, base) for base in allowed_bases):
        raise HTTPException(status_code=400, detail="Invalid checkpoint path: must be inside runs/ or models/")
    try:
        pipeline.load_model(ckpt)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except IncompatibleCheckpointError as e:
        raise HTTPException(status_code=409, detail=str(e))
    try:
        pipeline.start()
    except CameraUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"started": True, "ckpt_dir": req.ckpt_dir}


@app.post("/api/live/stop")
async def live_stop():
    pipeline = get_live_pipeline()
    pipeline.stop()
    return {"stopped": True}


@app.get("/api/live/state")
async def live_state():
    pipeline = get_live_pipeline()
    s = pipeline.get_state()
    return {
        "state": s.state,
        "running": pipeline.is_running(),
        "model_loaded": get_inference_service().has_loaded_model(),
        "letter": s.letter,
        "confidence": s.confidence,
        "entropy": s.entropy,
        "margin": s.margin,
        "top_k": s.top_k,
        "detail": s.detail,
        "error": s.error,
        "buffer_size": s.buffer_size,
        "buffer_capacity": s.buffer_capacity,
        "buffer_has_min": s.buffer_has_min,
        "face_recently": s.face_recently,
        "fps_camera": s.fps_camera,
        "fps_inference": s.fps_inference,
        "last_inference_ms": s.last_inference_ms,
        "last_phase2_ms": s.last_phase2_ms,
        "last_phase3_ms": s.last_phase3_ms,
        "last_phase4_ms": s.last_phase4_ms,
        "last_phase5_ms": s.last_phase5_ms,
        "stabilize_count": s.stabilize_count,
        "frames_inferred": s.frames_inferred,
        "frames_with_face": s.frames_with_face,
    }


@app.post("/api/live/save_recording")
async def live_save(req: LiveSaveRequest):
    pipeline = get_live_pipeline()
    out = Path(req.out_dir) if req.out_dir else (
        PROJECT_ROOT / "test_recordings" / f"live_{int(__import__('time').time())}"
    )
    try:
        n = pipeline.save_test_recording(out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"saved": n, "path": str(out)}


@app.get("/api/live/cameras")
async def live_cameras():
    """Probe OpenCV for available camera indices (best-effort).

    The actual device-open check lives in CameraService.start() which
    surfaces CameraUnavailableError. This endpoint only returns indices
    to try.
    """
    return {"cameras": [{"index": i, "available": None} for i in range(4)]}


# ── SPA serving ──────────────────────────────────────────────────────────────

def _media_type(path: Path) -> Optional[str]:
    ext = path.suffix.lower()
    mapping = {
        ".js": "application/javascript",
        ".mjs": "application/javascript",
        ".css": "text/css",
        ".woff": "font/woff",
        ".woff2": "font/woff2",
        ".svg": "image/svg+xml",
        ".ico": "image/x-icon",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".mp4": "video/mp4",
    }
    return mapping.get(ext)


def _spa_response(path: str):
    # Try compiled React bundle first
    full = (REACT_BUILD / path.lstrip("/")).resolve()
    if _is_safe_subpath(full, REACT_BUILD) and full.exists() and full.is_file():
        mt = _media_type(full)
        return FileResponse(str(full), media_type=mt)
    idx = REACT_BUILD / "index.html"
    if idx.exists() and idx.is_file():
        return FileResponse(str(idx))

    # Fallback to embedded frontend
    legacy_full = (LEGACY_FRONTEND / path.lstrip("/")).resolve()
    if _is_safe_subpath(legacy_full, LEGACY_FRONTEND) and legacy_full.exists() and legacy_full.is_file():
        mt = _media_type(legacy_full)
        return FileResponse(str(legacy_full), media_type=mt)
    legacy_idx = LEGACY_FRONTEND / "index.html"
    if legacy_idx.exists() and legacy_idx.is_file():
        return FileResponse(str(legacy_idx))

    return HTMLResponse(
        "<p>Frontend build not found. Place built assets in frontend/dist or app/frontend.</p>",
        status_code=404,
    )


@app.get("/")
async def root():
    return _spa_response("")


# SPA catch-all must be LAST — must not match /api/* or /health
@app.api_route("/{full_path:path}", methods=["GET", "HEAD"])
async def spa_fallback(full_path: str):
    if full_path == "api" or full_path.startswith(("api/", "ws/")):
        raise HTTPException(status_code=404, detail="API endpoint not found.")
    return _spa_response(full_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
