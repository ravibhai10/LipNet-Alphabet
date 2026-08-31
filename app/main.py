"""
FastAPI backend for the Visual Speech AI frontend.
Serves the React SPA and provides dataset/research API endpoints.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "alphabet_dataset"
RECORDINGS_ROOT = DATASET_ROOT / "recordings"
REACT_BUILD = PROJECT_ROOT / "frontend" / "dist"


# ── API endpoints (must be defined BEFORE SPA catch-all) ─────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/health")
async def api_health():
    return {"status": "ok"}


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


@app.post("/api/predict")
async def predict():
    """
    Inference endpoint — returns 503 until a trained model backend is deployed.
    The frontend never fabricates predictions; it shows "Model connection
    unavailable" when the backend is unreachable.
    """
    raise HTTPException(status_code=503, detail="Model backend not connected — no trained model deployed")


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


@app.post("/api/training/start")
async def training_start():
    jm = get_job_manager()
    ts = get_training_service()
    if jm.is_running():
        raise HTTPException(status_code=409, detail="A training job is already running.")
    cfg = ts.get_config()
    try:
        job_id = jm.submit(name="training", target=ts.train, kwargs={"config": cfg})
        return {"job_id": job_id}
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


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


@app.get("/api/training/cancel/{job_id}")
async def training_cancel(job_id: str):
    jm = get_job_manager()
    if jm.cancel(job_id):
        return {"cancelled": True}
    raise HTTPException(status_code=404, detail="Job not found or already finished.")


@app.get("/api/training/checkpoints")
async def training_checkpoints():
    ts = get_training_service()
    dirs = ts.list_checkpoints()
    return [{"path": str(d), "name": d.name} for d in dirs]


@app.get("/api/inference/checkpoints")
async def inference_checkpoints():
    isvc = get_inference_service()
    dirs = isvc.list_checkpoints()
    return [{"path": str(d), "name": d.name} for d in dirs]


# ── Live inference API (Phase 2-5) ─────────────────────────────────────

from typing import Optional
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
    try:
        pipeline.load_model(Path(req.ckpt_dir))
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
    full = REACT_BUILD / path.lstrip("/")
    if full.exists() and full.is_file():
        mt = _media_type(full)
        return FileResponse(str(full), media_type=mt)
    idx = REACT_BUILD / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return HTMLResponse(
        "<p>Frontend build not found. Run <code>npm run build</code> in the frontend/ directory.</p>",
        status_code=404,
    )


@app.get("/")
async def root():
    return _spa_response("")


# SPA catch-all must be LAST — must not match /api/* or /health
@app.api_route("/{full_path:path}", methods=["GET", "HEAD"])
async def spa_fallback(full_path: str):
    return _spa_response(full_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
