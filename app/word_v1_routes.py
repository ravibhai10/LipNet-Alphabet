"""
FastAPI Router for v1 High-Speed Neural Visual Speech API and WebSocket Stream.
Provides /api/v1/predict/word and /ws/live-speech.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from app.services.neural_word_service import neural_word_service

logger = logging.getLogger(__name__)

router = APIRouter()


class WordPredictionV1Request(BaseModel):
    frames: List[str] = Field(..., min_length=1, description="Base64 encoded mouth crop frames")
    convention: Optional[str] = Field("aligned_mouth_140x46", description="Cropping convention")
    return_attention: Optional[bool] = Field(True, description="Whether to compute 29-frame temporal attention")


@router.get("/api/v1/word/health")
def word_v1_health():
    """Get the current operational health of the neural LRW-500 engine."""
    return neural_word_service.health()


@router.post("/api/v1/predict/word")
def predict_word_v1(req: WordPredictionV1Request):
    """Predict isolated word from mouth crop sequence with LRW-500 Neural Engine."""
    try:
        result = neural_word_service.predict(req.frames, return_attention=req.return_attention if req.return_attention is not None else True)
        return result
    except ValueError as val_err:
        raise HTTPException(status_code=422, detail=str(val_err)) from val_err
    except BlockingIOError as block_err:
        raise HTTPException(status_code=429, detail=str(block_err)) from block_err
    except Exception as exc:
        logger.exception("Word prediction v1 failed: %s", exc)
        raise HTTPException(status_code=500, detail="Neural word analysis failed. Check backend log.") from exc


@router.websocket("/ws/live-speech")
async def websocket_live_speech(websocket: WebSocket):
    """High-speed WebSocket streaming channel for real-time visual speech decoding."""
    await websocket.accept()
    try:
        # Handshake
        health = neural_word_service.health()
        await websocket.send_json({
            "type": "connected",
            "protocol": "lipnet-v1-live-speech",
            "engine_mode": health.get("engine_mode", "neural_cpu"),
            "vocabulary_size": health.get("vocabulary_size", 500),
            "timestamp": time.time(),
        })

        while True:
            raw_msg = await websocket.receive_text()
            try:
                msg = json.loads(raw_msg)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON format."})
                continue

            msg_type = msg.get("type", "frames")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong", "time": time.time()})
                continue

            if msg_type in ("frames", "predict"):
                frames = msg.get("frames", [])
                if not isinstance(frames, list) or len(frames) < 4:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Expected at least 4 frames, received {len(frames) if isinstance(frames, list) else 0}",
                    })
                    continue

                # Run inference off the event loop
                try:
                    result = await asyncio.to_thread(neural_word_service.predict, frames, True)
                    await websocket.send_json({
                        "type": "prediction",
                        "data": result,
                        "timestamp": time.time(),
                    })
                except Exception as pred_err:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Inference error: {str(pred_err)}",
                    })

    except WebSocketDisconnect:
        logger.info("Live speech WebSocket disconnected cleanly.")
    except Exception as exc:
        logger.warning("Live speech WebSocket terminated: %s", exc)
