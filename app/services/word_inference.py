"""Experimental CPU inference service for the existing GRID word checkpoint.

Loads the trained checkpoint from runs/word_recognition_v1/checkpoints/best_word_model.pt.
Predicts isolated whole words from browser mouth crops, with explicit uncertainty rejection.
Audio is completely absent and disabled.
"""
from __future__ import annotations
import base64
import binascii
import logging
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import torch

from pipeline.word_recognition.legacy_word_model import WordVideoNet
from src.video_training import resample_temporal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = PROJECT_ROOT / "runs" / "word_recognition_v1" / "checkpoints" / "best_word_model.pt"
MIN_FRAMES = 8
MAX_FRAMES = 76
MAX_ENCODED_FRAME = 250_000
CONFIDENCE_THRESHOLD = 0.35

logger = logging.getLogger(__name__)


class WordInferenceService:
    def __init__(self):
        self._model = None
        self._attempted = False
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()
        self.error = None
        self.vocabulary: list[str] = []
        self.checkpoint_metadata = {}

    def load(self) -> bool:
        with self._load_lock:
            if self._attempted:
                return self._model is not None
            self._attempted = True
            try:
                if not CHECKPOINT_PATH.is_file():
                    raise FileNotFoundError(f"Word recognition checkpoint not found at {CHECKPOINT_PATH}")

                checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
                words = checkpoint.get("vocabulary", checkpoint.get("words"))
                if (not isinstance(words,list) or len(words)<2 or len(set(words))!=len(words)
                        or any(not isinstance(word,str) or not word.strip() or len(word.split())!=1 for word in words)):
                    raise ValueError('Checkpoint must record its actual ordered whole-word vocabulary.')
                self.vocabulary = words
                model_cfg = checkpoint.get("model_config")
                if not isinstance(model_cfg,dict):raise ValueError('Checkpoint architecture metadata is required.')

                model = WordVideoNet(
                    num_classes=len(words),
                    channels=tuple(model_cfg["channels"]),
                    hidden_size=model_cfg["hidden_size"],
                    layers=model_cfg["layers"],
                    dropout=model_cfg["dropout"]
                )
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                model.eval()

                self._model = model
                self.checkpoint_metadata = {
                    "val_loss": checkpoint.get("val_loss"),
                    "val_acc": checkpoint.get("val_acc"),
                    "epoch": checkpoint.get("epoch"),
                    "num_classes": len(words)
                }
                logger.info("WordVideoNet loaded successfully: %d classes (%s)", len(words), words)
                return True
            except Exception as e:
                logger.warning("Failed to load WordVideoNet checkpoint: %s", e)
                self._model = None
                self.error = str(e)
                return False

    def health(self) -> dict:
        ready = self.load()
        return {
            "model_loaded": ready,
            "inference_ready": ready,
            "model_kind": "WordVideoNet (3D-CNN + BiGRU)",
            "task": "Track A: Isolated Whole-Word Recognition",
            "device": "cpu",
            "vocabulary": self.vocabulary,
            "vocabulary_size": len(self.vocabulary),
            "production_ready": False,
            "out_of_vocabulary_detection": "not validated; confidence thresholds are heuristic",
            "recognition_quality": "Loading weights does not establish recognition accuracy; inspect held-out evaluation.",
            "checkpoint": str(CHECKPOINT_PATH.name),
            "metadata": self.checkpoint_metadata if ready else None,
            "error": "The word model is unavailable. Check the local backend log." if not ready else None
        }

    @staticmethod
    def decode_jpegs(frames: list[str]) -> list[np.ndarray]:
        if not isinstance(frames, list) or not MIN_FRAMES <= len(frames) <= MAX_FRAMES:
            raise ValueError(f"Send between {MIN_FRAMES} and {MAX_FRAMES} mouth crops.")
        crops = []
        for encoded in frames:
            if not isinstance(encoded, str) or not encoded or len(encoded) > MAX_ENCODED_FRAME:
                raise ValueError("A mouth crop is missing or exceeds the allowed size.")
            try:
                payload = base64.b64decode(encoded.split(",", 1)[-1], validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError("A mouth crop is not valid base64.") from exc
            if not payload.startswith(b"\xff\xd8\xff"):
                raise ValueError("Mouth crops must be JPEG images.")
            from app.services.image_validation import jpeg_geometry
            jpeg_geometry(payload)
            crop = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if crop is None or crop.size == 0 or max(crop.shape) > 1024:
                raise ValueError("A mouth crop has invalid dimensions.")
            crops.append(crop)
        return crops

    def decode_frames_tensor(self, frames: list[str]) -> torch.Tensor:
        crops = [cv2.resize(c, (64, 64), interpolation=cv2.INTER_LINEAR) for c in self.decode_jpegs(frames)]
        sequence = resample_temporal(crops, 24) / 255.0 # [24, 64, 64] float in [0, 1]
        # WordVideoNet expects [B, 1, T, H, W]
        return torch.from_numpy(sequence).float().unsqueeze(0).unsqueeze(1)

    def predict(self, frames: list[str]) -> dict:
        if not self.load():
            raise RuntimeError(self.error or "Word recognition model unavailable.")
        if not self._predict_lock.acquire(blocking=False):
            raise BlockingIOError("Word inference engine is busy processing a clip.")

        start_time = time.perf_counter()
        try:
            inputs = self.decode_frames_tensor(frames) # [1, 1, 24, 64, 64]
            with torch.inference_mode():
                logits = self._model(inputs)
                probabilities = torch.softmax(logits, dim=-1)[0]

            if not torch.isfinite(probabilities).all():
                raise RuntimeError("Model returned non-finite probabilities.")

            scores, indices = torch.topk(probabilities, k=min(3, len(self.vocabulary)))
            scores_list = scores.tolist()
            indices_list = indices.tolist()

            top_word = self.vocabulary[indices_list[0]]
            top_score = scores_list[0]
            margin = top_score - scores_list[1] if len(scores_list) > 1 else top_score
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum().item()

            is_uncertain = bool(top_score < CONFIDENCE_THRESHOLD or margin < 0.05)
            display_word = top_word if not is_uncertain else "UNCERTAIN — PLEASE TRY AGAIN"

            return {
                "task": "isolated_word",
                "predicted_word": top_word,
                "word": None if is_uncertain else top_word,
                "display_word": display_word,
                "confidence": float(round(top_score, 4)),
                "uncertain": is_uncertain,
                "top_predictions": [
                    {"word": self.vocabulary[idx], "prob": float(round(score, 4))}
                    for idx, score in zip(indices_list, scores_list)
                ],
                "all_probabilities": [
                    {"word": word, "prob": float(round(p, 4))}
                    for word, p in zip(self.vocabulary, probabilities.tolist())
                ],
                "diagnostics": {
                    "margin": float(round(margin, 4)),
                    "entropy": float(round(entropy, 4)),
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                    "rejection_status": "REJECTED_LOW_CONFIDENCE" if is_uncertain else "ACCEPTED"
                },
                "temporal": {
                    "received_frames": len(frames),
                    "model_target_frames": 24,
                    "input_spatial": [64, 64]
                },
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
                "model_name": "WordVideoNet (Track A)",
                "vocabulary": self.vocabulary
            }
        finally:
            self._predict_lock.release()


word_inference = WordInferenceService()
