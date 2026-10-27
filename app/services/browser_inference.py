"""Read-only CPU inference for the saved 26-letter CNN-BiLSTM checkpoint.

This adapter does not claim Conformer or three-stream inference. The model's
softmax scores are uncalibrated, and browser crops are an experimental input.
"""
from __future__ import annotations

import base64
import binascii
import threading
import time
import os
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from src.video_training import resample_temporal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "alphabet_classifier.pt"
MODEL_KIND = "Alphabet CNN-BiLSTM (experimental)"
MIN_FRAMES = 8
MAX_FRAMES = 76
MAX_ENCODED_FRAME = 200_000
logger = logging.getLogger(__name__)


class BrowserInference:
    def __init__(self):
        self._model = None
        self._attempted = False
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()
        self.error = None
        self.backend = os.environ.get("LIPNET_MODEL_BACKEND", "alphabet").lower()
        self.model_kind = MODEL_KIND if self.backend == "alphabet" else "AV-HuBERT + supervised alphabet head"

    def load(self):
        with self._load_lock:
            if self._attempted:
                return self._model is not None
            self._attempted = True
            try:
                if self.backend == "avhubert":
                    from pipeline.avhubert.classification import AVHubertClassifier
                    head_path = os.environ.get("LIPNET_AVHUBERT_HEAD", str(PROJECT_ROOT / "models/avhubert_head.pt"))
                    self._model = AVHubertClassifier(head_path).load()
                    if self._model.metadata["vocabulary"] != [chr(65 + index) for index in range(26)]:
                        self._model = None
                        raise ValueError("The live alphabet endpoint requires an A-Z head. Use the classifier CLI for word heads.")
                    return True
                if self.backend != "alphabet":
                    raise ValueError("LIPNET_MODEL_BACKEND must be alphabet or avhubert.")
                from models.alphabet_model import AlphabetCNNBiLSTM
                if not CHECKPOINT_PATH.is_file():
                    raise FileNotFoundError("The alphabet checkpoint is missing.")
                checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
                config = checkpoint.get("config", {})
                if any(config.get(key, value) != value for key, value in
                       {"num_classes": 26, "t": 30, "h": 140, "w": 46}.items()):
                    raise ValueError("The alphabet checkpoint has an incompatible input or vocabulary.")
                state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
                if not isinstance(state, dict):
                    raise ValueError("The alphabet checkpoint contains no trained state dictionary.")
                model = AlphabetCNNBiLSTM(num_classes=26)
                model.load_state_dict(state, strict=True)
                model.eval()
                torch.set_num_threads(min(torch.get_num_threads(), 4))
                self._model = model
            except Exception as exc:
                logger.warning("Selected browser model could not load: %s", exc)
                self._model = None
                self.error = ("AV-HuBERT classification is unavailable. Supply a compatible trained A-Z head and the CPU runtime."
                              if self.backend == "avhubert" else "The alphabet checkpoint is missing or incompatible. Check the local backend log.")
            return self._model is not None

    def health(self):
        ready = self.load()
        return {
            "model_loaded": ready,
            "inference_ready": ready,
            "inference_reason": (
                f"{self.model_kind} loaded on CPU. Scores are uncalibrated; browser-crop accuracy is not validated."
                if ready else self.error or "Alphabet checkpoint unavailable."
            ),
            "model_kind": self.model_kind,
            "device": "cpu",
            "streaming_ready": False,
        }

    @staticmethod
    def decode_jpegs(frames):
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
            # Browsers send JPEG crops. Restrict format before decoding.
            if not payload.startswith(b"\xff\xd8\xff"):
                raise ValueError("Mouth crops must be JPEG images.")
            crop = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if crop is None or crop.size == 0 or max(crop.shape) > 1024:
                raise ValueError("A mouth crop has invalid dimensions.")
            crops.append(crop)
        return crops

    @staticmethod
    def decode_frames(frames):
        crops = [cv2.resize(crop, (46, 140), interpolation=cv2.INTER_AREA) for crop in BrowserInference.decode_jpegs(frames)]
        # The protected legacy checkpoint was trained with 30 frames. New
        # AV-HuBERT heads use their own explicit 29-frame metadata.
        sequence = resample_temporal(crops, 30) / 255.0
        return torch.from_numpy(sequence).unsqueeze(0).unsqueeze(2)

    def predict(self, frames):
        if not self.load():
            raise RuntimeError(self.error or "Alphabet model unavailable.")
        if not self._predict_lock.acquire(blocking=False):
            raise BlockingIOError("Inference is busy. Try again after the current clip finishes.")
        started = time.perf_counter()
        try:
            if self.backend == "avhubert":
                crops = np.stack([cv2.resize(crop, (96, 96)) for crop in self.decode_jpegs(frames)])
                result = self._model.predict_frames(crops)
                letter = result.pop("label")
                return {**result, "letter": letter, "model": self.model_kind,
                        "temporal": {"frames": len(frames), "model_frames": self._model.metadata["target_frames"]},
                        "uncertainty": {"state": "UNCALIBRATED"},
                        "latency_ms": round((time.perf_counter() - started) * 1000, 2)}
            inputs = self.decode_frames(frames)
            with torch.inference_mode():
                probabilities = self._model(inputs).softmax(dim=-1)[0]
            if not torch.isfinite(probabilities).all():
                raise RuntimeError("The checkpoint returned invalid probabilities.")
            scores, indices = torch.topk(probabilities, k=3)
            values = scores.tolist()
            letters = [chr(ord("A") + index) for index in indices.tolist()]
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum().item()
            return {
                "letter": letters[0], "confidence": values[0],
                "top3": [{"letter": letter, "prob": score} for letter, score in zip(letters, values)],
                "probabilities": [{"letter": chr(65 + index), "prob": value} for index, value in enumerate(probabilities.tolist())],
                "uncertainty": {"margin": values[0] - values[1], "entropy": entropy, "state": "UNCALIBRATED"},
                "temporal": {"frames": len(frames), "model_frames": 30},
                "model": MODEL_KIND,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        finally:
            self._predict_lock.release()


browser_inference = BrowserInference()
