"""InferenceService - thread-safe model loading + canonical inference.

Reuses the existing alphabet/ modules for:
    - Phase 4:  TemporalEncoder (4x Conformer)
    - Phase 5:  MultiTaskAlphabetModel (CTC head)
    - Decoder:  GreedyCTC / ClassMappingGreedyCTCDecoder
    - Uncertainty: single_letter_uncertainty + decide_state
    - Calibration: temperature + thresholds from calibration.json
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    letter: str = ""
    confidence: float = 0.0
    entropy: float = 0.0
    margin: float = 0.0
    top_k: List[Tuple[str, float]] = field(default_factory=list)
    state: str = "PROCESSING"
    raw_probs: Optional[List[float]] = None


class ModelHandle:
    """A loaded checkpoint + canonical decoder wiring."""

    def __init__(self, ckpt_dir, model, vocab, calibration, config):
        self.ckpt_dir = Path(ckpt_dir)
        self.model = model
        self.vocab = vocab
        self.calibration = calibration
        self.config = config
        self._lock = threading.Lock()
        self._device = next(model.parameters()).device

    def predict(self, fused: np.ndarray) -> PredictionResult:
        """Phase 4+5 inference on a [T, 192] fused temporal sequence."""
        import numpy as _np
        from alphabet.decoding import ClassMappingGreedyCTCDecoder
        from alphabet.uncertainty import single_letter_uncertainty, decide_state

        if fused is None or fused.size == 0 or fused.ndim != 2:
            raise ValueError(
                f"fused must be non-empty [T, D] array; got shape "
                f"{None if fused is None else fused.shape}"
            )
        with self._lock:
            self.model.eval()
            with torch.no_grad():
                x = torch.from_numpy(fused).float().unsqueeze(0).to(self._device)
                T = x.shape[1]
                if T < 1:
                    raise ValueError("T must be >= 1")
                mask = torch.ones(1, T, dtype=torch.bool, device=self._device)
                outputs = self.model(x, mask)
                logits = outputs["letter_logits"]  # [1, T, 27]

                # Temperature scaling (canonical)
                temp = float(self.calibration.get("temperature", 1.0))
                if temp > 0 and temp != 1.0:
                    logits = logits / temp

                # CTC greedy decode via canonical decoder
                decoder = ClassMappingGreedyCTCDecoder(
                    self.vocab.index_to_letter,
                    blank_index=self.vocab.blank_index,
                )
                decoded = decoder.decode_batch(logits, torch.tensor([T]))

                # Canonical uncertainty (single_letter regime)
                unc_results = single_letter_uncertainty(
                    logits, mask, self.vocab, temperature=temp,
                )
                conf_thr = float(
                    self.calibration.get("confidence_threshold", 0.5)
                )
                margin_thr = float(
                    self.calibration.get("margin_threshold", 0.1)
                )
                if unc_results:
                    unc = unc_results[0]
                    decide_state(unc, conf_thr, margin_thr)
                    state_map = {
                        "letter": "CONFIDENT",
                        "unknown": "LOW_CONFIDENCE",
                        "ambiguous": "LOW_CONFIDENCE",
                    }
                    state = state_map.get(unc.state.value, "LOW_CONFIDENCE")
                    if unc.abstain:
                        state = "ABSTAIN"
                    return PredictionResult(
                        letter=unc.prediction,
                        confidence=unc.confidence,
                        entropy=unc.entropy,
                        margin=unc.margin,
                        top_k=unc.top_k,
                        state=state,
                    )

                # Fallback
                probs = (
                    torch.softmax(logits, dim=-1).mean(dim=1).cpu().numpy()[0]
                )
                idxs = _np.argsort(-probs)[:5]
                top_list = [
                    (self.vocab.index_to_letter[int(i)], float(probs[i]))
                    for i in idxs
                ]
                top1_prob = float(probs[idxs[0]])
                margin = (
                    top1_prob - float(probs[idxs[1]]) if len(idxs) > 1
                    else top1_prob
                )
                entropy = float(
                    -_np.sum(probs * _np.log(probs + 1e-12))
                )
                return PredictionResult(
                    letter=self.vocab.index_to_letter[int(idxs[0])],
                    confidence=top1_prob,
                    entropy=entropy,
                    margin=margin,
                    top_k=top_list,
                    state="LOW_CONFIDENCE"
                    if (top1_prob < conf_thr or margin < margin_thr)
                    else "CONFIDENT",
                    raw_probs=probs.tolist(),
                )


class _IncompatibleCheckpointError(Exception):
    """Raised when a checkpoint does not match the current model definition."""


class InferenceService:
    """Thread-safe checkpoint loading and inference.

    Each checkpoint is loaded ONCE and cached by absolute path. Concurrent
    load_checkpoint calls for the same path return the same ModelHandle.
    Incompatible checkpoints are rejected loudly.
    """

    EXPECTED_TEMPORAL = {
        "d_model": 192, "num_layers": 4, "num_heads": 4,
        "ffn_dim": 768, "conv_kernel": 7, "input_dim": 192,
    }

    def __init__(self):
        self._models: Dict[str, ModelHandle] = {}
        self._lock = threading.Lock()
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def list_checkpoints(self) -> list:
        ckpt_root = Path("runs/letter_decoder")
        if not ckpt_root.exists():
            return []
        return [
            d for d in sorted(
                ckpt_root.iterdir(),
                key=lambda p: p.stat().st_mtime, reverse=True,
            )
            if d.is_dir()
            and (d / "best.ckpt").exists()
            and (d / "summary.json").exists()
        ]

    def get_checkpoint_summary(self, ckpt_dir) -> Dict:
        summary_path = Path(ckpt_dir) / "summary.json"
        if summary_path.exists():
            return json.loads(summary_path.read_text(encoding="utf-8"))
        return {}

    def load_checkpoint(self, ckpt_dir) -> ModelHandle:
        """Load the checkpoint. Cached by absolute path.

        Raises:
            FileNotFoundError: no checkpoint file.
            _IncompatibleCheckpointError: shape/architecture mismatch.
        """
        key = str(Path(ckpt_dir).resolve())
        with self._lock:
            if key in self._models:
                return self._models[key]
            handle = self._build_handle(Path(ckpt_dir))
            self._models[key] = handle
            return handle

    def _build_handle(self, ckpt_dir: Path) -> ModelHandle:
        from alphabet.vocab import LetterVocabulary
        from alphabet.heads import MultiTaskAlphabetModel
        from temporal.encoder import TemporalEncoderConfig

        best_path = (
            ckpt_dir / "best.ckpt"
            if (ckpt_dir / "best.ckpt").exists()
            else ckpt_dir / "latest.ckpt"
        )
        if not best_path.exists():
            raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")
        ck = torch.load(best_path, map_location=self._device, weights_only=False)
        cfg = ck.get("decoder_config") or {}
        tcfg_dict = cfg.get("temporal_config") or {}
        # ── Compatibility check ────────────────────────────────────────
        for k, v in self.EXPECTED_TEMPORAL.items():
            actual = tcfg_dict.get(k)
            if actual is None:
                continue
            if int(actual) != int(v):
                raise _IncompatibleCheckpointError(
                    f"temporal_config.{k}={actual} != expected {v}; "
                    f"checkpoint {best_path} is not compatible with the "
                    f"current Phase 4 architecture."
                )
        tcfg = TemporalEncoderConfig(
            **{
                k: v for k, v in tcfg_dict.items()
                if k in TemporalEncoderConfig.__dataclass_fields__
            }
        )
        vocab = LetterVocabulary()
        # Vocabulary check: must be 27 (blank + A-Z)
        if vocab.size != 27:
            raise _IncompatibleCheckpointError(
                f"vocab size {vocab.size} != 27; checkpoint not compatible"
            )
        aux = cfg.get("auxiliary") or {}
        model = MultiTaskAlphabetModel(
            temporal_config=tcfg,
            letter_vocab_size=vocab.size,
            phone_vocab_size=aux.get("phone_vocab_size"),
            viseme_vocab_size=aux.get("viseme_vocab_size"),
            letter_head_hidden=cfg.get("training", {}).get("letter_head_hidden"),
        )
        # State-dict check
        state_dict = ck.get("model", ck.get("model_state_dict"))
        if state_dict is None:
            raise _IncompatibleCheckpointError(
                f"checkpoint {best_path} has no 'model' / 'model_state_dict'."
            )
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            raise _IncompatibleCheckpointError(
                f"state_dict mismatch: {e}"
            ) from e
        model.to(self._device)
        model.eval()
        cal_path = ckpt_dir / "calibration.json"
        calibration: Dict = {}
        if cal_path.exists():
            calibration = json.loads(cal_path.read_text(encoding="utf-8"))
        return ModelHandle(
            ckpt_dir=ckpt_dir, model=model, vocab=vocab,
            calibration=calibration, config=cfg,
        )

    def unload_all(self) -> None:
        with self._lock:
            self._models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_loaded(self, ckpt_dir) -> bool:
        return str(Path(ckpt_dir).resolve()) in self._models

    def has_loaded_model(self) -> bool:
        """Return whether at least one compatible checkpoint is loaded."""
        with self._lock:
            return bool(self._models)


# IncompatibleCheckpointError is the user-facing alias
IncompatibleCheckpointError = _IncompatibleCheckpointError


_inference_service: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
    return _inference_service