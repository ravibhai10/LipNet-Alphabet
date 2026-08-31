"""Tests for InferenceService: model loading, validation, prediction.

Tests:
    - model loaded once
    - incompatible checkpoint rejected
    - no model -> informative error
    - valid prediction
    - low confidence
    - abstention
    - invalid sequence
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch

from app.services.inference_service import (
    InferenceService, ModelHandle, PredictionResult,
    IncompatibleCheckpointError, get_inference_service,
)


class TestInferenceServiceSingleton:
    """Singleton management."""

    def test_get_inference_service_returns_same_instance(self):
        s1 = get_inference_service()
        s2 = get_inference_service()
        assert s1 is s2

    def test_inference_service_has_device_attribute(self):
        svc = InferenceService()
        assert svc._device is not None
        assert svc._device.type in ("cpu", "cuda")

    def test_inference_service_models_dict_initially_empty(self):
        svc = InferenceService()
        assert len(svc._models) == 0


class TestInferenceServiceListCheckpoints:
    """Checkpoint discovery."""

    def test_list_checkpoints_empty_when_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        svc = InferenceService()
        assert svc.list_checkpoints() == []

    def test_list_checkpoints_ignores_dirs_without_best_ckpt(self, tmp_path, monkeypatch):
        ckpt_root = tmp_path / "runs" / "letter_decoder"
        ckpt_root.mkdir(parents=True)
        # Create a dir without best.ckpt
        bad = ckpt_root / "incomplete"
        bad.mkdir()
        monkeypatch.chdir(tmp_path)
        svc = InferenceService()
        assert svc.list_checkpoints() == []

    def test_list_checkpoints_returns_dirs_with_best_and_summary(self, tmp_path, monkeypatch):
        ckpt_root = tmp_path / "runs" / "letter_decoder" / "exp1"
        ckpt_root.mkdir(parents=True)
        (ckpt_root / "best.ckpt").write_bytes(b"fake")
        (ckpt_root / "summary.json").write_text("{}", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        svc = InferenceService()
        ckpts = svc.list_checkpoints()
        assert len(ckpts) == 1
        assert ckpts[0].name == "exp1"


class TestInferenceServiceCheckpointSummary:
    """Summary file reading."""

    def test_get_checkpoint_summary_returns_empty_dict_when_no_summary(self, tmp_path):
        svc = InferenceService()
        summary = svc.get_checkpoint_summary(tmp_path)
        assert summary == {}

    def test_get_checkpoint_summary_reads_json(self, tmp_path):
        (tmp_path / "summary.json").write_text(
            json.dumps({"best_val_accuracy": 0.95, "epoch": 42}),
            encoding="utf-8",
        )
        svc = InferenceService()
        summary = svc.get_checkpoint_summary(tmp_path)
        assert summary["best_val_accuracy"] == 0.95
        assert summary["epoch"] == 42


class TestInferenceServiceCheckpointLoading:
    """Loading and caching behavior."""

    def test_load_checkpoint_missing_file_raises(self, tmp_path):
        svc = InferenceService()
        with pytest.raises(FileNotFoundError, match="No checkpoint"):
            svc.load_checkpoint(tmp_path)

    def test_load_checkpoint_caches_handle(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        # Create a fake checkpoint that we can load with mocked torch
        (ckpt_dir / "best.ckpt").touch()
        (ckpt_dir / "summary.json").write_text("{}", encoding="utf-8")

        # Build a minimal valid checkpoint
        from alphabet.heads import MultiTaskAlphabetModel
        from temporal.encoder import TemporalEncoderConfig
        from alphabet.vocab import LetterVocabulary

        tcfg = TemporalEncoderConfig()
        vocab = LetterVocabulary()
        model = MultiTaskAlphabetModel(
            temporal_config=tcfg,
            letter_vocab_size=vocab.size,
        )
        state = model.state_dict()
        ckpt = {
            "model": state,
            "decoder_config": {
                "temporal_config": {
                    "input_dim": 192, "d_model": 192,
                    "num_layers": 4, "num_heads": 4,
                    "ffn_dim": 768, "conv_kernel": 7,
                },
            },
        }
        torch.save(ckpt, ckpt_dir / "best.ckpt")

        svc = InferenceService()
        h1 = svc.load_checkpoint(ckpt_dir)
        h2 = svc.load_checkpoint(ckpt_dir)  # cached
        assert h1 is h2

    def test_load_checkpoint_rejects_incompatible_temporal_config(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        ckpt = {
            "model": {},
            "decoder_config": {
                "temporal_config": {
                    "input_dim": 999,  # WRONG
                    "d_model": 192, "num_layers": 4, "num_heads": 4,
                    "ffn_dim": 768, "conv_kernel": 7,
                },
            },
        }
        torch.save(ckpt, ckpt_dir / "best.ckpt")
        svc = InferenceService()
        with pytest.raises(IncompatibleCheckpointError, match="temporal_config"):
            svc.load_checkpoint(ckpt_dir)

    def test_load_checkpoint_rejects_incompatible_d_model(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        ckpt = {
            "model": {},
            "decoder_config": {
                "temporal_config": {
                    "input_dim": 192, "d_model": 256,  # WRONG
                    "num_layers": 4, "num_heads": 4,
                    "ffn_dim": 768, "conv_kernel": 7,
                },
            },
        }
        torch.save(ckpt, ckpt_dir / "best.ckpt")
        svc = InferenceService()
        with pytest.raises(IncompatibleCheckpointError, match="d_model"):
            svc.load_checkpoint(ckpt_dir)

    def test_load_checkpoint_rejects_state_dict_mismatch(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        ckpt = {
            "model": {"this_is_not_a_real_key.weight": torch.zeros(1)},
            "decoder_config": {
                "temporal_config": {
                    "input_dim": 192, "d_model": 192,
                    "num_layers": 4, "num_heads": 4,
                    "ffn_dim": 768, "conv_kernel": 7,
                },
            },
        }
        torch.save(ckpt, ckpt_dir / "best.ckpt")
        svc = InferenceService()
        with pytest.raises(IncompatibleCheckpointError, match="state_dict"):
            svc.load_checkpoint(ckpt_dir)

    def test_load_checkpoint_rejects_missing_state_dict(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        ckpt = {
            "decoder_config": {
                "temporal_config": {
                    "input_dim": 192, "d_model": 192,
                    "num_layers": 4, "num_heads": 4,
                    "ffn_dim": 768, "conv_kernel": 7,
                },
            },
        }
        torch.save(ckpt, ckpt_dir / "best.ckpt")
        svc = InferenceService()
        with pytest.raises(IncompatibleCheckpointError, match="no 'model'"):
            svc.load_checkpoint(ckpt_dir)

    def test_load_checkpoint_uses_latest_when_no_best(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        from alphabet.heads import MultiTaskAlphabetModel
        from temporal.encoder import TemporalEncoderConfig
        from alphabet.vocab import LetterVocabulary
        tcfg = TemporalEncoderConfig()
        vocab = LetterVocabulary()
        model = MultiTaskAlphabetModel(
            temporal_config=tcfg, letter_vocab_size=vocab.size,
        )
        ckpt = {
            "model": model.state_dict(),
            "decoder_config": {
                "temporal_config": {
                    "input_dim": 192, "d_model": 192,
                    "num_layers": 4, "num_heads": 4,
                    "ffn_dim": 768, "conv_kernel": 7,
                },
            },
        }
        torch.save(ckpt, ckpt_dir / "latest.ckpt")
        svc = InferenceService()
        handle = svc.load_checkpoint(ckpt_dir)
        assert handle is not None


class TestInferenceServiceCalibration:
    """Calibration.json loading."""

    def test_calibration_loaded_when_present(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        from alphabet.heads import MultiTaskAlphabetModel
        from temporal.encoder import TemporalEncoderConfig
        from alphabet.vocab import LetterVocabulary
        tcfg = TemporalEncoderConfig()
        vocab = LetterVocabulary()
        model = MultiTaskAlphabetModel(
            temporal_config=tcfg, letter_vocab_size=vocab.size,
        )
        ckpt = {
            "model": model.state_dict(),
            "decoder_config": {
                "temporal_config": {
                    "input_dim": 192, "d_model": 192,
                    "num_layers": 4, "num_heads": 4,
                    "ffn_dim": 768, "conv_kernel": 7,
                },
            },
        }
        torch.save(ckpt, ckpt_dir / "best.ckpt")
        (ckpt_dir / "calibration.json").write_text(
            json.dumps({
                "temperature": 1.5,
                "confidence_threshold": 0.6,
                "margin_threshold": 0.1,
            }),
            encoding="utf-8",
        )
        svc = InferenceService()
        handle = svc.load_checkpoint(ckpt_dir)
        assert handle.calibration["temperature"] == 1.5
        assert handle.calibration["confidence_threshold"] == 0.6


class TestInferenceServicePredict:
    """Predict path with a real (small) model."""

    def _build_real_handle(self):
        from alphabet.heads import MultiTaskAlphabetModel
        from alphabet.vocab import LetterVocabulary
        from temporal.encoder import TemporalEncoderConfig
        tcfg = TemporalEncoderConfig()
        vocab = LetterVocabulary()
        model = MultiTaskAlphabetModel(
            temporal_config=tcfg, letter_vocab_size=vocab.size,
        )
        model.eval()
        return ModelHandle(
            ckpt_dir=Path("/tmp/fake"),
            model=model,
            vocab=vocab,
            calibration={},
            config={},
        )

    def test_predict_rejects_empty_input(self):
        handle = self._build_real_handle()
        with pytest.raises(ValueError, match="non-empty"):
            handle.predict(np.zeros((0, 192), dtype=np.float32))

    def test_predict_rejects_wrong_dim(self):
        handle = self._build_real_handle()
        with pytest.raises(ValueError, match="non-empty"):
            handle.predict(np.zeros(192, dtype=np.float32))

    def test_predict_returns_letter(self):
        handle = self._build_real_handle()
        fused = np.random.randn(8, 192).astype(np.float32)
        result = handle.predict(fused)
        assert isinstance(result, PredictionResult)
        # Either a letter or empty string (low confidence)
        assert result.letter == "" or (len(result.letter) == 1 and result.letter.isalpha())
        assert 0.0 <= result.confidence <= 1.0
        assert result.state in ("CONFIDENT", "LOW_CONFIDENCE", "ABSTAIN")

    def test_predict_top_k_structure(self):
        handle = self._build_real_handle()
        fused = np.random.randn(10, 192).astype(np.float32)
        result = handle.predict(fused)
        assert len(result.top_k) >= 1
        for letter, prob in result.top_k:
            assert isinstance(letter, str)
            assert 0.0 <= prob <= 1.0

    def test_predict_with_low_confidence_calibration(self):
        handle = self._build_real_handle()
        # Set very high thresholds so the model abstains
        handle.calibration = {
            "confidence_threshold": 0.9999,
            "margin_threshold": 0.9999,
        }
        fused = np.random.randn(8, 192).astype(np.float32)
        result = handle.predict(fused)
        # Should abstain or be low confidence
        assert result.state in ("ABSTAIN", "LOW_CONFIDENCE")

    def test_predict_with_temperature(self):
        handle = self._build_real_handle()
        handle.calibration = {"temperature": 2.0}
        fused = np.random.randn(8, 192).astype(np.float32)
        result = handle.predict(fused)
        assert isinstance(result, PredictionResult)


class TestInferenceServiceUnload:
    """Resource management."""

    def test_unload_all_clears_cache(self):
        svc = InferenceService()
        svc._models["/tmp/fake"] = MagicMock()
        assert len(svc._models) == 1
        svc.unload_all()
        assert len(svc._models) == 0

    def test_is_loaded_reflects_cache(self, tmp_path):
        svc = InferenceService()
        # Use a real path so Path.resolve() returns the expected string
        real_path = tmp_path / "fake"
        real_path.mkdir()
        resolved = str(real_path.resolve())
        assert not svc.is_loaded(real_path)
        svc._models[resolved] = MagicMock()
        assert svc.is_loaded(real_path)