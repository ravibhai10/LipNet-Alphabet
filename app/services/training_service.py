"""
TrainingService — background training using the frozen Phase 2-5 pipeline.

Reuses:
  alphabet/train.run_training()
  alphabet/config.load_decoder_config()
  alphabet/dataset.load_manifest_entries()
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any

from data.validate import DatasetValidator
from alphabet.config import load_decoder_config, DecoderConfig
from alphabet.dataset import load_manifest_entries
from alphabet.train import train_decoder, set_seed, resolve_device

logger = logging.getLogger(__name__)


@dataclass
class PreTrainingCheck:
    passed: bool
    dataset_recordings: int = 0
    dataset_speakers: int = 0
    dataset_splits: Dict[str, int] = field(default_factory=dict)
    validation_errors: int = 0
    data_source_ok: bool = False
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


class TrainingService:

    def __init__(self):
        from app.services.dataset_service import get_dataset_service
        self._ds = get_dataset_service()

    def pre_training_check(self, dataset_root=None) -> PreTrainingCheck:
        errors: list = []
        warnings: list = []

        if dataset_root is None:
            dataset_root = self._ds.DATASET_ROOT

        stats = self._ds.get_stats()
        if stats["recordings"] == 0:
            errors.append("DATASET_EMPTY: No recordings found. "
                         "Record real samples in Dataset Lab first.")
            return PreTrainingCheck(passed=False, errors=errors)

        validator = DatasetValidator(dataset_root)
        report = validator.validate_all()
        if report.has_critical_errors():
            errors.append(f"VALIDATION_FAILED: {report.critical_errors} critical error(s).")
            for r in report.results:
                if not r.passed:
                    warnings.append(f"  [{r.check_name}] {r.message}")
                    if r.details:
                        warnings.extend(f"    {d}" for d in r.details[:3])

        manifest_path = dataset_root / "manifest.jsonl"
        split_counts: Dict[str, int] = {}
        if not manifest_path.exists():
            errors.append("MANIFEST_MISSING: Run 'Build Manifest' in Dataset Lab first.")
        else:
            try:
                entries = load_manifest_entries(dataset_root, require_split=True)
                for e in entries:
                    split_counts[e.split] = split_counts.get(e.split, 0) + 1
                if "train" not in split_counts:
                    errors.append("SPLIT_NO_TRAIN: No training split found.")
                if "val" not in split_counts:
                    warnings.append("SPLIT_NO_VAL: No validation split.")
            except FileNotFoundError:
                errors.append("MANIFEST_MISSING: Run 'Build Manifest' first.")
            except Exception as e:
                errors.append(f"MANIFEST_LOAD_ERROR: {e}")

        data_src_check = next(
            (r for r in report.results if r.check_name == "data_source"), None
        )
        data_source_ok = data_src_check is None or data_src_check.passed
        if not data_source_ok:
            errors.append(f"DATA_SOURCE_GATE_FAILED: {data_src_check.message}")

        # ── Real-dataset gate (§15) ─────────────────────────────────────
        # Production training is forbidden unless:
        #   - data_source is real_camera (synthetic never)
        #   - landmark_source is mediapipe
        #   - all manifest entries have data_source=real_camera
        real_gate = self._check_real_dataset_gate(dataset_root)
        if not real_gate["passed"]:
            errors.extend(real_gate["errors"])

        passed = len(errors) == 0
        return PreTrainingCheck(
            passed=passed, dataset_recordings=stats["recordings"],
            dataset_speakers=stats["speakers"], dataset_splits=split_counts,
            validation_errors=report.critical_errors, data_source_ok=data_source_ok,
            errors=errors, warnings=warnings,
        )

    def _check_real_dataset_gate(self, dataset_root: Path) -> Dict[str, Any]:
        """Refuse synthetic / non-mediapipe data for production training.

        Hard requirements:
            - manifest.jsonl exists
            - every entry's data_source == real_camera
            - every entry's landmark_source == mediapipe
        """
        errors = []
        manifest_path = dataset_root / "manifest.jsonl"
        if not manifest_path.exists():
            errors.append(
                "MANIFEST_MISSING: cannot verify real-dataset gate without manifest."
            )
            return {"passed": False, "errors": errors}

        try:
            from data.schema import ManifestEntry
        except Exception as exc:
            errors.append(f"SCHEMA_IMPORT_FAILED: {exc}")
            return {"passed": False, "errors": errors}

        n_total = 0
        n_synth = 0
        n_non_mp = 0
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = ManifestEntry.from_jsonl(line)
                n_total += 1
                ds = getattr(e, "data_source", None)
                if ds is None or ds != "real_camera":
                    n_synth += 1
                ls = getattr(e, "landmark_source", None)
                if ls is None or ls != "mediapipe":
                    n_non_mp += 1

        if n_total == 0:
            errors.append("MANIFEST_EMPTY: manifest has zero entries.")
        if n_synth > 0:
            errors.append(
                f"REAL_DATASET_GATE_FAILED: {n_synth}/{n_total} entries have "
                "data_source != 'real_camera'. Synthetic data is forbidden in "
                "production training."
            )
        if n_non_mp > 0:
            errors.append(
                f"LANDMARK_SOURCE_GATE_FAILED: {n_non_mp}/{n_total} entries have "
                "landmark_source != 'mediapipe'."
            )
        return {"passed": len(errors) == 0, "errors": errors}

    def get_config(self, yaml_path="configs/alphabet_dataset.yaml") -> DecoderConfig:
        return load_decoder_config(yaml_path)

    def train(self, job_id, config, max_epochs=None, batch_size=None,
              lr=None, device=None, phase2_cfg=None) -> Dict[str, Any]:
        """Execute training in background thread. Called by JobManager._worker."""
        from config.loader import load_phase2_config
        self._set_progress(job_id, state="INITIALIZING", epoch=0,
                          note="Loading config...")
        if max_epochs is not None:
            config.training.epochs = max_epochs
        if batch_size is not None:
            config.training.batch_size = batch_size
        if lr is not None:
            config.training.lr = lr
        set_seed(config.training.seed)
        dev = resolve_device(device)
        if phase2_cfg is None:
            phase2_cfg = load_phase2_config("configs/alphabet_dataset.yaml")
        self._set_progress(job_id, state="LOADING", epoch=0,
                          note=f"Device: {dev}, loading manifest...")
        entries = load_manifest_entries(config.dataset_root, require_split=True)
        split_counts: Dict[str, int] = {}
        for e in entries:
            split_counts[e.split] = split_counts.get(e.split, 0) + 1
        self._set_progress(job_id, state="TRAINING", epoch=0,
                          note=f"Train: {split_counts.get('train', 0)}, "
                               f"Val: {split_counts.get('val', 0)}, "
                               f"Test: {split_counts.get('test', 0)}")
        try:
            test_eval = train_decoder(
                dec_cfg=config, phase2_cfg=phase2_cfg,
                max_epochs=config.training.epochs,
                device=str(dev), resume=False,
                run_test=split_counts.get("test", 0) > 0,
            )
        except TypeError:
            test_eval = train_decoder(
                dec_cfg=config, phase2_cfg=phase2_cfg,
                max_epochs=config.training.epochs,
                device=str(dev), resume=False,
            )
        summary = {
            "epochs_run": config.training.epochs,
            "test_eval": test_eval,
            "device": str(dev),
        }
        self._set_progress(job_id, state="COMPLETED",
                          epoch=config.training.epochs,
                          note="Training finished.")
        return summary

    def _set_progress(self, job_id, **kwargs):
        from app.services.job_manager import get_job_manager
        jm = get_job_manager()
        job = jm.get(job_id)
        if job:
            job.progress.update(kwargs)

    def list_checkpoints(self) -> list:
        ckpt_root = Path("runs/letter_decoder")
        if not ckpt_root.exists():
            return []
        return sorted(
            [d for d in ckpt_root.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime, reverse=True,
        )

    def get_checkpoint_info(self, ckpt_dir: Path) -> Dict[str, Any]:
        summary_path = ckpt_dir / "summary.json"
        if summary_path.exists():
            return json.loads(summary_path.read_text(encoding="utf-8"))
        return {}


_training_service: Optional[TrainingService] = None


def get_training_service() -> TrainingService:
    global _training_service
    if _training_service is None:
        _training_service = TrainingService()
    return _training_service

