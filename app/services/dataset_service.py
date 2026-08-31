"""
DatasetService — recording, validation, and statistics for the production dataset.
Never touches synthetic_dataset/.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

from data.ingestion import VideoIngestor, IngestionConfig, get_video_info, validate_video
from data.manifest import ManifestManager, ManifestConfig, create_splits, save_split_info
from data.validate import DatasetValidator, ValidationReport
from data.schema import RecordingMetadata, Letter, DATA_SOURCE_REAL_CAMERA

logger = logging.getLogger(__name__)


@dataclass
class RecordingResult:
    success: bool
    recording_id: Optional[str] = None
    recording_path: Optional[Path] = None
    metadata: Optional[RecordingMetadata] = None
    error: Optional[str] = None
    validation_errors: List[str] = field(default_factory=list)


class DatasetService:

    def __init__(self, dataset_root: Optional[Path] = None):
        if dataset_root is None:
            dataset_root = Path(__file__).resolve().parents[2] / "alphabet_dataset"
        self.DATASET_ROOT = Path(dataset_root)
        self.RECORDINGS_ROOT = self.DATASET_ROOT / "recordings"

    def record(self, letter, speaker_id, session_id, duration_seconds=3.0,
               fps=30, width=1280, height=720, device_index=0, show_preview=True,
               lighting_tag=None, accent_tag=None, notes=None) -> RecordingResult:
        letter = letter.upper().strip()
        if not Letter.is_valid(letter):
            return RecordingResult(success=False, error=f"Invalid letter '{letter}'. Must be A-Z.")
        if not speaker_id or not speaker_id.strip():
            return RecordingResult(success=False, error="speaker_id is required.")
        if not session_id or not session_id.strip():
            return RecordingResult(success=False, error="session_id is required.")
        if duration_seconds <= 0:
            return RecordingResult(success=False, error="duration must be > 0.")
        speaker_id = speaker_id.strip()
        session_id = session_id.strip()
        config = IngestionConfig(
            dataset_root=self.DATASET_ROOT, fps=float(fps),
            max_sequence_length=76, min_sequence_length=15,
            input_resolution=(width, height), codec="mp4v",
            preprocessing_version="1.0", config_version="1.0",
        )
        try:
            ingestor = VideoIngestor(config)
            metadata = ingestor.record_webcam(
                letter=letter, speaker_id=speaker_id, session_id=session_id,
                duration_seconds=duration_seconds,
                lighting_tag=lighting_tag, accent_tag=accent_tag, notes=notes,
                device_index=device_index, show_preview=show_preview,
            )
            recording_path = (self.RECORDINGS_ROOT / letter / speaker_id
                             / session_id / metadata.recording_id)
            val_errors = self._validate_recording_path(recording_path)
            return RecordingResult(success=True, recording_id=metadata.recording_id,
                                  recording_path=recording_path, metadata=metadata,
                                  validation_errors=val_errors)
        except RuntimeError as e:
            return RecordingResult(success=False, error=f"Recording failed: {e}")
        except Exception as e:
            logger.exception("Unexpected recording error")
            return RecordingResult(success=False, error=str(e))

    def _validate_recording_path(self, rec_dir: Path) -> List[str]:
        errors: List[str] = []
        video_path = rec_dir / "video.mp4"
        meta_path = rec_dir / "metadata.json"
        if not video_path.exists():
            errors.append("VIDEO_NOT_FOUND")
        elif not video_path.is_file():
            errors.append("VIDEO_NOT_A_FILE")
        else:
            valid, msg = validate_video(video_path)
            if not valid:
                errors.append(f"VIDEO_INVALID: {msg}")
            info = get_video_info(video_path)
            if info.frame_count <= 0:
                errors.append("VIDEO_EMPTY_OR_UNREADABLE")
            if info.fps <= 0:
                errors.append(f"VIDEO_INVALID_FPS: {info.fps}")
        if not meta_path.exists():
            errors.append("METADATA_NOT_FOUND")
        else:
            try:
                meta = RecordingMetadata.load(meta_path)
                if not Letter.is_valid(meta.letter):
                    errors.append(f"INVALID_LETTER_IN_METADATA: {meta.letter}")
                if not meta.speaker_id:
                    errors.append("MISSING_SPEAKER_ID")
                if not meta.session_id:
                    errors.append("MISSING_SESSION_ID")
                if not meta.recording_id:
                    errors.append("MISSING_RECORDING_ID")
                if meta.data_source != DATA_SOURCE_REAL_CAMERA:
                    errors.append(f"WRONG_DATA_SOURCE: {meta.data_source}")
            except Exception as e:
                errors.append(f"METADATA_LOAD_ERROR: {e}")
        return errors
    def get_stats(self) -> Dict[str, Any]:
        if not self.RECORDINGS_ROOT.exists():
            return {"recordings": 0, "speakers": 0, "sessions": 0,
                    "letters": 0, "letter_counts": {},
                    "missing_letters": [l.value for l in Letter],
                    "min_count": 0, "max_count": 0}
        speakers: set = set(); sessions: set = set()
        letter_counts: Dict[str, int] = {}; total = 0
        for letter_dir in sorted(self.RECORDINGS_ROOT.iterdir()):
            if not letter_dir.is_dir():
                continue
            lname = letter_dir.name.upper()
            if not Letter.is_valid(lname):
                continue
            count = 0
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
                            count += 1; total += 1
            if count > 0:
                letter_counts[lname] = count
        all_letters = {l.value for l in Letter}
        missing = sorted(all_letters - set(letter_counts.keys()))
        counts = list(letter_counts.values())
        return {"recordings": total, "speakers": len(speakers),
                "sessions": len(sessions), "letters": len(letter_counts),
                "letter_counts": letter_counts, "missing_letters": missing,
                "min_count": min(counts) if counts else 0,
                "max_count": max(counts) if counts else 0}

    def validate(self) -> ValidationReport:
        validator = DatasetValidator(self.DATASET_ROOT)
        return validator.validate_all()

    def build_manifest(self) -> int:
        manifest_path = self.DATASET_ROOT / "manifest.jsonl"
        cfg = ManifestConfig(dataset_root=self.DATASET_ROOT, manifest_path=manifest_path)
        manager = ManifestManager(cfg)
        entries = manager.build_manifest()
        with open(manifest_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(e.to_jsonl() + "\n")
        return len(entries)

    def create_splits(self) -> Dict[str, int]:
        manifest_path = self.DATASET_ROOT / "manifest.jsonl"
        split_info_path = self.DATASET_ROOT / "split_info.json"
        cfg = ManifestConfig(dataset_root=self.DATASET_ROOT, manifest_path=manifest_path,
                           split_info_path=split_info_path)
        manager = ManifestManager(cfg)
        entries = manager.build_manifest()
        manifest_path.write_text("\n".join(e.to_jsonl() for e in entries) + "\n",
                                encoding="utf-8")
        split_map = create_splits(manifest_path, train_ratio=0.8, val_ratio=0.1,
                                 test_ratio=0.1, min_speakers_per_split=1)
        save_split_info(split_map, manifest_path, split_info_path, random_seed=42)
        counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
        for e in entries:
            if split_map.get(e.recording_id):
                counts[split_map[e.recording_id]] += 1
        return counts

    def has_real_data(self) -> bool:
        return self.get_stats()["recordings"] > 0


_dataset_service: Optional[DatasetService] = None


def get_dataset_service() -> DatasetService:
    global _dataset_service
    if _dataset_service is None:
        _dataset_service = DatasetService()
    return _dataset_service

