"""Base dataset adapter schema and interface for visual speech recognition datasets.

All dataset adapters normalize raw annotations into a unified, task-aware
schema while strictly preventing speaker leakage, data corruption, and
invalid label assignments.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Optional


@dataclass
class DatasetSample:
    sample_id: str
    dataset: str
    video_path: str
    start_frame: int
    end_frame: int
    speaker_id: str
    task_type: Literal["isolated_word", "phrase", "continuous_sentence"]
    transcript: str
    word_label: Optional[str]
    split: Literal["train", "valid", "test"]
    fps: float
    source_checksum: str
    validation_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseDatasetAdapter(ABC):
    def __init__(self, dataset_name: str, root_dir: Path):
        self.dataset_name = dataset_name
        self.root_dir = Path(root_dir).resolve()

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether genuine source files exist locally."""
        pass

    @abstractmethod
    def validate_integrity(self) -> dict[str, Any]:
        """Perform checksum and structural validation on local assets."""
        pass

    @abstractmethod
    def generate_manifest(self) -> list[DatasetSample]:
        """Generate standardized, speaker-disjoint dataset samples."""
        pass

    @staticmethod
    def compute_sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
