"""Schema + validation for GRID manifest entries."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from pipeline.grid.manifest import AlignmentFile, AlignmentToken


class ValidationStatus(str, Enum):
    OK = "ok"
    MISSING_VIDEO = "missing_video"
    MISSING_AUDIO = "missing_audio"
    MISSING_ALIGNMENT = "missing_alignment"
    CORRUPT_MEDIA = "corrupt_media"


class AlignmentStatus(str, Enum):
    OK = "ok"
    MISSING = "missing"
    MALFORMED = "malformed"


@dataclass(frozen=True)
class GridManifestEntry:
    """One screen of GRID audio/video + alignment."""

    talk_id: str
    speaker_id: str
    utterance_id: str
    video_path: str
    audio_path: str
    alignment_path: str
    frame_count: int = 0
    fps: float = 0.0
    transcript: str = ""
    alignment_status: AlignmentStatus = AlignmentStatus.OK
    validation_status: ValidationStatus = ValidationStatus.OK
    value: str = ""
    alignment: Optional[AlignmentFile] = None
    errors: list[str] = field(default_factory=list)

    @property
    def is_ok(self) -> bool:
        return self.validation_status == ValidationStatus.OK

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(obj: dict) -> "GridManifestEntry":
        return GridManifestEntry(**obj)

    def to_json(self, ensure_ascii: bool = False, indent: Optional[int] = None) -> str:
        import json
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, indent=indent)

    @staticmethod
    def from_json(text: str) -> "GridManifestEntry":
        import json
        return GridManifestEntry.from_dict(json.loads(text))


@dataclass
class GridValidationReport:
    """Validation summary for a whole corpus extraction."""

    talk_id: str
    entries: list[GridManifestEntry] = field(default_factory=list)
    critical_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.entries)

    @property
    def ok(self) -> int:
        return sum(1 for e in self.entries if e.is_ok)

    @property
    def failed(self) -> int:
        return self.total - self.ok

    def record(self, entry: GridManifestEntry) -> None:
        self.entries.append(entry)
        if not entry.is_ok:
            self.critical_errors.append(entry.utterance_id)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def has_critical_errors(self) -> bool:
        return len(self.critical_errors) > 0

    def summarize(self) -> str:
        return (
            f"talk={self.talk_id} total={self.total} ok={self.ok} failed={self.failed} "
            f"critical_errors={len(self.critical_errors)} warnings={len(self.warnings)}"
        )
