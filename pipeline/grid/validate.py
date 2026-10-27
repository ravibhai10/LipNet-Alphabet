"""Reconstruction / validity checker for an already-extracted GRID corpus."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

from pipeline.grid.builder import _guess_fps_and_frames
from pipeline.grid.schema import (
    GridManifestEntry,
    GridValidationReport,
    ValidationStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class GridValidator:
    """Re-validates manifest entries against the on-disk corpus."""

    grid_root: Path
    entries: list[GridManifestEntry] = field(default_factory=list)

    def _mark(self, entry: GridManifestEntry, status: ValidationStatus, msg: str) -> GridManifestEntry:
        errors = list(entry.errors)
        errors.append(msg)
        return replace(entry, validation_status=status, errors=errors)

    def run(self) -> GridValidationReport:
        report = GridValidationReport(talk_id="<all>")
        for entry in self.entries:
            rep = self._check(entry)
            report.record(rep)
        return report

    def _check(self, entry: GridManifestEntry) -> GridManifestEntry:
        issues: list[str] = list(entry.errors)
        workdir = Path(self.grid_root)
        validation_status = entry.validation_status
        alignment_status = entry.alignment_status

        # Re-open video to confirm readability
        if entry.video_path:
            vpath = workdir / entry.video_path
            if not vpath.exists():
                return self._mark(entry, ValidationStatus.MISSING_VIDEO, "video not found on disk")
            fps, frames = _guess_fps_and_frames(vpath)
            if fps <= 0 or frames <= 0:
                issues.append(f"unreadable video: fps={fps} frames={frames}")

        # Audio
        if entry.audio_path:
            apath = workdir / entry.audio_path
            if not apath.exists():
                validation_status = ValidationStatus.MISSING_AUDIO
                issues.append("audio not found on disk")

        # Alignment
        if entry.alignment_path:
            alpath = workdir / entry.alignment_path
            if not alpath.exists():
                alignment_status = AlignmentStatus.MALFORMED
                issues.append("alignment not found on disk")

        if issues:
            return replace(entry, validation_status=validation_status, alignment_status=alignment_status, errors=issues)
        elif validation_status != ValidationStatus.OK:
            return replace(entry, validation_status=ValidationStatus.OK, alignment_status=AlignmentStatus.OK, errors=[])
        return entry
