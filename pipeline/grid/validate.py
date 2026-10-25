"""Reconstruction / validity checker for an already-extracted GRID corpus."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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

    def run(self) -> GridValidationReport:
        report = GridValidationReport(talk_id="<all>")
        for entry in self.entries:
            rep = self._check(entry)
            report.record(rep)
        return report

    def _check(self, entry: GridManifestEntry) -> GridManifestEntry:
        issues: list[str] = []
        workdir = Path(self.grid_root)

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
                entry.validation_status = ValidationStatus.MISSING_AUDIO
                entry.errors.append("audio not found on disk")

        # Alignment
        if entry.alignment_path:
            apath = workdir / entry.alignment_path
            if not apath.exists():
                entry.alignment_status = AlignmentStatus.MALFORMED
                entry.errors.append("alignment not found on disk")

        if issues:
            entry.errors.extend(issues)
        elif entry.validation_status != ValidationStatus.OK:
            entry.validation_status = ValidationStatus.OK
            entry.errors = []
        return entry
