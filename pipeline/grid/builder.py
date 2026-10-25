"""Manifest builder + per-file validator for the GRID corpus."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pipeline.grid.manifest import GRIDAlignmentParser, AlignmentToken
from pipeline.grid.schema import (
    AlignmentStatus,
    GridManifestEntry,
    GridValidationReport,
    ValidationStatus,
)

logger = logging.getLogger(__name__)

VIDEO_EXT = frozenset({".mp4", ".avi", ".mov", ".mkv"})
AUDIO_EXT = frozenset({".wav", ".flac", ".mp3", ".aac", ".m4a"})
ALIGNMENT_EXT = frozenset({".txt", ".align", ".csv", ".trn"})


def _infer_utterance_id(talk_id: str, video_path: Path) -> str:
    name = video_path.stem
    if "_U" in name:
        return name.replace(".", "_")
    return f"{talk_id}_U{name.zfill(4)}"


def _find_media(talk_dir: Path, video_ext=VIDEO_EXT, audio_ext=AUDIO_EXT, alignment_ext=ALIGNMENT_EXT):
    video = audio = alignment = None
    for ext in video_ext:
        for p in talk_dir.rglob(f"*{ext}"):
            if p.is_file():
                video = p
                break
        if video is not None:
            break
    for ext in audio_ext:
        for p in talk_dir.rglob(f"*{ext}"):
            if p.is_file():
                audio = p
                break
        if audio is not None:
            break
    for ext in alignment_ext:
        for p in talk_dir.rglob(f"*{ext}"):
            if p.is_file():
                alignment = p
                break
        if alignment is not None:
            break
    return video, audio, alignment


def _guess_fps_and_frames(video_path: Path) -> tuple[float, int]:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0, 0
    try:
        return float(cap.get(cv2.CAP_PROP_FPS)) or 0.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def _best_alignment_status(parser) -> AlignmentStatus:
    if parser is not None and parser.valid() and parser.tokens:
        return AlignmentStatus.OK
    if parser is not None:
        return AlignmentStatus.MALFORMED
    return AlignmentStatus.MISSING


@dataclass
class GridManifestBuilder:
    """Build a list of GridManifestEntry from an extracted GRID tree."""

    grid_root: Path
    video_ext: frozenset[str] = frozenset(VIDEO_EXT)
    audio_ext: frozenset[str] = frozenset(AUDIO_EXT)
    alignment_ext: frozenset[str] = frozenset(ALIGNMENT_EXT)

    def build(self) -> list[GridManifestEntry]:
        root = Path(self.grid_root)
        entries: list[GridManifestEntry] = []
        for talk_dir in sorted(root.iterdir()):
            if not talk_dir.is_dir():
                continue
            talk_id = talk_dir.name.strip()
            if not talk_id:
                continue
            video, audio, alignment = _find_media(talk_dir, self.video_ext, self.audio_ext, self.alignment_ext)
            entries.extend(self._process_speaker(talk_id, talk_dir, video, audio, alignment))
        return entries

    def _process_speaker(self, talk_id: str, talk_dir: Path, video: Optional[Path], audio: Optional[Path], alignment: Optional[Path]) -> list[GridManifestEntry]:
        entries: list[GridManifestEntry] = []
        alignment_path = Path(alignment) if alignment is not None else None
        parser = None
        if alignment_path is not None:
            parser = GRIDAlignmentParser(alignment_path)
            parser.parse()

        if video is None:
            entries.append(GridManifestEntry(talk_id=talk_id, speaker_id=f"grid_{talk_id}", utterance_id=_infer_utterance_id(talk_id, Path('.')), video_path='', audio_path='', alignment_path='', frame_count=0, fps=0.0, validation_status=ValidationStatus.MISSING_VIDEO, alignment_status=AlignmentStatus.MISSING, errors=['no video found']))
            return entries

        fps, frames = _guess_fps_and_frames(video)
        entry = GridManifestEntry(talk_id=talk_id, speaker_id=f"grid_{talk_id}", utterance_id=_infer_utterance_id(talk_id, video), video_path=str(video.relative_to(talk_dir.parent)), audio_path=(str(audio.relative_to(talk_dir.parent)) if audio is not None else ''), alignment_path=(str(alignment.relative_to(talk_dir.parent)) if alignment is not None else ''), frame_count=frames, fps=fps, alignment=parser, alignment_status=_best_alignment_status(parser), validation_status=ValidationStatus.OK)
        missing = []
        if not video.exists(): missing.append("video")
        if audio is not None and not audio.exists(): missing.append("audio")
        if alignment is not None and not alignment.exists(): missing.append("alignment")
        if missing:
            if "video" in missing:
                entry.validation_status = ValidationStatus.MISSING_VIDEO
            elif "alignment" in missing:
                entry.validation_status = ValidationStatus.MISSING_ALIGNMENT
            else:
                entry.validation_status = ValidationStatus.MISSING_AUDIO
            entry.errors.append("missing media: " + ",".join(missing))
        entries.append(entry)
        return entries

    def _parse_csv_alignment(self, path: Path) -> dict[str, list[AlignmentToken]]:
        result: dict[str, list[AlignmentToken]] = {}
        try:
            with path.open(newline="", encoding="utf-8", errors="replace") as f:
                for row in csv.reader(f):
                    if len(row) >= 4:
                        rid = row[0].strip()
                        result.setdefault(rid, []).append(AlignmentToken(word=row[1].strip(), start_ms=int(row[2]), end_ms=int(row[3])))
        except Exception as exc:
            logger.warning("could not parse CSV alignment %s: %s", path, exc)
        return result
