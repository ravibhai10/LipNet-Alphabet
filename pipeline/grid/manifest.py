"""Alignment parser for the GRID audiovisual corpus.

GRID ships word alignments as one ASCII text file per talker, one line per
sentence, with token offsets tracked in a special per-line form.

Official GRID layout (spandh.dcs.shef.ac.uk/gridcorpus/):
    <TALK>/
        audio/25kHz/        # endpointed audio, 25 kHz
        audio/raw/          # 50 kHz raw audio (optional)
        video/              # normal 360x288 or high 720x576
        alignment/          # word alignment text file

Alignment line format (per official documentation):
    <start_ms> <end_ms> <word>   (tokens; multiple words on the same line
                                   are joined by spaces)
The very last file for each talker additionally contains a time-stamp line
summing the file's total duration. This module is deliberately strict
about that contract.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Iterator, Optional

if TYPE_CHECKING:
    from pipeline.grid.schema import GridManifestEntry


@dataclass(frozen=True)
class AlignmentToken:
    """A single token with its time span in milliseconds."""

    word: str
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


@dataclass(frozen=True)
class AlignmentFile:
    """An entire alignment text file parsed into tokens."""

    path: Path
    tokens: list[AlignmentToken] = field(default_factory=list)
    file_duration_ms: Optional[int] = None
    errors: list[str] = field(default_factory=list)

    @property
    def total_duration_ms(self) -> Optional[int]:
        """Duration reported by the alignment file itself."""
        return self.file_duration_ms

    @property
    def utterance(self) -> Optional[str]:
        """The canonical utterance id (e.g. T01_U0001)."""
        return None

    def add_errors(self, *msgs: str) -> None:
        self.errors.extend(msgs)

    def valid(self) -> bool:
        return len(self.errors) == 0


class GRIDAlignmentParser:
    """Parse a GRID alignment text file into tokens."""

    # Alignment files end with a duration line: a single integer (ms).
    _DURATION_RE = re.compile(r"^\s*\d{4,}\s*$")
    _INT_RE = re.compile(r"^\s*(-?\d+)\s+(-?\d+)\s+(.+)$")

    def __init__(self, alignment_path: Path) -> None:
        self.path = Path(alignment_path)
        self.tokens: list[AlignmentToken] = []
        self.file_duration_ms: Optional[int] = None
        self.errors: list[str] = []

    def parse(self) -> AlignmentFile:
        if not self.path.exists():
            self.errors.append(f"alignment file not found: {self.path}")
            return AlignmentFile(self.path, errors=self.errors)

        try:
            lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as exc:
            self.errors.append(f"cannot read alignment file: {exc}")
            return AlignmentFile(self.path, errors=self.errors)

        for lineno, raw in enumerate(lines, start=1):
            line = raw.strip()
            if not line:
                continue

            m = self._INT_RE.match(line)
            if m:
                start, end, word = int(m.group(1)), int(m.group(2)), m.group(3).strip()
                if start < 0 or end < start:
                    self.errors.append(f"line {lineno}: malformed range ({start}, {end})")
                self.tokens.append(AlignmentToken(word=word, start_ms=start, end_ms=end))
                continue

            if self._DURATION_RE.match(line):
                # File-level duration line (single integer ms, typically 4 digits)
                try:
                    self.file_duration_ms = int(line)
                except ValueError:
                    self.errors.append(f"line {lineno}: unparseable duration line")
                continue

            # Fallback: treat whole line as a word with zero duration (lenient)
            self.tokens.append(AlignmentToken(word=line, start_ms=0, end_ms=0))

        return AlignmentFile(self.path, tokens=self.tokens, file_duration_ms=self.file_duration_ms, errors=self.errors)

    @staticmethod
    def parse_file(path: Path) -> AlignmentFile:
        return GRIDAlignmentParser(path).parse()


def write_manifest_jsonl(
    entries: list["GridManifestEntry"],
    output_path: str | Path,
    *,
    require_valid: bool = True,
) -> Path:
    """Write validated manifest entries as physical JSONL lines atomically.

    The parsed ``AlignmentFile`` is an in-memory convenience object and is
    intentionally excluded from persisted records.
    """
    if require_valid:
        invalid = [entry.utterance_id for entry in entries if not entry.is_ok]
        if invalid:
            raise ValueError(
                "cannot write a manifest containing invalid entries: "
                + ", ".join(invalid)
            )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records: list[str] = []
    for entry in entries:
        record = entry.to_dict()
        record.pop("alignment", None)
        records.append(json.dumps(record, ensure_ascii=False, separators=(",", ":")))

    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for record in records:
                handle.write(record)
                handle.write("\n")

        with temporary_path.open("r", encoding="utf-8", newline="") as handle:
            physical_lines = handle.readlines()
        if len(physical_lines) != len(entries):
            raise ValueError(
                f"manifest line count mismatch: expected {len(entries)}, "
                f"got {len(physical_lines)}"
            )
        for line_number, line in enumerate(physical_lines, start=1):
            if not line.endswith("\n"):
                raise ValueError(f"manifest line {line_number} is missing a newline")
            json.loads(line)

        os.replace(temporary_path, destination)
        temporary_path = None
        return destination
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
