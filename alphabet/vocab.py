"""Canonical letter vocabulary — the SINGLE source of truth for Phase 5.

Everything that needs the CTC output vocabulary (heads, targets, decoding,
uncertainty, metrics, checkpoints) must import it from here. The vocabulary
size is NEVER hard-coded anywhere else.

CTC vocabulary (27 classes):
    index 0  : <blank>   (CTC blank — a TRAINING/ALIGNMENT symbol, never a
                           prediction; greedy decoding removes it)
    index 1..26: A..Z

UNKNOWN and AMBIGUOUS are deliberately NOT vocabulary entries. They are
POST-PROCESSING STATES (see ``OutputState``) computed from calibrated
uncertainty on validation data. The model is never trained to output them
because no training samples exist for those states.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Sequence

CTC_BLANK_TOKEN = "<blank>"
#: Letters covered by the dataset schema (data/schema.py Letter enum).
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class OutputState(str, Enum):
    """Post-processing decision states — NOT model output classes.

    The MODEL only ever produces letter posteriors. UNKNOWN / AMBIGUOUS are
    emitted by the uncertainty layer (alphabet/uncertainty.py) using
    thresholds calibrated on validation data only.
    """

    LETTER = "letter"
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"


class VocabularyError(ValueError):
    """Raised when a text/id cannot be represented in the vocabulary."""


@dataclass(frozen=True)
class LetterVocabulary:
    """Immutable A-Z + CTC-blank vocabulary (size 27)."""

    letters: tuple = tuple(ALPHABET)

    def __post_init__(self):
        letters = tuple(self.letters)
        if len(letters) == 0 or len(set(letters)) != len(letters):
            raise VocabularyError("letters must be non-empty and unique")
        for ch in letters:
            if not (isinstance(ch, str) and len(ch) == 1 and ch.isupper() and ch.isalpha()):
                raise VocabularyError(f"invalid letter token: {ch!r}")
        object.__setattr__(self, "letters", letters)

    # -- structure ----------------------------------------------------------
    @property
    def blank_index(self) -> int:
        return 0

    @property
    def letters_offset(self) -> int:
        """Letter index = position + this offset (CTC blank sits at 0)."""
        return 1

    @property
    def num_letters(self) -> int:
        return len(self.letters)

    @property
    def size(self) -> int:
        """Full CTC output dimension (blank + letters) = 27."""
        return len(self.letters) + 1

    # -- conversion ----------------------------------------------------------
    def letter_to_index(self, letter: str) -> int:
        try:
            return self.letters.index(letter) + self.letters_offset
        except ValueError:
            raise VocabularyError(f"letter {letter!r} not in vocabulary")

    def index_to_letter(self, index: int):
        """Return the letter for a class index, or None for the blank."""
        if index == self.blank_index:
            return None
        offset = index - self.letters_offset
        if 0 <= offset < self.num_letters:
            return self.letters[offset]
        raise VocabularyError(f"class index {index} out of range [0, {self.size})")

    def encode(self, text: str) -> List[int]:
        """Encode a non-empty A-Z string to CTC target ids (no blanks)."""
        if not text:
            raise VocabularyError("cannot encode empty target text")
        text = text.strip().upper()
        if any(ch not in self.letters for ch in text):
            bad = sorted({ch for ch in text if ch not in self.letters})
            raise VocabularyError(f"target text contains non-letter chars: {bad}")
        return [self.letter_to_index(ch) for ch in text]

    def decode_ids(self, ids: Sequence[int]) -> str:
        """Map class ids to text; blank ids are illegal here (use decoding.py)."""
        out = []
        for i in ids:
            ch = self.index_to_letter(int(i))
            if ch is None:
                raise VocabularyError("blank index inside target ids")
            out.append(ch)
        return "".join(out)

    # -- persistence (checkpoint provenance) ----------------------------------
    def to_dict(self) -> Dict:
        return {
            "kind": "letter_ctc_v1",
            "blank_token": CTC_BLANK_TOKEN,
            "blank_index": self.blank_index,
            "letters": list(self.letters),
            "size": self.size,
        }

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Dict) -> "LetterVocabulary":
        if data.get("kind") != "letter_ctc_v1":
            raise VocabularyError(f"unknown vocabulary kind: {data.get('kind')!r}")
        vocab = cls(letters=tuple(data["letters"]))
        if vocab.size != data.get("size") or vocab.blank_index != data.get("blank_index"):
            raise VocabularyError("vocabulary file inconsistent with class definition")
        return vocab

    @classmethod
    def load(cls, path: Path) -> "LetterVocabulary":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


#: Module-level canonical instance. Import this — never construct ad-hoc maps.
DEFAULT_VOCABULARY = LetterVocabulary()
