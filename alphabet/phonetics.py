"""Versioned phonetic / viseme supervision definitions — AUXILIARY ONLY.

SUPERVISION PROVENANCE (audited, reproducible, no invented labels)
-----------------------------------------------------------------
The dataset contains NO phoneme annotations, NO viseme annotations and NO
phoneme-level timestamps. The ONLY observed supervision is the recording-level
spoken letter (data/schema.py ``RecordingMetadata.letter``).

This module defines the two auxiliary supervision sources that CAN be derived
deterministically from that observed label, per the Phase 5 decision rules:

1. Letter-name ARPABET strings (``LETTER_NAME_PHONES``)
   Source: DESIGN.md Table 1 ("Letter name -> phone string", ARPABET, US
   default accent; Z = Z IY). Version: ``letter_names_arpabet_us_v1.0``.
   Derivation rule: a recording labelled X means the speaker produced the
   *name* of letter X; the canonical US letter-name pronunciation is a fixed
   phone string, so the phone supervision is generated as
   ``phones(letter_label)`` — legitimate "explicitly generated pronunciation
   targets" (source class #5 in the Phase 5 audit). Nothing is aligned to
   frames by hand; CTC learns the alignment.

2. Phone -> viseme map (``PHONE_TO_VISVHEME_NAME``)
   Source: DESIGN.md Table 3 ("Viseme label topology (initial)"). Version:
   ``bear_harvey_initial_v1.0``. One documented extension beyond Table 3:
   AH (schwa-like central vowel, occurs in W "D AH B ..." and R "AA R")
   is mapped to V12 (back/central vowel group) — recorded here so the
   mapping is fully reproducible.

These are DERIVED, not observed, supervision. Both auxiliary heads are
DISABLED by default (alphabet/config.py AuxiliaryConfig.enabled=False) and
must never contaminate the primary letter-CTC experiment. Any experimental
use must record the versions above.

Phone vocabulary note: DESIGN.md mentions a 45-phone ARPABET space, but only
25 distinct phones occur in the 26 letter names. The auxiliary phone
vocabulary is built from the phones that ACTUALLY receive supervision
(25 phones + SIL + CTC blank = 27 classes) — not from the unused 45.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

LETTER_NAME_PHONES_VERSION = "letter_names_arpabet_us_v1.0"
LETTER_NAME_PHONES_SOURCE = (
    "DESIGN.md Table 1 — letter-name ARPABET, US default accent (Z = Z IY)"
)

#: Canonical US letter-name pronunciations (ARPABET), DESIGN.md Table 1.
LETTER_NAME_PHONES: Dict[str, Tuple[str, ...]] = {
    "A": ("EY",),
    "B": ("B", "IY"),
    "C": ("S", "IY"),
    "D": ("D", "IY"),
    "E": ("IY",),
    "F": ("EH", "F"),
    "G": ("JH", "IY"),
    "H": ("EY", "CH"),
    "I": ("AY",),
    "J": ("JH", "EY"),
    "K": ("K", "EY"),
    "L": ("EH", "L"),
    "M": ("EH", "M"),
    "N": ("EH", "N"),
    "O": ("OW",),
    "P": ("P", "IY"),
    "Q": ("K", "Y", "UW"),
    "R": ("AA", "R"),
    "S": ("EH", "S"),
    "T": ("T", "IY"),
    "U": ("Y", "UW"),
    "V": ("V", "IY"),
    "W": ("D", "AH", "B", "AH", "L", "Y", "UW"),
    "X": ("EH", "K", "S"),
    "Y": ("W", "AY"),
    "Z": ("Z", "IY"),
}

SIL = "SIL"  # silence token between letters in multi-letter targets

#: Phones that actually occur in letter-name supervision (sorted, stable).
OCCURRING_PHONES: Tuple[str, ...] = tuple(sorted(
    {p for phones in LETTER_NAME_PHONES.values() for p in phones}
))


class PhoneVocabulary:
    """Auxiliary CTC phone vocabulary: [blank] + occurring phones + SIL."""

    def __init__(self, phones: Tuple[str, ...] = OCCURRING_PHONES,
                 version: str = LETTER_NAME_PHONES_VERSION):
        self.version = version
        self.phones = tuple(phones)
        self.sil_index = len(self.phones) + 1  # after blank + phones
        self._to_index = {p: i + 1 for i, p in enumerate(self.phones)}

    @property
    def blank_index(self) -> int:
        return 0

    @property
    def size(self) -> int:
        return len(self.phones) + 2  # blank + phones + SIL

    def encode(self, phone_sequence: List[str]) -> List[int]:
        ids = []
        for p in phone_sequence:
            if p == SIL:
                ids.append(self.sil_index)
            elif p in self._to_index:
                ids.append(self._to_index[p])
            else:
                raise ValueError(f"phone {p!r} not in auxiliary phone vocabulary")
        return ids

    def to_dict(self) -> Dict:
        return {
            "kind": "phone_ctc_v1",
            "version": self.version,
            "blank_index": self.blank_index,
            "sil_index": self.sil_index,
            "phones": list(self.phones),
            "size": self.size,
        }
VISEME_MAP_VERSION = "bear_harvey_initial_v1.0"
VISEME_MAP_SOURCE = (
    "DESIGN.md Table 3 (initial viseme topology, Bear-Harvey-derived); "
    "documented extension: AH -> V12 (central/back vowel group)"
)

#: Phone -> viseme name, per DESIGN.md Table 3 (+ documented AH extension).
_PHONE_TO_VISVHEME_NAME: Dict[str, str] = {
    "B": "V1", "P": "V1", "M": "V1",          # bilabial
    "F": "V2", "V": "V2",                     # labiodental
    "TH": "V3", "DH": "V3",                   # (not in letter names; kept for completeness)
    "S": "V4", "Z": "V4",                     # sibilant
    "SH": "V5", "ZH": "V5", "CH": "V5", "JH": "V5",
    "T": "V6", "D": "V6", "N": "V6",
    "K": "V7", "G": "V7", "Y": "V7",
    "R": "V8",
    "L": "V9",
    "W": "V10",
    "IY": "V11", "IH": "V11", "EH": "V11", "AE": "V11",
    "AA": "V12", "AO": "V12", "OW": "V12", "UW": "V12", "AH": "V12",
    "EY": "V13", "AY": "V13", "OY": "V13", "AW": "V13",
}

#: Ordered viseme names (V1..V13) so ids are stable.
VISemes = tuple(f"V{i}" for i in range(1, 14))


class VisemeVocabulary:
    """Auxiliary CTC viseme vocabulary: [blank] + V1..V13 + SIL."""

    def __init__(self, version: str = VISEME_MAP_VERSION):
        self.version = version
        self.visemes = VISemes
        self.sil_index = len(self.visemes) + 1
        self._to_index = {v: i + 1 for i, v in enumerate(self.visemes)}

    @property
    def blank_index(self) -> int:
        return 0

    @property
    def size(self) -> int:
        return len(self.visemes) + 2

    def encode(self, viseme_sequence: List[str]) -> List[int]:
        ids = []
        for v in viseme_sequence:
            if v == SIL:
                ids.append(self.sil_index)
            elif v in self._to_index:
                ids.append(self._to_index[v])
            else:
                raise ValueError(f"viseme {v!r} not in viseme vocabulary")
        return ids

    def to_dict(self) -> Dict:
        return {
            "kind": "viseme_ctc_v1",
            "version": self.version,
            "blank_index": self.blank_index,
            "sil_index": self.sil_index,
            "visemes": list(self.visemes),
            "size": self.size,
        }


def letter_to_phones(letter: str) -> Tuple[str, ...]:
    """ARPABET string of a single letter name (raises on non A-Z)."""
    key = str(letter).strip().upper()
    if key not in LETTER_NAME_PHONES:
        raise ValueError(f"no phone string defined for letter {letter!r}")
    return LETTER_NAME_PHONES[key]


def text_to_phone_sequence(text: str) -> List[str]:
    """Phone supervision for a target text.

    Rule: phones(letter_1) + [SIL] + phones(letter_2) + [SIL] + ...
    SIL is inserted BETWEEN letters so CTC must emit silence at letter
    boundaries (this is what keeps e.g. "LL" = EH L | SIL | EH L distinct
    from a single "L" at the phone level, mirroring the letter-level blank
    mechanism).
    """
    t = str(text).strip().upper()
    if not t:
        raise ValueError("empty target text")
    seq: List[str] = []
    for i, ch in enumerate(t):
        if i > 0:
            seq.append(SIL)
        seq.extend(letter_to_phones(ch))
    return seq


def text_to_viseme_sequence(text: str) -> List[str]:
    """Viseme supervision for a target text (same SIL-between-letters rule)."""
    phone_seq = text_to_phone_sequence(text)
    out: List[str] = []
    for p in phone_seq:
        out.append(SIL if p == SIL else _PHONE_TO_VISVHEME_NAME[p])
    return out

