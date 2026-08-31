"""Target encoding for letter-sequence supervision.

The dataset label is RECORDING-level: ``metadata.letter == "A"`` means the
whole temporal sequence is the spoken letter A. It does NOT mean every frame
is A. Therefore targets are per-RECORDING id sequences consumed by CTC:

    single-letter recording  "A"    ->  [1]                (TargetType.SINGLE_LETTER)
    future multi-letter      "HELLO"->  [8, 5, 12, 12, 15] (TargetType.MULTI_LETTER)

Frame-level letter labels are NEVER produced by this module — CTC learns the
temporal alignment itself. Repeated letters (e.g. "LL") encode as two ids and
remain distinguishable from "L" through CTC's blank-collapse rules, which is
exactly why CTC (not per-frame CE) is the correct objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Sequence

import torch

from alphabet.vocab import LetterVocabulary, VocabularyError


class TargetType(str, Enum):
    """Recording target type (metadata distinction, §15 of the spec)."""

    SINGLE_LETTER = "single_letter"
    MULTI_LETTER = "multi_letter"


def normalize_text(text: str) -> str:
    """Canonicalize a target text: strip + uppercase."""
    if text is None:
        raise VocabularyError("target text is None")
    return str(text).strip().upper()


def classify_target(text: str) -> TargetType:
    """single_letter for exactly one A-Z char, multi_letter otherwise."""
    t = normalize_text(text)
    if len(t) == 0:
        raise VocabularyError("empty target text")
    return TargetType.SINGLE_LETTER if len(t) == 1 else TargetType.MULTI_LETTER


def encode_text(vocab: LetterVocabulary, text: str) -> List[int]:
    """Encode a recording-level target text to CTC ids (validates A-Z)."""
    return vocab.encode(normalize_text(text))


@dataclass
class TargetBatch:
    """Padded CTC target batch.

    targets        : [B, S_max] int64, PAD-filled (pad value is never a valid
                     class because CTC only reads ``target_lengths[i]`` ids
                     of row i; pad = 0 == blank is intentionally outside the
                     valid-id space 1..26 so bugs surface loudly).
    target_lengths : [B] int64
    """

    targets: torch.Tensor
    target_lengths: torch.Tensor
    texts: List[str]

    def __len__(self) -> int:
        return self.targets.shape[0]


PAD_TARGET_ID = 0  # == CTC blank; illegal as a real target id (see above)


def collate_targets(vocab: LetterVocabulary, texts: Sequence[str]) -> TargetBatch:
    """Build [B, S_max] padded target tensor + lengths from label texts."""
    if len(texts) == 0:
        raise VocabularyError("collate_targets received zero texts")
    encoded = [encode_text(vocab, t) for t in texts]
    lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
    if int(lengths.max()) == 0:
        raise VocabularyError("all target texts are empty")
    s_max = int(lengths.max())
    targets = torch.full((len(encoded), s_max), PAD_TARGET_ID, dtype=torch.long)
    for i, e in enumerate(encoded):
        targets[i, : len(e)] = torch.tensor(e, dtype=torch.long)
    return TargetBatch(targets=targets, target_lengths=lengths, texts=list(texts))
