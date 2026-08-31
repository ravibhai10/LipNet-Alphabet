"""CTC sequence decoding.

Greedy CTC decoding is implemented and tested; the ``SequenceDecoder``
protocol keeps the interface modular so beam search / a language prior can be
added later WITHOUT touching the model or training code (Section 8: no LM yet).

Greedy rule (per recording):
    tokens = argmax_v logits[t] for t in 0..input_len-1
    collapse consecutive repeats, then remove the blank:
        blank H H blank E E blank L L blank  ->  H E L
        blank L L blank L blank              ->  L L   (repeats preserved)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Protocol, Sequence, Tuple

import torch


@dataclass
class DecodedSequence:
    """Result of decoding one recording."""

    text: str
    #: First frame index of each emitted (collapsed) token
    emission_frames: List[int] = field(default_factory=list)
    #: Per-frame argmax class indices BEFORE collapsing
    raw_frames: List[int] = field(default_factory=list)


class SequenceDecoder(Protocol):
    """Decode [B, T, V] letter logits into per-recording sequences."""

    def decode_batch(
        self, letter_logits: torch.Tensor, input_lengths: torch.Tensor
    ) -> List[DecodedSequence]: ...


class GreedyCTCDecoder:
    """Canonical greedy CTC decoder (blank index configurable, default 0)."""

    def __init__(self, blank_index: int = 0):
        self.blank_index = int(blank_index)

    def _token_to_char(self, token: int) -> str:
        """Map a non-blank class id to its output symbol."""
        return str(token)

    def decode_batch(
        self, letter_logits: torch.Tensor, input_lengths: torch.Tensor
    ) -> List[DecodedSequence]:
        if letter_logits.ndim != 3:
            raise ValueError(
                f"letter_logits must be [B, T, V]; got {tuple(letter_logits.shape)}"
            )
        b = letter_logits.shape[0]
        lengths = input_lengths.detach().cpu().tolist()
        if len(lengths) != b:
            raise ValueError("input_lengths batch size mismatch")
        results = []
        for i in range(b):
            results.append(self.decode_single(letter_logits[i], int(lengths[i])))
        return results

    def decode_single(
        self, letter_logits_i: torch.Tensor, input_length: int
    ) -> DecodedSequence:
        """Decode one [T, V] logits using its first input_length frames.

        Also accepts [1, T, V] (batch-size 1).
        """
        if letter_logits_i.ndim == 3 and letter_logits_i.shape[0] == 1:
            letter_logits_i = letter_logits_i.squeeze(0)
        if letter_logits_i.ndim != 2:
            raise ValueError(
                f"expected [T, V] or [1, T, V] logits; got {tuple(letter_logits_i.shape)}"
            )
        t_max = letter_logits_i.shape[0]
        if input_length <= 0 or input_length > t_max:
            raise ValueError(f"invalid input_length {input_length} for T={t_max}")
        raw = letter_logits_i.detach().cpu()[:input_length].argmax(dim=-1).tolist()
        return self._collapse(raw)

    def _collapse(self, raw: Sequence[int]) -> DecodedSequence:
        text_chars: List[str] = []
        emission_frames: List[int] = []
        prev = None
        for t, tok in enumerate(raw):
            tok = int(tok)
            if tok != prev and tok != self.blank_index:
                text_chars.append(self._token_to_char(tok))
                emission_frames.append(t)
            prev = tok
        return DecodedSequence(
            text="".join(text_chars), emission_frames=emission_frames,
            raw_frames=[int(x) for x in raw],
        )


class ClassMappingGreedyCTCDecoder(GreedyCTCDecoder):
    """Greedy decoder that maps class ids -> letters via a vocabulary."""

    def __init__(self, index_to_letter, blank_index: int = 0):
        super().__init__(blank_index=blank_index)
        self._index_to_letter = index_to_letter

    def _token_to_char(self, token: int) -> str:
        ch = self._index_to_letter(token)
        if ch is None:
            raise ValueError(f"non-blank class {token} maps to no letter")
        return ch
