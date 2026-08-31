"""Phase 5 decoder heads on top of the Phase 4 Conformer (T x 192).

    Conformer output h [B, T, d_model]
        -> LetterCTCHead   -> [B, T, 27]  (blank + A-Z)   PRIMARY, always on
        -> PhoneticHead    -> [B, T, 26]  optional, DISABLED by default
        -> VisemeHead      -> [B, T, 15]  optional, DISABLED by default

The letter head is a linear projection (DESIGN.md §6 "Letter CTC head");
an optional non-linear variant exists for experimentation. The model treats
the temporal representation intact: heads are per-timestep, the sequence
structure is resolved by CTC, never by per-frame argmax classification.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from temporal.encoder import TemporalEncoder, TemporalEncoderConfig


class CTCHead(nn.Module):
    """Per-timestep projection to CTC logits (optionally with a hidden layer)."""

    def __init__(self, d_model: int, num_classes: int,
                 hidden: Optional[int] = None):
        super().__init__()
        if hidden is None:
            self.proj: nn.Module = nn.Linear(d_model, num_classes)
        else:
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, num_classes),
            )

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.proj(h)
        if mask is not None:
            logits = logits * mask.unsqueeze(-1).to(logits.dtype)
        return logits


class MultiTaskAlphabetModel(nn.Module):
    """Phase 4 encoder + Phase 5 heads.

    ``phone_vocab_size`` / ``viseme_vocab_size`` are None (heads absent) by
    default; auxiliary supervision must be explicitly enabled in config AND
    backed by the versioned derived labels in alphabet/phonetics.py.
    """

    def __init__(
        self,
        temporal_config: TemporalEncoderConfig,
        letter_vocab_size: int,
        phone_vocab_size: Optional[int] = None,
        viseme_vocab_size: Optional[int] = None,
        letter_head_hidden: Optional[int] = None,
    ):
        super().__init__()
        self.encoder = TemporalEncoder(temporal_config)
        d_model = temporal_config.d_model
        self.letter_head = CTCHead(d_model, letter_vocab_size, letter_head_hidden)
        self.phone_head = (
            CTCHead(d_model, phone_vocab_size) if phone_vocab_size else None
        )
        self.viseme_head = (
            CTCHead(d_model, viseme_vocab_size) if viseme_vocab_size else None
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        h = self.encoder(x, mask)                       # [B, T, d_model]
        out: Dict[str, torch.Tensor] = {
            "letter_logits": self.letter_head(h, mask)  # [B, T, 27]
        }
        if self.phone_head is not None:
            out["phone_logits"] = self.phone_head(h, mask)
        if self.viseme_head is not None:
            out["viseme_logits"] = self.viseme_head(h, mask)
        return out

    # -- introspection (exact parameter counting, never estimated) ----------
    def parameter_counts(self) -> Dict[str, int]:
        def count(module):
            return sum(p.numel() for p in module.parameters()) if module is not None else 0

        return {
            "encoder": count(self.encoder),
            "letter_head": count(self.letter_head),
            "phone_head": count(self.phone_head),
            "viseme_head": count(self.viseme_head),
            "total_trainable": count(self),
        }
