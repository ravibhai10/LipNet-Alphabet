"""CTC losses for the letter task and optional auxiliary heads.

L_total = lambda_letter * L_CTC(letter)
        + lambda_phone  * L_CTC(phone)    (only if the aux head is enabled)
        + lambda_viseme * L_CTC(viseme)   (only if the aux head is enabled)

CTC correctness contract (tested in tests/test_phase5_ctc.py):
    letter_logits : [B, T, V]   raw logits (V = 27 for the letter head)
    input_lengths : [B]         valid frames per recording (mask.sum(-1))
    targets       : [B, S]      class ids in 1..26 (0 == blank is illegal)
    target_lengths: [B]         S_i for each recording

torch.nn.functional.ctc_loss reads exactly input_lengths[i] frames and
target_lengths[i] target ids per sample, so padding NEVER leaks into the
loss. zero_infinity=True guards the (T < S) degenerate case with a 0 loss
instead of inf.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def logits_to_log_probs(
    logits: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """[B, T, V] logits -> [T, B, V] log-softmax (CTC layout, float32).

    The loss is always computed in float32 even under AMP — CTC log-sum-exps
    are the least numerically safe op in the pipeline.
    """
    if logits.ndim != 3:
        raise ValueError(f"expected [B, T, V] logits; got {tuple(logits.shape)}")
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return log_probs.transpose(0, 1)  # [T, B, V]


def ctc_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_index: int = 0,
) -> torch.Tensor:
    """Scalar CTC loss with explicit, validated lengths."""
    b, t, v = logits.shape
    if mask.shape != (b, t):
        raise ValueError(f"mask shape {tuple(mask.shape)} != {(b, t)}")
    if targets.shape[0] != b or target_lengths.shape[0] != b:
        raise ValueError("targets/target_lengths batch mismatch")
    input_lengths = mask.sum(dim=-1).long()
    if int(input_lengths.min()) <= 0:
        raise ValueError("recording with zero valid frames in batch")
    if (target_lengths > input_lengths).any():
        # CTC requires T_i >= S_i (need room for blanks between repeated ids).
        bad = (target_lengths > input_lengths).nonzero().flatten().tolist()
        raise ValueError(
            f"target longer than input for samples {bad}; CTC undefined"
        )
    log_probs = logits_to_log_probs(logits, mask)
    loss = F.ctc_loss(
        log_probs,
        targets.long(),
        input_lengths.cpu(),
        target_lengths.long().cpu(),
        blank=blank_index,
        reduction="mean",
        zero_infinity=True,
    )
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite CTC loss: {loss.item()}")
    return loss


class MultitaskLoss(nn.Module):
    """Weighted letter + optional phone/viseme CTC objectives.

    Auxiliary terms are active only when the corresponding head exists in the
    model outputs AND its weight > 0. With auxiliaries disabled (the default)
    this is exactly the primary letter CTC loss.
    """

    def __init__(
        self,
        lambda_letter: float = 1.0,
        lambda_phone: float = 0.0,
        lambda_viseme: float = 0.0,
        blank_index: int = 0,
    ):
        super().__init__()
        self.lambda_letter = float(lambda_letter)
        self.lambda_phone = float(lambda_phone)
        self.lambda_viseme = float(lambda_viseme)
        self.blank_index = int(blank_index)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        letter_targets: torch.Tensor,
        letter_target_lengths: torch.Tensor,
        phone_targets: Optional[torch.Tensor] = None,
        phone_target_lengths: Optional[torch.Tensor] = None,
        viseme_targets: Optional[torch.Tensor] = None,
        viseme_target_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        parts: Dict[str, torch.Tensor] = {}
        letter_loss = ctc_loss(
            outputs["letter_logits"], mask, letter_targets,
            letter_target_lengths, self.blank_index,
        )
        parts["letter"] = letter_loss
        total = self.lambda_letter * letter_loss

        if (
            "phone_logits" in outputs
            and self.lambda_phone > 0
            and phone_targets is not None
        ):
            phone_loss = ctc_loss(
                outputs["phone_logits"], mask, phone_targets,
                phone_target_lengths, self.blank_index,
            )
            parts["phone"] = phone_loss
            total = total + self.lambda_phone * phone_loss

        if (
            "viseme_logits" in outputs
            and self.lambda_viseme > 0
            and viseme_targets is not None
        ):
            viseme_loss = ctc_loss(
                outputs["viseme_logits"], mask, viseme_targets,
                viseme_target_lengths, self.blank_index,
            )
            parts["viseme"] = viseme_loss
            total = total + self.lambda_viseme * viseme_loss

        parts["total"] = total
        return parts
