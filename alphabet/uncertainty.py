"""Ambiguity estimation — model belief + calibrated post-processing states.

The MODEL OUTPUT is a posterior over the 27 CTC classes per frame. The
POST-PROCESSING STATE (letter / UNKNOWN / AMBIGUOUS) is decided here from
calibrated uncertainty, using thresholds fit on VALIDATION data only (see
alphabet/calibration.py). Top-1 is never forced to be correct: every result
carries top-k, entropy and margin so downstream consumers can abstain.

Two uncertainty regimes (single-letter vs multi-letter recordings):

- single_letter: the frame-averaged letter evidence distribution —
  log-mean-exp over valid frames of the letter-only softmax (blank excluded,
  because "which letter is this" is a different question from "is there
  speech here"; the UNKNOWN state covers the latter). This is DESIGN.md §6.3.
- multi_letter: statistics along the greedy emission frames (confidence =
  mean posterior of the emitted letters at their emission frames; entropy /
  margin averaged over valid frames).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import torch

from alphabet.vocab import LetterVocabulary, OutputState


@dataclass
class UncertaintyResult:
    """Uncertainty readout for ONE recording."""

    prediction: str                    # greedy/aggregate predicted text
    confidence: float                  # posterior mass of the prediction
    top_k: List[Tuple[str, float]] = field(default_factory=list)
    entropy: float = float("nan")
    margin: float = float("nan")       # p(top1) - p(top2)
    abstain: bool = False
    state: OutputState = OutputState.LETTER
    temperature: float = 1.0


def aggregate_letter_posterior(
    letter_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Frame-averaged letter-only posterior, [B, num_letters].

    letter_logits: [B, T, V]; mask: [B, T] bool (True = valid frame).
    Blank column is EXCLUDED before normalization (documented above).
    Uses log-mean-exp (geometric mean) of the per-frame letter distributions.
    """
    if letter_logits.ndim != 3:
        raise ValueError("expected [B, T, V] letter logits")
    b, t, v = letter_logits.shape
    if mask.shape != (b, t):
        raise ValueError(f"mask {tuple(mask.shape)} != {(b, t)}")
    letter_logits = letter_logits.float() / float(temperature)
    letter_only = letter_logits[:, :, 1:]                    # drop blank
    log_p = torch.log_softmax(letter_only, dim=-1)           # [B, T, L]
    mask_f = mask.unsqueeze(-1).to(log_p.dtype)
    log_mean = (log_p * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
    return torch.softmax(log_mean, dim=-1)


def _posterior_readout(post: torch.Tensor, vocab: LetterVocabulary,
                       k: int = 5) -> Tuple[str, float, List[Tuple[str, float]], float, float]:
    """(prediction, confidence, top_k, entropy, margin) from [L] posterior."""
    probs = post.detach().cpu()
    top_probs, top_ids = torch.topk(probs, k=min(k, probs.shape[-1]))
    top_k = [(vocab.letters[i], float(p)) for p, i in zip(top_probs, top_ids)]
    prediction = top_k[0][0]
    confidence = float(top_probs[0])
    entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum())
    margin = float(top_probs[0] - top_probs[1]) if len(top_probs) > 1 else 1.0
    return prediction, confidence, top_k, entropy, margin


def single_letter_uncertainty(
    letter_logits: torch.Tensor,
    mask: torch.Tensor,
    vocab: LetterVocabulary,
    temperature: float = 1.0,
) -> List[UncertaintyResult]:
    """Uncertainty results for a batch, single-letter regime."""
    post = aggregate_letter_posterior(letter_logits, mask, temperature)
    results = []
    for i in range(post.shape[0]):
        pred, conf, topk, ent, marg = _posterior_readout(post[i], vocab)
        results.append(UncertaintyResult(
            prediction=pred, confidence=conf, top_k=topk,
            entropy=ent, margin=marg, temperature=float(temperature),
        ))
    return results
def sequence_uncertainty(
    letter_logits: torch.Tensor,
    mask: torch.Tensor,
    input_lengths: torch.Tensor,
    decoded_texts: Sequence[str],
    vocab: LetterVocabulary,
    temperature: float = 1.0,
) -> List[UncertaintyResult]:
    """Uncertainty results for a batch, multi-letter regime.

    Confidence: mean over emission frames of the blank-excluded letter
    posterior of the emitted letter. Entropy/margin: mean over valid frames.
    """
    probs_frames = torch.softmax(
        letter_logits.float()[:, :, 1:] / float(temperature), dim=-1)
    results = []
    for i in range(letter_logits.shape[0]):
        L = int(input_lengths[i])
        frame_probs = probs_frames[i, :L]                     # [L, letters]
        frame_ent = -(frame_probs * frame_probs.clamp_min(1e-12).log()).sum(-1)
        text = decoded_texts[i]
        if text:
            raw = letter_logits[i, :L].argmax(dim=-1).tolist()
            confs, prev = [], None
            for t, tok in enumerate(raw):
                if tok != prev and tok != vocab.blank_index:
                    letter_idx = tok - vocab.letters_offset
                    confs.append(float(frame_probs[t, letter_idx]))
                prev = tok
            confidence = sum(confs) / max(len(confs), 1)
        else:
            confidence = 0.0
        results.append(UncertaintyResult(
            prediction=text, confidence=confidence,
            entropy=float(frame_ent.mean()) if L else float("nan"),
            margin=float("nan"),
            temperature=float(temperature),
        ))
    # top_k / margin for display consistency use the aggregate posterior
    post = aggregate_letter_posterior(letter_logits, mask, temperature)
    for i, r in enumerate(results):
        _, _, topk, _, marg = _posterior_readout(post[i], vocab)
        r.top_k = topk
        if r.margin != r.margin:  # NaN -> use aggregate margin
            r.margin = marg
    return results


def decide_state(
    result: UncertaintyResult,
    confidence_threshold: float,
    margin_threshold: float,
) -> UncertaintyResult:
    """Apply calibrated thresholds; sets abstain + OutputState.

    Rule order (documented, deterministic):
      1. confidence < confidence_threshold  -> UNKNOWN (not enough evidence)
      2. margin      < margin_threshold     -> AMBIGUOUS (two letters tie)
      3. otherwise                          -> LETTER
    UNKNOWN implies abstain; AMBIGUOUS also abstains from committing top-1.
    """
    if result.confidence < confidence_threshold:
        result.state = OutputState.UNKNOWN
        result.abstain = True
    elif result.margin < margin_threshold:
        result.state = OutputState.AMBIGUOUS
        result.abstain = True
    else:
        result.state = OutputState.LETTER
        result.abstain = False
    return result

