"""Deterministic corpus-level lip-reading metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


def _normalize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, ref_token in enumerate(reference, start=1):
        current = [row]
        for column, hyp_token in enumerate(hypothesis, start=1):
            current.append(min(
                previous[column] + 1,
                current[column - 1] + 1,
                previous[column - 1] + int(ref_token != hyp_token),
            ))
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class CorpusMetric:
    errors: int
    reference_units: int
    rate: float
    samples: int


def _corpus_metric(
    references: Iterable[str],
    hypotheses: Iterable[str],
    *,
    character_level: bool,
) -> CorpusMetric:
    refs = list(references)
    hyps = list(hypotheses)
    if len(refs) != len(hyps):
        raise ValueError("references and hypotheses length mismatch")

    errors = 0
    reference_units = 0
    for reference, hypothesis in zip(refs, hyps):
        ref_units = list(reference.lower()) if character_level else _normalize(reference)
        hyp_units = list(hypothesis.lower()) if character_level else _normalize(hypothesis)
        errors += _edit_distance(ref_units, hyp_units)
        reference_units += len(ref_units)

    rate = 0.0 if reference_units == 0 and errors == 0 else (
        1.0 if reference_units == 0 else errors / reference_units
    )
    return CorpusMetric(errors, reference_units, rate, len(refs))


def corpus_wer(references: Iterable[str], hypotheses: Iterable[str]) -> CorpusMetric:
    """Compute corpus WER using aggregate word edit distance."""
    return _corpus_metric(references, hypotheses, character_level=False)


def corpus_cer(references: Iterable[str], hypotheses: Iterable[str]) -> CorpusMetric:
    """Compute corpus CER using aggregate character edit distance."""
    return _corpus_metric(references, hypotheses, character_level=True)
