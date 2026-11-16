"""Track B: Continuous Video-to-Text Recognition and CTC Sequence Decoder.

Visual-only continuous speech sequence decoding for silent video.
Audio is completely disabled during inference.
Computes Word Error Rate (WER) and Character Error Rate (CER) on continuous sentences.
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any, List, Tuple
import cv2
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]


def levenshtein_distance(ref: List[str], hyp: List[str]) -> int:
    """Compute Levenshtein edit distance between token sequences."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,     # Deletion
                    dp[i][j - 1] + 1,     # Insertion
                    dp[i - 1][j - 1] + 1  # Substitution
                )
    return dp[n][m]


def compute_wer(reference_words: List[str], hypothesis_words: List[str]) -> float:
    if not reference_words:
        return 1.0 if hypothesis_words else 0.0
    dist = levenshtein_distance(reference_words, hypothesis_words)
    return dist / len(reference_words)


def compute_cer(reference_text: str, hypothesis_text: str) -> float:
    ref_chars = list(reference_text.replace(" ", ""))
    hyp_chars = list(hypothesis_text.replace(" ", ""))
    if not ref_chars:
        return 1.0 if hyp_chars else 0.0
    dist = levenshtein_distance(ref_chars, hyp_chars)
    return dist / len(ref_chars)


class VisualCTCSequenceDecoder:
    """Greedy CTC sequence decoder mapping spatio-temporal video features to token sequences."""
    def __init__(self, vocabulary: List[str], blank_idx: int = 0):
        self.vocabulary = vocabulary
        self.blank_idx = blank_idx

    def decode_greedy(self, logits: torch.Tensor) -> Tuple[List[str], str]:
        """logits: [T, vocab_size] unnormalized or log-softmax."""
        best_indices = torch.argmax(logits, dim=-1).tolist()
        collapsed = []
        prev = self.blank_idx

        for idx in best_indices:
            if idx != prev:
                if idx != self.blank_idx:
                    collapsed.append(idx)
                prev = idx

        tokens = [self.vocabulary[idx] for idx in collapsed if idx < len(self.vocabulary)]
        transcript = " ".join(tokens)
        return tokens, transcript


def evaluate_continuous_sentences(
    test_manifest_path: Path,
    max_samples: int = 20,
    output_report_path: Path | None = None
) -> dict[str, Any]:
    """Fail closed until an actual trained visual-only decoder is available.

    CTC collapse and edit-distance helpers are software utilities, not a speech
    model. A constant sentence is not a prediction and cannot be evaluated as one.
    Use pipeline.multidataset.sequence for scoring actual decoder outputs.
    """
    raise RuntimeError(
        "Continuous recognition unavailable: no verified trained visual-only "
        "decoder checkpoint. No predictions or recognition metrics were produced."
    )


if __name__ == "__main__":
    evaluate_continuous_sentences(ROOT / "data/processed/grid_multispeaker_v1/video_inventory.jsonl")
