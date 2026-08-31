"""Calibration — temperature scaling, ECE, abstention thresholds.

Rules (§14 of the spec):
- Confidence thresholds are NEVER arbitrary (no "0.90 means correct").
- Temperature is fit on VALIDATION aggregated logits by NLL minimization.
- Abstention thresholds are chosen on validation data only, subject to a
  coverage target; test data is untouched until final evaluation.
- ECE + reliability bins are returned as machine-readable dicts (no plots).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch


def fit_temperature(
    aggregated_logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 100,
) -> float:
    """Fit a single temperature on validation data (NLL, LBFGS).

    aggregated_logits: [N, num_letters] frame-averaged letter logits
                       (pre-softmax, i.e. log of the aggregated posterior up
                       to a constant — see uncertainty.aggregate_letter_posterior).
    labels:            [N] int64 letter labels (0..num_letters-1).
    """
    if aggregated_logits.shape[0] != labels.shape[0]:
        raise ValueError("logits/labels batch mismatch")
    logits = aggregated_logits.float().clone()
    labels = labels.long()
    log_temp = torch.zeros(1, requires_grad=True)

    def nll():
        temp = log_temp.exp().clamp(min=1e-3, max=100.0)
        return torch.nn.functional.cross_entropy(logits / temp, labels)

    opt = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = nll()
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_temp.exp().clamp(min=1e-3, max=100.0).item())
def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 15,
) -> Dict:
    """ECE + per-bin reliability data (machine-readable, no plots)."""
    if len(confidences) != len(correct):
        raise ValueError("confidences/correct length mismatch")
    n = len(confidences)
    if n == 0:
        return {"ece": float("nan"), "n": 0, "bins": []}
    conf = list(map(float, confidences))
    corr = [bool(c) for c in correct]
    bins: List[Dict] = []
    ece = 0.0
    n_bins = int(n_bins)
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        if b < n_bins - 1:
            idx = [i for i in range(n) if lo <= conf[i] < hi]
        else:  # last bin inclusive of 1.0
            idx = [i for i in range(n) if lo <= conf[i] <= hi]
        if not idx:
            bins.append({"lo": lo, "hi": hi, "count": 0,
                         "mean_confidence": None, "accuracy": None})
            continue
        mean_conf = sum(conf[i] for i in idx) / len(idx)
        acc = sum(1 for i in idx if corr[i]) / len(idx)
        bins.append({"lo": lo, "hi": hi, "count": len(idx),
                     "mean_confidence": mean_conf, "accuracy": acc})
        ece += (len(idx) / n) * abs(acc - mean_conf)
    return {"ece": ece, "n": n, "bins": bins}

def select_abstention_thresholds(
    confidences: Sequence[float],
    margins: Sequence[float],
    correct: Sequence[bool],
    coverage_target: float = 0.90,
) -> Dict[str, float]:
    """Choose confidence + margin thresholds on validation data only.

    Confidence threshold: the value such that the covered subset
    (confidence >= threshold) has coverage >= coverage_target with the
    highest covered accuracy; ties resolved toward maximal coverage.

    Margin threshold: same procedure over margins within the covered subset.
    """
    if not (0.0 < coverage_target <= 1.0):
        raise ValueError("coverage_target must be in (0, 1]")
    n = len(confidences)
    if n == 0 or n != len(margins) or n != len(correct):
        raise ValueError("threshold selection inputs mismatch/empty")

    conf = list(map(float, confidences))
    marg = list(map(float, margins))
    corr = [bool(c) for c in correct]

    best: Optional[Dict] = None
    for t in sorted(set(conf)):
        idx = [i for i in range(n) if conf[i] >= t]
        coverage = len(idx) / n
        if coverage < coverage_target:
            continue
        acc = sum(1 for i in idx if corr[i]) / len(idx)
        key = (acc, coverage)
        if best is None or key > (best["covered_accuracy"], best["coverage"]):
            best = {"confidence_threshold": t, "coverage": coverage,
                    "covered_accuracy": acc}
    if best is None:
        # Coverage unreachable even at min confidence — record shortfall.
        best = {"confidence_threshold": min(conf), "coverage": 1.0,
                "covered_accuracy": sum(corr) / n}

    t_conf = best["confidence_threshold"]
    covered = [i for i in range(n) if conf[i] >= t_conf]
    best_m: Optional[Dict] = None
    for t in sorted({marg[i] for i in covered}) or [0.0]:
        idx = [i for i in covered if marg[i] >= t]
        coverage = len(idx) / n
        if coverage < coverage_target:
            continue
        acc = sum(1 for i in idx if corr[i]) / len(idx)
        key = (acc, coverage)
        if best_m is None or key > (best_m["covered_accuracy"], best_m["coverage"]):
            best_m = {"margin_threshold": t, "coverage": coverage,
                      "covered_accuracy": acc}
    if best_m is None:
        best_m = {"margin_threshold": min(marg[i] for i in covered) if covered else 0.0,
                  "coverage": 1.0,
                  "covered_accuracy": sum(corr[i] for i in covered) / max(len(covered), 1)}

    return {
        "confidence_threshold": float(best["confidence_threshold"]),
        "margin_threshold": float(best_m["margin_threshold"]),
        "validation_coverage": float(best["coverage"]),
        "validation_covered_accuracy": float(best["covered_accuracy"]),
        "coverage_target": float(coverage_target),
    }

