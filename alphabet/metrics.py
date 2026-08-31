"""Phase 5 evaluation metrics (§20).

letter accuracy / sequence accuracy / macro F1 / top-3 / top-5 / CER /
confusion matrix / ECE / abstention rate. Single-letter recordings are
reported with an A-Z confusion matrix; multi-letter recordings with CER.
All outputs are plain Python data (JSON-serializable) — NO plots, NO UI.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from alphabet.vocab import LetterVocabulary


def levenshtein(a: str, b: str) -> int:
    """Character edit distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate, length-normalized; empty ref -> 0 iff empty hyp."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(reference, hypothesis) / len(reference)


def letter_accuracy(trues: Sequence[str], preds: Sequence[str]) -> float:
    _check_pair(trues, preds)
    if not trues:
        return float("nan")
    return sum(t == p for t, p in zip(trues, preds)) / len(trues)


def sequence_accuracy(trues: Sequence[str], preds: Sequence[str]) -> float:
    """Exact-match sequence accuracy (== letter accuracy for single letters)."""
    return letter_accuracy(trues, preds)


def macro_f1(trues: Sequence[str], preds: Sequence[str],
             vocab: LetterVocabulary) -> float:
    """Macro-averaged F1 over the full A-Z label space."""
    _check_pair(trues, preds)
    f1s = []
    for letter in vocab.letters:
        tp = sum(1 for t, p in zip(trues, preds) if t == letter and p == letter)
        fp = sum(1 for t, p in zip(trues, preds) if t != letter and p == letter)
        fn = sum(1 for t, p in zip(trues, preds) if t == letter and p != letter)
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(sum(f1s) / len(f1s))


def top_k_accuracy(probs: Sequence[Sequence[float]], labels: Sequence[int],
                   k: int) -> float:
    """probs: [N, num_letters] posterior; labels: [N] letter indices."""
    if len(probs) != len(labels):
        raise ValueError("probs/labels length mismatch")
    if not labels:
        return float("nan")
    hits = 0
    for p, y in zip(probs, labels):
        top = np.argsort(np.asarray(p))[::-1][:k]
        hits += int(y in top)
    return hits / len(labels)


def confusion_matrix(trues: Sequence[str], preds: Sequence[str],
                     vocab: LetterVocabulary) -> Dict[str, List[int]]:
    """26x26 confusion matrix as {letter: [counts over predicted letters]}."""
    _check_pair(trues, preds)
    index = {l: i for i, l in enumerate(vocab.letters)}
    mat = [[0] * vocab.num_letters for _ in vocab.letters]
    for t, p in zip(trues, preds):
        if t in index and p in index:
            mat[index[t]][index[p]] += 1
    return {l: mat[i] for l, i in index.items()}


def confusion_to_text(matrix: Dict[str, List[int]]) -> str:
    """Plain-text rendering of the confusion matrix (console only, not a UI)."""
    letters = list(matrix.keys())
    header = "     " + " ".join(letters)
    rows = [header]
    for t in letters:
        rows.append(f"{t:>4} " + " ".join(f"{c:>2}" for c in matrix[t]))
    return "\n".join(rows)


def _check_pair(trues, preds):
    if len(trues) != len(preds):
        raise ValueError("trues/preds length mismatch")


def abstention_rate(abstains: Sequence[bool]) -> float:
    if not abstains:
        return float("nan")
    return sum(1 for a in abstains if a) / len(abstains)


def letter_counts(entries) -> Dict[str, int]:
    """Recording counts per letter (class imbalance measurement, §18)."""
    counts: Dict[str, int] = {}
    for e in entries:
        counts[e.letter] = counts.get(e.letter, 0) + 1
    return dict(sorted(counts.items()))
