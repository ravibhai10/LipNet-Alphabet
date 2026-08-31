"""Phase 5 evaluation: metrics + machine-readable failure records (§21).

For EVERY validation/test recording a record is produced:
    recording_id, speaker_id, true, predicted, correct, top_k, confidence,
    entropy, margin, sequence_length, target_type, split, quality, state
Errors (and optionally all rows) are saved as JSONL — no UI, no plots.

Metrics: letter accuracy, sequence accuracy, macro F1, top-3/top-5,
CER (multi-letter), A-Z confusion matrix (single-letter), ECE,
abstention rate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from alphabet.calibration import expected_calibration_error
from alphabet.decoding import ClassMappingGreedyCTCDecoder, DecodedSequence
from alphabet.targets import TargetType
from alphabet.uncertainty import (
    decide_state, single_letter_uncertainty, sequence_uncertainty,
)
from alphabet.vocab import LetterVocabulary


@dataclass
class EvalRecord:
    recording_id: str
    speaker_id: str
    true: str
    predicted: str
    correct: bool
    confidence: float
    entropy: float
    margin: float
    top_k: List = field(default_factory=list)
    sequence_length: int = 0
    target_type: str = TargetType.SINGLE_LETTER.value
    split: Optional[str] = None
    state: str = "letter"
    quality: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


def _forward_logits(model, fused, mask) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        return model(fused.to(device_of(model)), mask.to(device_of(model)))


def device_of(module) -> torch.device:
    return next(module.parameters()).device
def evaluate_model(
    model,
    loader,
    vocab: LetterVocabulary,
    temperature: float = 1.0,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict:
    """Full evaluation pass; returns {"metrics": ..., "records": [...]}."""
    model.eval()
    decoder = ClassMappingGreedyCTCDecoder(
        index_to_letter=vocab.index_to_letter, blank_index=vocab.blank_index)
    records: List[EvalRecord] = []

    for batch in loader:
        letter_logits = _forward_logits(
            model, batch["fused"], batch["mask"])["letter_logits"]
        mask = batch["mask"]
        input_lengths = batch["input_lengths"]
        decoded = decoder.decode_batch(letter_logits, input_lengths)
        metas = batch["metas"]
        types = [m.target_type for m in metas]

        for i, m in enumerate(metas):
            if types[i] == TargetType.SINGLE_LETTER.value:
                r = single_letter_uncertainty(
                    letter_logits[i:i + 1], mask[i:i + 1], vocab, temperature)[0]
            else:
                r = sequence_uncertainty(
                    letter_logits[i:i + 1], mask[i:i + 1],
                    input_lengths[i:i + 1], [decoded[i].text],
                    vocab, temperature)[0]
            if thresholds:
                r = decide_state(r, thresholds["confidence_threshold"],
                                 thresholds["margin_threshold"])
            records.append(EvalRecord(
                recording_id=m.recording_id, speaker_id=m.speaker_id,
                true=m.label, predicted=r.prediction,
                correct=(r.prediction == m.label),
                confidence=r.confidence, entropy=r.entropy, margin=r.margin,
                top_k=r.top_k, sequence_length=int(input_lengths[i]),
                target_type=types[i], split=m.split, state=r.state.value,
                quality={"frame_count": m.frame_count, **m.quality_summary},
            ))

    from alphabet import metrics as M
    trues = [r.true for r in records]
    preds = [r.predicted for r in records]
    single = [r for r in records if r.target_type == TargetType.SINGLE_LETTER.value]
    multi = [r for r in records if r.target_type == TargetType.MULTI_LETTER.value]

    metrics: Dict = {
        "n": len(records),
        "sequence_accuracy": M.sequence_accuracy(trues, preds),
        "macro_f1": M.macro_f1(trues, preds, vocab),
        "per_type": {},
    }
    if single:
        s_trues = [r.true for r in single]
        s_preds = [r.predicted for r in single]
        s_labels = [vocab.letters.index(r.true) for r in single]
        rows = [_posterior_row(r, vocab) for r in single]
        metrics["per_type"]["single_letter"] = {
            "n": len(single),
            "letter_accuracy": M.letter_accuracy(s_trues, s_preds),
            "macro_f1": M.macro_f1(s_trues, s_preds, vocab),
            "top3_accuracy": M.top_k_accuracy(rows, s_labels, 3),
            "top5_accuracy": M.top_k_accuracy(rows, s_labels, 5),
            "confusion_matrix": M.confusion_matrix(s_trues, s_preds, vocab),
        }
    if multi:
        cer_values = [M.cer(r.true, r.predicted) for r in multi]
        metrics["per_type"]["multi_letter"] = {
            "n": len(multi),
            "sequence_accuracy": M.sequence_accuracy(
                [r.true for r in multi], [r.predicted for r in multi]),
            "cer_mean": float(sum(cer_values) / len(cer_values)),
            "cer_total_edits": int(sum(
                M.levenshtein(r.true, r.predicted) for r in multi)),
        }
    metrics["ece"] = expected_calibration_error(
        [r.confidence for r in records], [r.correct for r in records], 15)["ece"]
    if thresholds:
        metrics["abstention_rate"] = M.abstention_rate([r.abstain for r in records])
        metrics["state_counts"] = _count_states(records)
    return {"metrics": metrics, "records": records}


def _posterior_row(record: EvalRecord, vocab: LetterVocabulary) -> List[float]:
    """Sparse posterior row from top_k entries (for top-k accuracy)."""
    row = [0.0] * vocab.num_letters
    for letter, p in record.top_k:
        row[vocab.letters.index(letter)] = p
    return row


def _count_states(records: Sequence[EvalRecord]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in records:
        counts[r.state] = counts.get(r.state, 0) + 1
    return counts


def write_failure_records(records: Sequence[EvalRecord], path: Path,
                          errors_only: bool = True) -> int:
    """Machine-readable JSONL failure log (§21). Returns rows written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for r in records if (not r.correct) or not errors_only]
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    return len(rows)

