"""Per-class classification evidence, with rows=true and columns=predicted."""
import csv
import json
from pathlib import Path
import numpy as np


def classification_report(targets, predictions, vocabulary):
    labels = list(vocabulary)
    if not labels or len(set(labels)) != len(labels) or len(targets) != len(predictions) or not len(targets):
        raise ValueError("Nonempty paired predictions and a unique vocabulary are required.")
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for truth, predicted in zip(targets, predictions):
        if not 0 <= truth < len(labels) or not 0 <= predicted < len(labels):
            raise ValueError("A class index is outside the vocabulary.")
        matrix[truth, predicted] += 1
    per_class = {}
    for index, label in enumerate(labels):
        tp, support, emitted = int(matrix[index, index]), int(matrix[index].sum()), int(matrix[:, index].sum())
        precision, recall = tp / emitted if emitted else 0., tp / support if support else 0.
        per_class[label] = {"precision": precision, "recall": recall,
                            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
                            "support": support}
    return {"samples": len(targets), "accuracy": float(matrix.trace() / matrix.sum()),
            "macro_f1": float(np.mean([row["f1"] for row in per_class.values()])),
            "vocabulary": labels, "per_class": per_class, "confusion_matrix": matrix.tolist(),
            "confusion_matrix_axes": {"rows": "true", "columns": "predicted"}, "zero_division": 0}


def write_classification_report(output, report):
    output = Path(output)
    (output / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true / predicted", *report["vocabulary"]])
        for label, values in zip(report["vocabulary"], report["confusion_matrix"]):
            writer.writerow([label, *values])
