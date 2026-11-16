"""Evaluation metrics, diagnostic reports, and confusion matrix export for word-level lip reading."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from typing import Any, List
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.utils.data import DataLoader

from .common import (ROOT, atomic_json, fingerprint, load_config, load_dataset,
                     parser, project_path, run_contract, sha256, timestamp, write_rows)
from .data import WordDataset
from .model import WordVideoNet


def metrics(targets: List[int], predictions: Any, words: List[str]) -> dict:
    if len(targets) != len(predictions):
        raise ValueError("Length mismatch between targets and predictions.")
    if not words or len(words) < 2:
        raise ValueError("At least 2 vocabulary words required.")

    if not targets or len(set(words))!=len(words) or any(type(t) not in (int,np.int32,np.int64) or not 0<=t<len(words) for t in targets):
        raise ValueError('Nonempty valid targets and unique words required.')
    preds = []
    for p in predictions:
        if isinstance(p, (list, tuple, np.ndarray, torch.Tensor)):
            if len(p) != len(words):
                raise ValueError("Prediction dimension must match vocabulary size.")
            values=np.asarray(p)
            if np.issubdtype(values.dtype,np.integer):
                # Historical contract: integer vectors are ranked CLASS IDs.
                if sorted(values.tolist())!=list(range(len(words))):raise ValueError('Class ranking must be a vocabulary permutation.')
                pred_idx=int(values[0])
            else:
                if not np.isfinite(values).all() or (values<0).any() or not np.isclose(values.sum(),1.,atol=1e-5):raise ValueError('Finite normalized probabilities required.')
                pred_idx = int(np.argmax(values))
        elif isinstance(p, (int, np.integer)):
            pred_idx = int(p)
        else:
            raise ValueError(f"Unsupported prediction format: {type(p)}")
        if not 0<=pred_idx<len(words):raise ValueError('Prediction outside vocabulary.')
        preds.append(pred_idx)

    targets_arr = np.array(targets)
    preds_arr = np.array(preds)

    total = len(targets)
    correct = int(np.sum(targets_arr == preds_arr))
    acc = correct / max(1, total)
    wer = 1.0 - acc

    # Per-word stats
    per_word = {}
    for idx, w in enumerate(words):
        mask = (targets_arr == idx)
        w_total = int(np.sum(mask))
        w_correct = int(np.sum((targets_arr == idx) & (preds_arr == idx)))
        per_word[w] = {
            "correct": w_correct,
            "total": w_total,
            "accuracy": float(w_correct / max(1, w_total))
        }

    cm = confusion_matrix(targets, preds, labels=list(range(len(words)))).tolist()
    cls_rep = classification_report(targets, preds, labels=list(range(len(words))), target_names=words, output_dict=True, zero_division=0)

    return {
        "word_accuracy": float(acc),
        "wer": float(wer),
        "per_word": per_word,
        "confusion_matrix": cm,
        "classification_report": cls_rep,
        "vocabulary": words,
        "total_samples": total
    }


def export_matrix(path: Path | str, result: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    words = result["vocabulary"]
    cm = np.array(result["confusion_matrix"])

    # Calculate row percentages (recall)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_pcts = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0

    html = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>Confusion Matrix</title>",
        "<style>",
        "body { font-family: monospace; background: #0c0d10; color: #e0e0e0; padding: 20px; }",
        "table { border-collapse: collapse; margin-top: 15px; }",
        "th, td { border: 1px solid #333; padding: 8px 12px; text-align: center; }",
        "th { background: #1a1b20; color: #8fdfcb; }",
        ".header-pct { font-size: 0.8em; color: #888; }",
        "</style></head><body>",
        f"<h2>Word Lip-Reading Confusion Matrix & Diagnostic Matrix</h2>",
        f"<p>Overall Accuracy: <strong>{result.get('word_accuracy', 0)*100:.2f}%</strong> | WER: <strong>{result.get('wer', 0)*100:.2f}%</strong></p>",
        f"<p><strong>Row percentages</strong> indicate recall distribution for each true word:</p>",
        "<table>",
        "<tr><th>True \\ Pred</th>" + "".join(f"<th>{w}</th>" for w in words) + "<th>Total</th></tr>"
    ]

    for i, w in enumerate(words):
        row_str = f"<tr><td><strong>{w}</strong></td>"
        for j in range(len(words)):
            count = cm[i, j]
            pct = row_pcts[i, j]
            row_str += f"<td>{count}<br><span class='header-pct'>{pct:.1f}%</span></td>"
        row_str += f"<td><strong>{int(row_sums[i, 0])}</strong></td></tr>"
        html.append(row_str)

    html.extend([
        "</table>",
        "</body></html>"
    ])
    path.write_text("\n".join(html), encoding="utf-8")


def evaluate(config: dict, run: Path, split: str = "test") -> dict:
    run = Path(run)
    device = torch.device("cpu")
    info, rows = load_dataset(run, config, verify_pixels=False)
    words = config["words"]

    target_rows = [r for r in rows if r["split"] == split]
    if not target_rows:
        raise ValueError(f"No samples found for split '{split}'")

    best_checkpoint = run / "best.pt"
    if not best_checkpoint.exists():
        best_checkpoint = run / "checkpoints" / "best_word_model.pt"
    if not best_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found at {best_checkpoint}")

    checkpoint_bytes = best_checkpoint.read_bytes()
    ckpt_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
    from .train import validate_checkpoint
    validate_checkpoint(checkpoint,config,info)

    model_cfg = config["model"]
    model = WordVideoNet(
        num_classes=len(words),
        channels=tuple(model_cfg["channels"]),
        hidden_size=model_cfg["hidden_size"],
        layers=model_cfg["layers"],
        dropout=model_cfg["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    dataset = WordDataset(run, target_rows, config, training=False)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    all_targets = []
    all_preds_vec = []
    records = []

    with torch.no_grad():
        for tensors, labels, ids in loader:
            logits = model(tensors.to(device))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for i in range(len(labels)):
                target = int(labels[i])
                pred_idx = int(np.argmax(probs[i]))
                all_targets.append(target)
                all_preds_vec.append(probs[i].tolist())
                records.append({
                    "id": ids[i],
                    "target": words[target],
                    "prediction": words[pred_idx],
                    "prob": float(probs[i][pred_idx]),
                    "correct": bool(target == pred_idx)
                })

    eval_result = metrics(all_targets, all_preds_vec, words)
    eval_result["samples"] = len(target_rows)
    eval_result["split"] = split
    eval_result["checkpoint_sha256"] = ckpt_sha
    eval_result["evaluated_at"] = timestamp()

    out_dir = run / "evaluation" / ckpt_sha[:16]
    eval_result['path']=str(out_dir.resolve())
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "predictions.jsonl", records)
    export_matrix(out_dir / "confusion_matrix.html", eval_result)
    atomic_json(run / "evaluation_summary.json", eval_result)
    return eval_result


def main():
    args = parser(__doc__).parse_args()
    config = load_config(args.config)
    run = run_contract(config, args.run)
    res = evaluate(config, run)
    print(f"Evaluation complete. Accuracy: {res['word_accuracy']*100:.2f}%, WER: {res['wer']*100:.2f}%")


if __name__ == "__main__":
    main()
