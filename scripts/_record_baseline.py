import json
import hashlib
from pathlib import Path

baseline_ckpt = Path("runs/word_recognition_v1/best_collapsed_baseline.pt")
data = baseline_ckpt.read_bytes()
sha256 = hashlib.sha256(data).hexdigest()

baseline_summary = {
    "checkpoint": "runs/word_recognition_v1/best_collapsed_baseline.pt",
    "sha256": sha256,
    "model_architecture": "WordVideoNet (Conv3D + BiGRU)",
    "failure_mode": "Single-class majority collapse",
    "collapsed_predicted_class": "lay (epoch 8) / bin (earlier run)",
    "held_out_test_accuracy": 0.25,
    "held_out_test_wer": 0.75,
    "held_out_macro_f1": 0.10,
    "confusion_matrix": [
        [0, 8, 0, 0],
        [0, 8, 0, 0],
        [0, 8, 0, 0],
        [0, 8, 0, 0]
    ],
    "per_word_accuracy": {
        "bin": 0.0,
        "lay": 1.0,
        "place": 0.0,
        "set": 0.0
    },
    "diagnosis": "All samples produced almost uniform probabilities ~0.254 due to missing input instance normalization, destructive zero-padded augmentation, and premature early stopping."
}

Path("runs/word_recognition_v1/baseline_collapsed_summary.json").write_text(json.dumps(baseline_summary, indent=2))
print("Saved baseline summary with SHA256:", sha256)
