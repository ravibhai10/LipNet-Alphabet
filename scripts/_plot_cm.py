import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

summary = json.loads(Path("runs/word_recognition_v1/evaluation_summary.json").read_text())
cm = np.array(summary["confusion_matrix"])
words = ["bin", "lay", "place", "set"]

fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.matshow(cm, cmap="Blues", alpha=0.8)

for i in range(len(words)):
    for j in range(len(words)):
        ax.text(j, i, str(cm[i, j]), va="center", ha="center", fontsize=12, fontweight="bold")

fig.colorbar(cax)
ax.set_xticks(range(len(words)))
ax.set_yticks(range(len(words)))
ax.set_xticklabels(words)
ax.set_yticklabels(words)
ax.set_xlabel("Predicted Word", labelpad=10, fontweight="bold")
ax.set_ylabel("True Word", labelpad=10, fontweight="bold")
ax.set_title("Held-out Test Confusion Matrix (Speakers s30-s34)\nTop-1 Accuracy: 34.38%", pad=15, fontweight="bold")

plt.tight_layout()
diag_dir = Path("runs/word_recognition_v1/diagnostics")
diag_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(diag_dir / "confusion_matrix.png", dpi=150)
print("Saved confusion_matrix.png successfully!")
