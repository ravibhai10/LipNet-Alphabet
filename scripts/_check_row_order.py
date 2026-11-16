import json
from pathlib import Path

root = Path("runs/word_recognition_v1")
with open(root / "words.jsonl") as f:
    rows = [json.loads(line) for line in f]

train_rows = [r for r in rows if r["split"] == "train"]
print(f"Total train rows: {len(train_rows)}")
print("First 30 train labels and words:")
for i, r in enumerate(train_rows[:30]):
    print(f"Index {i:2d}: word={r['word']}, label={r['label']}, speaker={r['speaker']}")
