import json
import torch
import numpy as np
from pathlib import Path
from collections import Counter
import cv2

root = Path("runs/word_recognition_v1")
cfg = json.loads(Path("pipeline/word_recognition/config.cpu.json").read_text())
words = cfg["words"]

# 1. Audit words.jsonl
with open(root / "words.jsonl") as f:
    rows = [json.loads(line) for line in f]

print("=== 1. DATASET MANIFEST AUDIT ===")
print(f"Total rows: {len(rows)}")
for split in ["train", "valid", "test"]:
    split_rows = [r for r in rows if r["split"] == split]
    class_counts = Counter(r["word"] for r in split_rows)
    speakers = sorted(set(r["speaker"] for r in split_rows))
    print(f"Split '{split}': {len(split_rows)} samples across {len(speakers)} speakers: {speakers}")
    print(f"  Class counts: {dict(class_counts)}")

# 2. Check frame pixel statistics across a sample of clips
print("\n=== 2. FRAME PIXEL QUALITY CHECK ===")
prep_root = Path(cfg["processed_root"]) / "80ff4af3decb1bdd"
corrupted = 0
empty = 0
for r in rows:
    dir_path = prep_root / r["frames_path"]
    frames = sorted(dir_path.glob("frame_*.png"))
    if len(frames) != 24:
        corrupted += 1
        continue
    imgs = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in frames]
    if any(img is None for img in imgs):
        corrupted += 1
        continue
    clip = np.stack(imgs)
    if clip.std() < 1.0 or clip.max() == clip.min():
        empty += 1

print(f"Total checked: {len(rows)}, Corrupted count: {corrupted}, Constant/empty count: {empty}")
first_dir = prep_root / rows[0]["frames_path"]
imgs = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in sorted(first_dir.glob("frame_*.png"))]
clip0 = np.stack(imgs)
print(f"Sample 0 ({rows[0]['word']}, {rows[0]['speaker']}): shape={clip0.shape}, min={clip0.min()}, max={clip0.max()}, mean={clip0.mean():.2f}, std={clip0.std():.2f}")

# 3. Model weights and output logits inspection
print("\n=== 3. CHECKPOINT & LOGIT AUDIT ===")
ckpt = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
print(f"Best checkpoint epoch: {ckpt.get('epoch')}, val_loss: {ckpt.get('val_loss')}, val_acc: {ckpt.get('val_acc')}")

import sys
sys.path.insert(0, str(Path(".").resolve()))

from pipeline.word_recognition.model import WordVideoNet
from pipeline.word_recognition.data import WordDataset
from torch.utils.data import DataLoader

model = WordVideoNet(num_classes=len(words), channels=tuple(cfg["model"]["channels"]),
                     hidden_size=cfg["model"]["hidden_size"], layers=cfg["model"]["layers"],
                     dropout=cfg["model"]["dropout"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

for split in ["train", "valid", "test"]:
    s_rows = [r for r in rows if r["split"] == split]
    ds = WordDataset(root, s_rows, cfg, training=False)
    loader = DataLoader(ds, batch_size=len(s_rows), shuffle=False)
    for tensors, labels, ids in loader:
        with torch.no_grad():
            logits = model(tensors)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            
        print(f"\nSplit: {split.upper()} (N={len(labels)})")
        print(f"  Logits mean across samples: {logits.mean(dim=0).numpy().round(4)}")
        print(f"  Logits std across samples:  {logits.std(dim=0).numpy().round(4)}")
        print(f"  Sample 0 logits: {logits[0].numpy().round(4)} -> probs: {probs[0].numpy().round(4)}")
        print(f"  Predicted class distribution: {dict(Counter([words[p.item()] for p in preds]))}")
        acc = (preds == labels).float().mean().item()
        print(f"  Accuracy: {acc*100:.2f}%")
