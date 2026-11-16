import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

from pipeline.word_recognition.model import WordVideoNet
from pipeline.word_recognition.data import WordDataset

# Revert config to exact original
orig_cfg = {
  "dataset": "grid_aligned_whole_words",
  "corpus_root": "data/external/grid",
  "source_inventory": "data/processed/grid_multispeaker_v1",
  "processed_root": "data/processed/grid_whole_words",
  "words": ["bin", "lay", "place", "set"],
  "quotas": {"train": 24, "valid": 8, "test": 8},
  "minimum_speakers": {"train": 3, "valid": 2, "test": 2},
  "preprocessing": {"version": "grid-whole-word-mouth-v1", "frames": 24, "size": 64, "grayscale": True},
  "model": {"channels": [8, 16, 32], "hidden_size": 48, "layers": 2, "dropout": 0.15},
  "training": {"epochs": 12, "batch_size": 4, "lr": 0.001, "weight_decay": 0.0001, "label_smoothing": 0.1, "patience": 4, "min_delta": 0.0001, "threads": 2, "seed": 42, "max_minutes": 12},
  "reserve_gib": 20,
  "download_max_minutes": 10
}
Path("pipeline/word_recognition/config.cpu.json").write_text(json.dumps(orig_cfg, indent=2))

root = Path("runs/word_recognition_v1")
with open(root / "words.jsonl") as f:
    rows = [json.loads(line) for line in f]
train_rows = [r for r in rows if r["split"] == "train"]
valid_rows = [r for r in rows if r["split"] == "valid"]
test_rows = [r for r in rows if r["split"] == "test"]

train_ds = WordDataset(root, train_rows, orig_cfg, training=True)
valid_ds = WordDataset(root, valid_rows, orig_cfg, training=False)
test_ds = WordDataset(root, test_rows, orig_cfg, training=False)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=4, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

torch.manual_seed(42)
model = WordVideoNet(4, channels=(8, 16, 32), hidden_size=48, layers=2, dropout=0.15)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05) # mild smoothing
optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)

print("Training 12 epochs with original config parameters & normalized WordVideoNet...")
best_val_loss = float("inf")
best_weights = None

for epoch in range(1, 13):
    model.train()
    tot_loss, correct, tot = 0.0, 0, 0
    for tensors, labels, _ in train_loader:
        optimizer.zero_grad()
        logits = model(tensors)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        tot += len(labels)
    
    train_acc = correct / tot
    train_loss = tot_loss / tot
    
    model.eval()
    val_loss, val_correct, val_tot = 0.0, 0, 0
    val_preds = []
    with torch.no_grad():
        for tensors, labels, _ in valid_loader:
            logits = model(tensors)
            loss = criterion(logits, labels)
            val_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            val_correct += (preds == labels).sum().item()
            val_tot += len(labels)
            val_preds.extend(preds.tolist())
            
    val_acc = val_correct / val_tot
    val_loss = val_loss / val_tot
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        star = " *"
    else:
        star = ""
        
    pred_dist = dict(Counter([orig_cfg["words"][p] for p in val_preds]))
    print(f"Epoch {epoch:2d}: Train Loss={train_loss:.4f}, Train Acc={train_acc*100:5.1f}% | Val Loss={val_loss:.4f}, Val Acc={val_acc*100:5.1f}% | Val Preds: {pred_dist}{star}")

# Held-out Test
model.load_state_dict(best_weights)
model.eval()
test_preds, test_targets = [], []
with torch.no_grad():
    for tensors, labels, _ in test_loader:
        logits = model(tensors)
        test_preds.extend(logits.argmax(dim=-1).tolist())
        test_targets.extend(labels.tolist())

print("\n=== HELD-OUT TEST (s30-s34) ===")
print("Predictions:", dict(Counter([orig_cfg["words"][p] for p in test_preds])))
print(classification_report(test_targets, test_preds, target_names=orig_cfg["words"], zero_division=0))
