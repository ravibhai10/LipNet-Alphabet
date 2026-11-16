import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import json
import time
from collections import Counter
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from pipeline.word_recognition.model import WordVideoNet
from pipeline.word_recognition.data import WordDataset

cfg = json.loads(Path("pipeline/word_recognition/config.cpu.json").read_text())
root = Path("runs/word_recognition_v1")
words = cfg["words"]

with open(root / "words.jsonl") as f:
    rows = [json.loads(line) for line in f]
train_rows = [r for r in rows if r["split"] == "train"]
valid_rows = [r for r in rows if r["split"] == "valid"]
test_rows = [r for r in rows if r["split"] == "test"]

# Gentle lip-reading augmentation
def gentle_augment(tensor: torch.Tensor) -> torch.Tensor:
    res = tensor.clone()
    # Mild contrast/brightness adjustment: +- 5%
    scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.10
    bias = (torch.rand(1).item() - 0.5) * 0.05
    res = res * scale + bias
    # Mild spatial jitter: +- 1 pixel with reflection
    dx = torch.randint(-1, 2, (1,)).item()
    dy = torch.randint(-1, 2, (1,)).item()
    if dx != 0 or dy != 0:
        res = torch.roll(res, shifts=(dy, dx), dims=(2, 3))
    return res.clamp(0.0, 1.0)

class CustomWordDataset(WordDataset):
    def __getitem__(self, index):
        row = self.rows[index]
        row_id = row["id"]
        if row_id not in self.cache:
            clip = self.cache_clip(row)
        else:
            clip = self.cache[row_id]
        tensor = clip.float().unsqueeze(0) / 255.0
        if self.training:
            tensor = gentle_augment(tensor)
        return tensor, row["label"], row_id
    
    def cache_clip(self, row):
        from pipeline.word_recognition.preprocess import read_frames
        clip = read_frames(self.run, row, self.config)
        self.cache[row["id"]] = torch.from_numpy(clip)
        return self.cache[row["id"]]

train_ds = CustomWordDataset(root, train_rows, cfg, training=True)
valid_ds = CustomWordDataset(root, valid_rows, cfg, training=False)
test_ds = CustomWordDataset(root, test_rows, cfg, training=False)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=8, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

torch.manual_seed(42)
model = WordVideoNet(num_classes=4, channels=tuple(cfg["model"]["channels"]),
                     hidden_size=cfg["model"]["hidden_size"], layers=cfg["model"]["layers"],
                     dropout=0.10)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-4)

print("Starting 35-epoch trial training with gentle augmentation and CosineAnnealing...")
best_val_loss = float("inf")
best_val_acc = 0.0
best_model_weights = None

start_time = time.perf_counter()
for epoch in range(1, 36):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for tensors, labels, _ in train_loader:
        optimizer.zero_grad()
        logits = model(tensors)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += len(labels)
    scheduler.step()
    
    train_acc = correct / total
    train_loss = total_loss / total
    
    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    all_val_preds, all_val_targets = [], []
    with torch.no_grad():
        for tensors, labels, _ in valid_loader:
            logits = model(tensors)
            loss = criterion(logits, labels)
            val_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            val_correct += (preds == labels).sum().item()
            val_total += len(labels)
            all_val_preds.extend(preds.tolist())
            all_val_targets.extend(labels.tolist())
            
    val_acc = val_correct / val_total
    val_loss = val_loss / val_total
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc = val_acc
        best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        star = " *"
    else:
        star = ""
        
    if epoch % 5 == 0 or epoch == 1 or star:
        pred_dist = dict(Counter([words[p] for p in all_val_preds]))
        print(f"Epoch {epoch:2d}: Train Loss={train_loss:.4f}, Train Acc={train_acc*100:5.1f}% | Val Loss={val_loss:.4f}, Val Acc={val_acc*100:5.1f}% | Val Preds: {pred_dist}{star}")

elapsed = time.perf_counter() - start_time
print(f"\nTraining completed in {elapsed:.1f}s ({elapsed/35:.2f}s/epoch)")
print(f"Best Validation Loss: {best_val_loss:.4f}, Best Validation Acc: {best_val_acc*100:.1f}%")

# Now evaluate best model on held-out test set
model.load_state_dict(best_model_weights)
model.eval()
test_preds, test_targets = [], []
with torch.no_grad():
    for tensors, labels, _ in test_loader:
        logits = model(tensors)
        test_preds.extend(logits.argmax(dim=-1).tolist())
        test_targets.extend(labels.tolist())

print("\n=== HELD-OUT TEST EVALUATION (Speakers s30-s34) ===")
test_pred_dist = dict(Counter([words[p] for p in test_preds]))
print(f"Test Predicted Class Distribution: {test_pred_dist}")
print(classification_report(test_targets, test_preds, target_names=words, zero_division=0))
