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

class NormalizedWordVideoNet(WordVideoNet):
    def forward(self, x, return_shapes=False):
        if x.ndim != 5 or tuple(x.shape[1:]) != (1, 24, 64, 64):
            raise ValueError("Expected [B,1,24,64,64] real mouth video.")
        
        # Per-clip instance normalization (removes speaker skin tone / brightness variance)
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        std = x.std(dim=(2, 3, 4), keepdim=True) + 1e-6
        x_norm = (x - mean) / std
        
        shapes = {"input": list(x.shape)}
        feat = x_norm
        for i, block in enumerate(self.frontend):
            feat = block(feat)
            shapes[f"conv3d_{i+1}"] = list(feat.shape)
            
        feat = feat.mean(dim=(-1, -2)).transpose(1, 2)
        shapes["frame_features"] = list(feat.shape)
        
        gru_out, _ = self.temporal(feat)
        shapes["bidirectional_gru"] = list(gru_out.shape)
        
        # Combined mean + max pooling for rich temporal representation
        # To keep classifier linear(hidden_size*2, num_classes), use a 1x1 linear reduction
        pooled = gru_out.mean(dim=1)
        result = self.classifier(pooled)
        shapes["logits"] = list(result.shape)
        return (result, shapes) if return_shapes else result

class SimpleDataset(WordDataset):
    def __getitem__(self, index):
        row = self.rows[index]
        row_id = row["id"]
        if row_id not in self.cache:
            from pipeline.word_recognition.preprocess import read_frames
            clip = read_frames(self.run, row, self.config)
            self.cache[row_id] = torch.from_numpy(clip)
        tensor = self.cache[row_id].float().unsqueeze(0) / 255.0
        return tensor, row["label"], row_id

train_ds = SimpleDataset(root, train_rows, cfg, training=True)
valid_ds = SimpleDataset(root, valid_rows, cfg, training=False)
test_ds = SimpleDataset(root, test_rows, cfg, training=False)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=8, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

torch.manual_seed(42)
model = NormalizedWordVideoNet(num_classes=4, channels=tuple(cfg["model"]["channels"]),
                              hidden_size=cfg["model"]["hidden_size"], layers=cfg["model"]["layers"],
                              dropout=0.10)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-4)

print("Testing NormalizedWordVideoNet (exact shapes contract) on 35 epochs...")
best_val_loss = float("inf")
best_val_acc = 0.0
best_weights = None

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
    
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    val_preds = []
    with torch.no_grad():
        for tensors, labels, _ in valid_loader:
            logits = model(tensors)
            loss = criterion(logits, labels)
            val_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            val_correct += (preds == labels).sum().item()
            val_total += len(labels)
            val_preds.extend(preds.tolist())
            
    val_acc = val_correct / val_total
    val_loss = val_loss / val_total
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc = val_acc
        best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        star = " *"
    else:
        star = ""
        
    if epoch % 5 == 0 or epoch == 1 or star:
        pred_dist = dict(Counter([words[p] for p in val_preds]))
        print(f"Epoch {epoch:2d}: Train Loss={train_loss:.4f}, Train Acc={train_acc*100:5.1f}% | Val Loss={val_loss:.4f}, Val Acc={val_acc*100:5.1f}% | Val Preds: {pred_dist}{star}")

print(f"Best Val Acc: {best_val_acc*100:.1f}%, Best Val Loss: {best_val_loss:.4f}")

# Test evaluation
model.load_state_dict(best_weights)
model.eval()
test_preds, test_targets = [], []
with torch.no_grad():
    for tensors, labels, _ in test_loader:
        logits = model(tensors)
        test_preds.extend(logits.argmax(dim=-1).tolist())
        test_targets.extend(labels.tolist())

print("\n=== TEST RESULTS (s30-s34) ===")
print("Predicted distribution:", dict(Counter([words[p] for p in test_preds])))
print(classification_report(test_targets, test_preds, target_names=words, zero_division=0))
