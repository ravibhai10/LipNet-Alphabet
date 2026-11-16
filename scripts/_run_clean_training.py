import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

from pipeline.word_recognition.data import WordDataset

cfg = json.loads(Path("pipeline/word_recognition/config.cpu.json").read_text())
root = Path("runs/word_recognition_v1")
words = cfg["words"]

with open(root / "words.jsonl") as f:
    rows = [json.loads(line) for line in f]
train_rows = [r for r in rows if r["split"] == "train"]
valid_rows = [r for r in rows if r["split"] == "valid"]
test_rows = [r for r in rows if r["split"] == "test"]

class NormalizedWordVideoNet(nn.Module):
    def __init__(self, num_classes, channels=(8, 16, 32), hidden_size=48, layers=2, dropout=.15):
        super().__init__()
        if num_classes < 2 or len(channels) != 3 or any(c < 1 for c in channels) or hidden_size < 1 or layers < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid word model configuration.")
        blocks, previous = [], 1
        for channels_out in channels:
            blocks.append(nn.Sequential(
                nn.Conv3d(previous, channels_out, 3, stride=(1, 2, 2), padding=1, bias=False),
                nn.GroupNorm(1, channels_out),
                nn.SiLU()
            ))
            previous = channels_out
        self.frontend = nn.ModuleList(blocks)
        self.temporal = nn.GRU(previous, hidden_size, layers, batch_first=True, bidirectional=True,
                               dropout=dropout if layers > 1 else 0.)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size * 2, num_classes))

    def forward(self, x, return_shapes=False):
        if x.ndim != 5 or tuple(x.shape[1:]) != (1, 24, 64, 64):
            raise ValueError("Expected [B,1,24,64,64] real mouth video.")
        
        # Instance normalization across frames and pixels to eliminate speaker illumination bias
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        std = x.std(dim=(2, 3, 4), keepdim=True) + 1e-6
        x = (x - mean) / std
        
        shapes = {"input": list(x.shape)}
        for i, block in enumerate(self.frontend):
            x = block(x)
            shapes[f"conv3d_{i+1}"] = list(x.shape)
        x = x.mean(dim=(-1, -2)).transpose(1, 2)
        shapes["frame_features"] = list(x.shape)
        x, _ = self.temporal(x)
        shapes["bidirectional_gru"] = list(x.shape)
        result = self.classifier(x.mean(dim=1))
        shapes["logits"] = list(result.shape)
        return (result, shapes) if return_shapes else result

class CleanAugmentDataset(WordDataset):
    def __getitem__(self, index):
        row = self.rows[index]
        row_id = row["id"]
        if row_id not in self.cache:
            from pipeline.word_recognition.preprocess import read_frames
            clip = read_frames(self.run, row, self.config)
            self.cache[row_id] = torch.from_numpy(clip)
        tensor = self.cache[row_id].float().unsqueeze(0) / 255.0
        
        if self.training:
            # Safe spatial roll (circular shift preserves all lips/pixels without black borders)
            if torch.rand(1).item() < 0.5:
                dy = torch.randint(-1, 2, (1,)).item()
                dx = torch.randint(-1, 2, (1,)).item()
                tensor = torch.roll(tensor, shifts=(dy, dx), dims=(2, 3))
            # Subtle contrast variation
            if torch.rand(1).item() < 0.5:
                scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.1
                tensor = (tensor * scale).clamp(0.0, 1.0)
                
        return tensor, row["label"], row_id

train_ds = CleanAugmentDataset(root, train_rows, cfg, training=True)
valid_ds = CleanAugmentDataset(root, valid_rows, cfg, training=False)
test_ds = CleanAugmentDataset(root, test_rows, cfg, training=False)

train_loader = DataLoader(train_ds, batch_size=6, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=8, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

torch.manual_seed(42)
model = NormalizedWordVideoNet(num_classes=4, channels=tuple(cfg["model"]["channels"]),
                              hidden_size=cfg["model"]["hidden_size"], layers=cfg["model"]["layers"],
                              dropout=0.10)

criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-4)

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
        
    if epoch % 5 == 0 or star:
        pred_dist = dict(Counter([words[p] for p in val_preds]))
        print(f"Epoch {epoch:2d}: Train Loss={train_loss:.4f}, Train Acc={train_acc*100:5.1f}% | Val Loss={val_loss:.4f}, Val Acc={val_acc*100:5.1f}% | Val Preds: {pred_dist}{star}")

print(f"\nBest Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc*100:.1f}%")

# Test Evaluation
model.load_state_dict(best_weights)
model.eval()
test_preds, test_targets, test_probs = [], [], []
with torch.no_grad():
    for tensors, labels, _ in test_loader:
        logits = model(tensors)
        probs = torch.softmax(logits, dim=-1)
        test_preds.extend(logits.argmax(dim=-1).tolist())
        test_targets.extend(labels.tolist())
        test_probs.extend(probs.tolist())

cm = confusion_matrix(test_targets, test_preds)
top1_acc = (torch.tensor(test_preds) == torch.tensor(test_targets)).float().mean().item()

# Top 2 accuracy
top2_correct = 0
for i in range(len(test_targets)):
    sorted_classes = torch.tensor(test_probs[i]).argsort(descending=True)[:2].tolist()
    if test_targets[i] in sorted_classes:
        top2_correct += 1
top2_acc = top2_correct / len(test_targets)

print("\n=== HELD-OUT TEST EVALUATION (s30-s34) ===")
print(f"Top-1 Accuracy: {top1_acc*100:.2f}%")
print(f"Top-2 Accuracy: {top2_acc*100:.2f}%")
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(test_targets, test_preds, target_names=words, zero_division=0))
