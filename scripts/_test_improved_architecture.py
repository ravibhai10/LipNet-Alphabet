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

from pipeline.word_recognition.data import WordDataset

cfg = json.loads(Path("pipeline/word_recognition/config.cpu.json").read_text())
root = Path("runs/word_recognition_v1")
words = cfg["words"]

with open(root / "words.jsonl") as f:
    rows = [json.loads(line) for line in f]
train_rows = [r for r in rows if r["split"] == "train"]
valid_rows = [r for r in rows if r["split"] == "valid"]
test_rows = [r for r in rows if r["split"] == "test"]

# Lip-reading architecture with instance normalization and temporal feature aggregation
class ImprovedWordVideoNet(nn.Module):
    def __init__(self, num_classes=4, hidden_size=64, dropout=0.2):
        super().__init__()
        # Conv3D frontend: extract spatio-temporal features
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        ) # [B, 16, 24, 16, 16]
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        ) # [B, 32, 24, 8, 8]
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((24, 2, 2)) # [B, 64, 24, 2, 2]
        )
        
        # Spatial projection to frame feature vector: 64 * 2 * 2 = 256
        self.frame_proj = nn.Linear(64 * 4, hidden_size)
        
        # Bidirectional GRU
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=dropout)
        
        # Classifier with both max and avg pooling: hidden_size * 2 * 2 = hidden_size * 4
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # x: [B, 1, 24, 64, 64]
        # Instance normalization across each sample to remove skin tone / lighting bias
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        std = x.std(dim=(2, 3, 4), keepdim=True) + 1e-6
        x = (x - mean) / std
        
        B = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x) # [B, 64, 24, 2, 2]
        
        # Rearrange to [B, 24, 64 * 2 * 2]
        x = x.permute(0, 2, 1, 3, 4).reshape(B, 24, -1)
        x = self.frame_proj(x) # [B, 24, hidden_size]
        
        gru_out, _ = self.gru(x) # [B, 24, hidden_size * 2]
        
        # Aggregate across time: concatenate mean and max pooling
        avg_pool = gru_out.mean(dim=1)
        max_pool, _ = gru_out.max(dim=1)
        pooled = torch.cat([avg_pool, max_pool], dim=-1) # [B, hidden_size * 4]
        
        logits = self.classifier(pooled)
        return logits

class NormalizedDataset(WordDataset):
    def __getitem__(self, index):
        row = self.rows[index]
        row_id = row["id"]
        if row_id not in self.cache:
            from pipeline.word_recognition.preprocess import read_frames
            clip = read_frames(self.run, row, self.config)
            self.cache[row_id] = torch.from_numpy(clip)
        tensor = self.cache[row_id].float().unsqueeze(0) / 255.0
        return tensor, row["label"], row_id

train_ds = NormalizedDataset(root, train_rows, cfg, training=True)
valid_ds = NormalizedDataset(root, valid_rows, cfg, training=False)
test_ds = NormalizedDataset(root, test_rows, cfg, training=False)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=8, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

torch.manual_seed(42)
model = ImprovedWordVideoNet(num_classes=4, hidden_size=48, dropout=0.2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-4)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print("Testing Improved Architecture on 30 epochs...")
best_val_loss = float("inf")
best_val_acc = 0.0
best_weights = None

start_time = time.perf_counter()
for epoch in range(1, 31):
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

elapsed = time.perf_counter() - start_time
print(f"Completed in {elapsed:.1f}s")
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

print("\n=== TEST RESULTS ON UNSEEN SPEAKERS (s30-s34) ===")
print("Predicted distribution:", dict(Counter([words[p] for p in test_preds])))
print(classification_report(test_targets, test_preds, target_names=words, zero_division=0))
