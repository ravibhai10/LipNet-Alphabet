import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import json
import torch
from torch import nn
from pipeline.word_recognition.model import WordVideoNet
from pipeline.word_recognition.data import WordDataset
from torch.utils.data import DataLoader

cfg = json.loads(Path("pipeline/word_recognition/config.cpu.json").read_text())
root = Path("runs/word_recognition_v1")
train_cfg = cfg["training"]

with open(root / "words.jsonl") as f:
    rows = [json.loads(line) for line in f]
train_rows = [r for r in rows if r["split"] == "train"]
valid_rows = [r for r in rows if r["split"] == "valid"]

train_ds = WordDataset(root, train_rows, cfg, training=True)
valid_ds = WordDataset(root, valid_rows, cfg, training=False)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=4, shuffle=False)

model = WordVideoNet(num_classes=4, channels=tuple(cfg["model"]["channels"]),
                     hidden_size=cfg["model"]["hidden_size"], layers=cfg["model"]["layers"],
                     dropout=cfg["model"]["dropout"])

criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg.get("label_smoothing", 0.0))
optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])

print("Starting trial training (3 epochs)...")
for epoch in range(1, 4):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for tensors, labels, _ in train_loader:
        optimizer.zero_grad()
        logits = model(tensors)
        loss = criterion(logits, labels)
        loss.backward()
        
        # Check grad norm of last layer
        grad_norm = model.classifier[1].weight.grad.norm().item()
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += len(labels)
        
    train_acc = correct / total
    train_loss = total_loss / total
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for tensors, labels, _ in valid_loader:
            logits = model(tensors)
            loss = criterion(logits, labels)
            val_loss += loss.item() * len(labels)
            val_correct += (logits.argmax(dim=-1) == labels).sum().item()
            val_total += len(labels)
            
    val_acc = val_correct / val_total
    val_loss = val_loss / val_total
    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.1f}%, Val Loss={val_loss:.4f}, Val Acc={val_acc*100:.1f}%, Last Grad Norm={grad_norm:.6f}")
