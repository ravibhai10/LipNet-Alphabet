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

with open(root / "words.jsonl") as f:
    rows = [json.loads(line) for line in f]
train_rows = [r for r in rows if r["split"] == "train"]

# Select 2 of each class:
balanced_rows = [r for r in train_rows if r["label"] == 0][:2] + \
                [r for r in train_rows if r["label"] == 1][:2] + \
                [r for r in train_rows if r["label"] == 2][:2] + \
                [r for r in train_rows if r["label"] == 3][:2]

ds = WordDataset(root, balanced_rows, cfg, training=False)
loader = DataLoader(ds, batch_size=8, shuffle=False)

for tensors, labels, ids in loader:
    break

print("Labels:", labels.tolist())
model = WordVideoNet(num_classes=4, channels=tuple(cfg["model"]["channels"]),
                     hidden_size=cfg["model"]["hidden_size"], layers=cfg["model"]["layers"],
                     dropout=0.0)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(101):
    optimizer.zero_grad()
    logits = model(tensors)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    preds = logits.argmax(dim=-1)
    acc = (preds == labels).float().mean().item()
    if step % 20 == 0 or step == 100:
        print(f"Step {step:3d}: Loss={loss.item():.4f}, Acc={acc*100:5.1f}%, Preds={preds.tolist()}")
