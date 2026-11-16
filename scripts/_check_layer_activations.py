import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import json
import torch
from pipeline.word_recognition.model import WordVideoNet
from pipeline.word_recognition.data import WordDataset
from torch.utils.data import DataLoader

cfg = json.loads(Path("pipeline/word_recognition/config.cpu.json").read_text())
root = Path("runs/word_recognition_v1")
ckpt = torch.load(root / "best.pt", map_location="cpu", weights_only=False)

model = WordVideoNet(num_classes=4, channels=tuple(cfg["model"]["channels"]),
                     hidden_size=cfg["model"]["hidden_size"], layers=cfg["model"]["layers"],
                     dropout=cfg["model"]["dropout"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

with open(root / "words.jsonl") as f:
    rows = [json.loads(line) for line in f]
train_rows = [r for r in rows if r["split"] == "train"]

ds = WordDataset(root, train_rows[:4], cfg, training=False)
loader = DataLoader(ds, batch_size=4, shuffle=False)

for tensors, labels, ids in loader:
    print("Input tensor: shape=", tensors.shape, "min=", tensors.min().item(), "max=", tensors.max().item(), "mean=", tensors.mean().item(), "std=", tensors.std().item())
    x = tensors
    for i, block in enumerate(model.frontend):
        x = block(x)
        print(f"Block {i+1} output: shape={x.shape}, mean={x.mean().item():.5f}, std={x.std().item():.5f}, max={x.max().item():.5f}")
    
    spatial_pooled = x.mean(dim=(-1, -2)).transpose(1, 2)
    print(f"Spatial pooled: shape={spatial_pooled.shape}, mean={spatial_pooled.mean().item():.5f}, std={spatial_pooled.std().item():.5f}")
    
    gru_out, _ = model.temporal(spatial_pooled)
    print(f"GRU out: shape={gru_out.shape}, mean={gru_out.mean().item():.5f}, std={gru_out.std().item():.5f}")
    
    temporal_pooled = gru_out.mean(dim=1)
    print(f"Temporal pooled: shape={temporal_pooled.shape}, mean={temporal_pooled.mean().item():.5f}, std={temporal_pooled.std().item():.5f}")
    
    # Check linear classifier weights and bias
    linear = model.classifier[1]
    print(f"Classifier Linear weight: shape={linear.weight.shape}, mean={linear.weight.mean().item():.5f}, std={linear.weight.std().item():.5f}")
    print(f"Classifier Linear bias: {linear.bias.detach().numpy().round(4)}")
    
    logits = model.classifier(temporal_pooled)
    print(f"Logits: {logits.detach().numpy().round(4)}")
