"""
Comprehensive Smoke Test for Model Checkpoints and Inference Pipelines.
"""
import sys
import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.alphabet_model import AlphabetCNNBiLSTM
from app.services.browser_inference import browser_inference

print("=== 1. Testing models/alphabet_classifier.pt (26-class baseline) ===")
ckpt_path = PROJECT_ROOT / "models" / "alphabet_classifier.pt"
assert ckpt_path.exists(), f"Missing {ckpt_path}"
state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
model_26 = AlphabetCNNBiLSTM(num_classes=26)
model_26.load_state_dict(state["model_state_dict"])
model_26.eval()
print(f"Loaded successfully. Parameters: {sum(p.numel() for p in model_26.parameters()):,}")

# Forward pass with synthetic test tensor of shape (1, 30, 1, 140, 46)
dummy_clip = torch.randn(1, 30, 1, 140, 46)
with torch.no_grad():
    logits = model_26(dummy_clip)
    probs = torch.softmax(logits, dim=-1)
    top_prob, top_idx = torch.topk(probs, k=3)

print("Forward shape:", logits.shape)
assert logits.shape == (1, 26), f"Expected (1, 26), got {logits.shape}"
top_letters = [chr(65 + i) for i in top_idx[0].tolist()]
top_scores = [round(p, 4) for p in top_prob[0].tolist()]
print(f"Top-3 predictions: {list(zip(top_letters, top_scores))}")
print("Test 1 PASS: models/alphabet_classifier.pt forward pass works.")

print("\n=== 2. Testing BrowserInference Service ===")
ready = browser_inference.load()
print("browser_inference.load():", ready)
print("browser_inference.health():", browser_inference.health())
assert ready, "browser_inference failed to load"
print("Test 2 PASS: BrowserInference service loads 26-class model.")

print("\n=== 3. Testing runs/finetune/lrw_500words/best_model.pt (500-class experimental) ===")
lrw_ckpt = PROJECT_ROOT / "runs" / "finetune" / "lrw_500words" / "best_model.pt"
if lrw_ckpt.exists():
    lrw_state = torch.load(lrw_ckpt, map_location="cpu", weights_only=False)
    print("LRW checkpoint keys:", list(lrw_state.keys()))
    model_500 = AlphabetCNNBiLSTM(num_classes=500)
    model_500.load_state_dict(lrw_state["model_state_dict"])
    model_500.eval()
    print(f"Loaded successfully. Parameters: {sum(p.numel() for p in model_500.parameters()):,}")
    
    # Forward pass
    with torch.no_grad():
        dummy_lrw = torch.randn(1, 30, 1, 140, 46)
        logits_500 = model_500(dummy_lrw)
        probs_500 = torch.softmax(logits_500, dim=-1)
        top5_p, top5_i = torch.topk(probs_500, k=5)
    print("Forward shape:", logits_500.shape)
    assert logits_500.shape == (1, 500), f"Expected (1, 500), got {logits_500.shape}"
    print("Top-5 indices:", top5_i[0].tolist(), "Probabilities:", [round(p, 4) for p in top5_p[0].tolist()])
    print("Test 3 PASS: 500-class model is loadable and can perform forward pass.")
else:
    print("LRW checkpoint not found on disk.")

print("\n=== ALL AUDIT SMOKE TESTS COMPLETED ===")
