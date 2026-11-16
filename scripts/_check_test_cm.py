import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import json
import torch
from sklearn.metrics import confusion_matrix
from scripts._test_improved_architecture import ImprovedWordVideoNet, NormalizedDataset, test_rows, cfg, root, words
from torch.utils.data import DataLoader

test_ds = NormalizedDataset(root, test_rows, cfg, training=False)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

# Let's inspect test predictions with probabilities and top-2 accuracy
