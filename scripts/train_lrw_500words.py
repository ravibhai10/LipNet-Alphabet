#!/usr/bin/env python
"""scripts/train_lrw_500words.py - LRW 500-Word Fine-Tuning Script.

High-Accuracy Configuration:
- Label-Smoothing Cross Entropy (label_smoothing=0.1)
- Cosine Annealing Learning Rate Schedule with minimum eta=1e-6
- Spatial, Temporal, and Photometric Video Augmentations
- Saves best checkpoint and history strictly to runs/finetune/lrw_500words/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.schema import NUM_CLASSES, VOCABULARY, get_word
from models.alphabet_model import AlphabetCNNBiLSTM
from pipeline.lrw.loader import LRWDataset
from pipeline.avhubert.classification import load_annotations
from src.video_training import warmup_cosine_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_lrw_500words")

FINETUNE_DIR = PROJECT_ROOT / "runs" / "finetune" / "lrw_500words"


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Execute one training epoch with gradient descent."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for data, target in loader:
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        # Gradient clipping for LSTM stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_size = data.size(0)
        total_loss += loss.item() * batch_size
        _, predicted = torch.max(output.data, 1)
        total_correct += (predicted == target).sum().item()
        total_samples += batch_size

    epoch_loss = total_loss / total_samples if total_samples > 0 else 0.0
    epoch_acc = total_correct / total_samples if total_samples > 0 else 0.0
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on validation or test set."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for data, target in loader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        loss = criterion(output, target)

        batch_size = data.size(0)
        total_loss += loss.item() * batch_size
        _, predicted = torch.max(output.data, 1)
        total_correct += (predicted == target).sum().item()
        total_samples += batch_size

    epoch_loss = total_loss / total_samples if total_samples > 0 else 0.0
    epoch_acc = total_correct / total_samples if total_samples > 0 else 0.0
    return epoch_loss, epoch_acc


def run_training(
    epochs: int = 5,
    batch_size: int = 2,
    lr: float = 1e-4,
    device_name: str = "cpu",
    output_dir: Optional[Path] = None,
    use_mock: bool = False,
    pretrained_backbone: Optional[str] = None,
    data_manifest: Optional[Path] = None,
    warmup_epochs: int = 5,
) -> Dict[str, float]:
    """Run LRW 500-word model training with high-accuracy optimizations."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")
    if not use_mock and data_manifest is None:
        raise ValueError("Provide --data-manifest with real-camera provenance and speaker-disjoint splits. Local LRW development fixtures cannot establish accuracy.")
    records = load_annotations(data_manifest) if not use_mock else None
    output_dir = Path(output_dir) if output_dir else FINETUNE_DIR.parent / ("lrw_500words_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Existing runs are protected; choose a new output directory.")
    device = torch.device(device_name)
    logger.info("Training on device: %s (classes: %d)", device, NUM_CLASSES)

    # Initialize datasets with augment=True for training
    def manifest_samples(split):
        if records is None:
            return None
        from data.schema import WORD_TO_IDX
        return [{"word": row["label"], "label": WORD_TO_IDX[row["label"]], "file_path": Path(row["video"]), "type": "video"} for row in records if row["split"] == split]
    train_dataset = LRWDataset(split="train", use_mock=use_mock, augment=True, samples=manifest_samples("train"))
    valid_dataset = LRWDataset(split="valid", use_mock=use_mock, augment=False, samples=manifest_samples("valid"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Train samples: %d | Valid samples: %d", len(train_dataset), len(valid_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # Initialize model with 500 output classes
    model = AlphabetCNNBiLSTM(num_classes=NUM_CLASSES).to(device)

    # Transfer learning from baseline alphabet encoder if available
    if pretrained_backbone and Path(pretrained_backbone).exists():
        logger.info("Loading pre-trained feature weights from: %s", pretrained_backbone)
        state = torch.load(pretrained_backbone, map_location=device, weights_only=True)
        state_dict = state.get("model_state_dict", state.get("state_dict", state))
        filtered_state = {
            k: v for k, v in state_dict.items()
            if not k.startswith("fc") and not k.startswith("classifier")
        }
        model.load_state_dict(filtered_state, strict=False)
        logger.info("Backbone weights transferred successfully.")

    # High-accuracy training loss & scheduling
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = warmup_cosine_scheduler(optimizer, epochs, warmup_epochs, 1e-6)

    best_val_loss = float("inf")
    history = []

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, valid_loader, criterion, device)
        scheduler.step()

        epoch_duration = time.time() - epoch_start
        logger.info(
            "Epoch [%d/%d] (%.1fs, lr=%.2e) - Train Loss: %.4f, Train Acc: %.2f%% | Val Loss: %.4f, Val Acc: %.2f%%",
            epoch, epochs, epoch_duration, current_lr, train_loss, train_acc * 100.0, val_loss, val_acc * 100.0
        )

        history.append({
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "duration_seconds": epoch_duration,
        })

        # Save epoch checkpoint
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "num_classes": NUM_CLASSES,
        }, checkpoint_path)

        # Update best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = output_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "num_classes": NUM_CLASSES,
            }, best_model_path)
            logger.info("Saved new best model to %s (Val Loss: %.4f)", best_model_path, val_loss)

    total_time = time.time() - start_time
    summary_path = output_dir / "training_summary.json"
    summary = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "label_smoothing": 0.1,
        "scheduler": "warmup_then_cosine",
        "warmup_epochs": min(warmup_epochs, max(0, epochs - 1)),
        "evaluation_kind": "synthetic_fixture" if use_mock else "speaker_disjoint_real_camera",
        "best_val_loss": best_val_loss,
        "total_time_seconds": total_time,
        "history": history,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Training complete in %.1f seconds. Summary saved to %s", total_time, summary_path)
    return summary


def main():
    parser = argparse.ArgumentParser(description="LRW 500-Word Fine-Tuning Pipeline")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--output-dir", type=Path, default=None, help="New checkpoint output folder")
    parser.add_argument("--data-manifest", type=Path, help="Audited JSONL real-data annotation manifest")
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--use-mock", action="store_true", help="Force mock samples for testing")
    parser.add_argument("--pretrained-backbone", type=str, default="models/alphabet_classifier.pt",
                        help="Optional pre-trained backbone checkpoint to initialize weights")
    args = parser.parse_args()

    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_name=args.device,
        output_dir=args.output_dir,
        use_mock=args.use_mock,
        pretrained_backbone=args.pretrained_backbone if Path(args.pretrained_backbone).exists() else None,
        data_manifest=args.data_manifest,
        warmup_epochs=args.warmup_epochs,
    )


if __name__ == "__main__":
    main()
