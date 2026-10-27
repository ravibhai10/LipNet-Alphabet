"""Training script for Alphabet Lip-Reading model.

Loads clips from `dataset/alphabet/`, 80/10/10 train/val/test split,
applies augmentation, trains CNN-BiLSTM for up to 50 epochs with early
stopping (patience=10), saves best model to a new run directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models"))
from alphabet_model import AlphabetCNNBiLSTM, T, H, W, NUM_CLASSES
from src.video_training import augment_video, warmup_cosine_scheduler

LOGGER = logging.getLogger(__name__)
ALPHABET = [chr(ord("A") + i) for i in range(26)]
LABEL_TO_IDX = {l: i for i, l in enumerate(ALPHABET)}
IDX_TO_LABEL = {i: l for l, i in LABEL_TO_IDX.items()}


class AlphabetDataset(Dataset):
    """Loads (T, H, W) numpy clips, returns (1, T, H, W) float tensor."""

    def __init__(self, root: Path, file_paths: List[Tuple[Path, int]],
                 augment: bool = False) -> None:
        self.root = root
        self.file_paths = file_paths
        self.augment = augment

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.file_paths[idx]
        clip = np.load(path).astype(np.float32) / 255.0
        clip = np.expand_dims(clip, axis=1)  # (T,1,H,W)
        tensor = torch.from_numpy(clip)
        return (augment_video(tensor) if self.augment else tensor), label


def load_dataset(root: Path) -> List[Tuple[Path, int]]:
    paths = []
    for letter in ALPHABET:
        ld = root / letter
        if not ld.is_dir():
            LOGGER.warning("Missing letter dir: %s", ld)
            continue
        for npy in sorted(ld.glob("clip_*.npy")):
            paths.append((npy, LABEL_TO_IDX[letter]))
    LOGGER.info("Loaded %d clips from %s", len(paths), root)
    return paths


def load_manifest_dataset(root: Path) -> list[tuple[Path, int, str]]:
    """Load processed clips and speaker IDs from the preprocessing manifest."""
    manifest_path = root / "alphabet_manifest.jsonl"
    if not manifest_path.is_file():
        return []
    samples = []
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            letter = str(record["letter"]).upper()
            speaker = str(record["speaker_id"])
            clip_path = root / str(record["clip_path"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid alphabet manifest line {line_number}: {exc}"
            ) from exc
        if letter not in LABEL_TO_IDX or not clip_path.is_file():
            raise ValueError(
                f"invalid alphabet manifest line {line_number}: "
                f"letter={letter!r}, clip={clip_path}"
            )
        samples.append((clip_path, LABEL_TO_IDX[letter], speaker))
    return samples


def split_by_speaker(
    samples: list[tuple[Path, int, str]],
    *,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Split samples by speaker, never by individual recording."""
    if not samples:
        raise ValueError("cannot split an empty sample list")
    speakers = sorted({sample[2] for sample in samples})
    if len(speakers) < 3:
        raise ValueError(
            "speaker-independent train/val/test splitting requires at least "
            "three speakers"
        )
    rng = random.Random(seed)
    rng.shuffle(speakers)
    n_train = max(1, int(len(speakers) * train_ratio))
    n_val = max(1, int(len(speakers) * val_ratio))
    if n_train + n_val >= len(speakers):
        n_train = max(1, len(speakers) - 2)
        n_val = 1
    assignments = {
        speaker: "train" if index < n_train else
        "val" if index < n_train + n_val else "test"
        for index, speaker in enumerate(speakers)
    }
    result = {"train": [], "val": [], "test": []}
    for path, label, speaker in samples:
        result[assignments[speaker]].append((path, label))
    return result["train"], result["val"], result["test"]


def time_mask(x, max_len=5, p=0.3):
    """Drop up to max_len consecutive frames with probability p."""
    if random.random() >= p:
        return x
    T_ = x.shape[1]
    l = random.randint(1, min(max_len, T_ - 1))
    t = random.randint(0, T_ - l)
    x = x.clone()
    x[:, t:t + l] = 0
    return x


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for clips, labels in loader:
        clips = clips.to(device); labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(clips)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * clips.size(0)
        correct += (logits.argmax(-1) == labels).sum().item()
        total += clips.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for clips, labels in loader:
        clips = clips.to(device); labels = labels.to(device)
        logits = model(clips)
        loss = criterion(logits, labels)
        total_loss += loss.item() * clips.size(0)
        correct += (logits.argmax(-1) == labels).sum().item()
        total += clips.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def per_class_acc(model, loader, device):
    model.eval()
    counts = Counter(); correct = Counter()
    for clips, labels in loader:
        preds = model(clips.to(device)).argmax(-1).cpu().tolist()
        for p, l in zip(preds, labels.tolist()):
            counts[l] += 1
            if p == l: correct[l] += 1
    return {IDX_TO_LABEL[i]: correct[i] / counts[i]
            for i in range(NUM_CLASSES) if counts[i]}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path,
                   default=PROJECT_ROOT / "dataset" / "alphabet")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", type=Path,
                   default=PROJECT_ROOT / "runs" / "finetune" / "alphabet_classifier_cpu")
    p.add_argument("--init-checkpoint", type=Path, default=None,
                   help="Existing checkpoint to fine-tune; never overwritten")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError("Choose a new --out-dir; existing trained artifacts are protected.")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")
    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if args.device == "auto" and
                          torch.cuda.is_available() else
                          (args.device if args.device != "auto" else "cpu"))
    LOGGER.info("Device: %s | PyTorch: %s", device, torch.__version__)
    manifest_samples = load_manifest_dataset(args.dataset)
    if manifest_samples:
        tr, va, te = split_by_speaker(manifest_samples, seed=args.seed)
        LOGGER.info(
            "Speaker-independent split from manifest: train=%d val=%d test=%d",
            len(tr), len(va), len(te),
        )
    else:
        all_paths = load_dataset(args.dataset)
        if not all_paths:
            LOGGER.error("No clips found in %s", args.dataset); return 1
        LOGGER.warning(
            "No alphabet_manifest.jsonl found; using legacy random sample split. "
            "This is not speaker-independent."
        )
        random.shuffle(all_paths)
        n = len(all_paths)
        n_tr, n_v = int(0.8 * n), int(0.1 * n)
        tr, va, te = all_paths[:n_tr], all_paths[n_tr:n_tr + n_v], all_paths[n_tr + n_v:]
    LOGGER.info("Split: train=%d val=%d test=%d", len(tr), len(va), len(te))
    if not tr or not va or not te:
        raise ValueError("Training, validation, and test splits must all contain samples.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tr_ds = AlphabetDataset(args.dataset, tr, augment=True)
    va_ds = AlphabetDataset(args.dataset, va, augment=False)
    te_ds = AlphabetDataset(args.dataset, te, augment=False)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size)
    te_ld = DataLoader(te_ds, batch_size=args.batch_size)
    model = AlphabetCNNBiLSTM().to(device)
    if args.init_checkpoint is not None:
        if not args.init_checkpoint.is_file():
            LOGGER.error("Initial checkpoint not found: %s", args.init_checkpoint)
            return 1
        checkpoint = torch.load(
            args.init_checkpoint, map_location=device, weights_only=False
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        LOGGER.info("Initialized from protected checkpoint: %s", args.init_checkpoint)
    LOGGER.info("Params: %d", model.num_parameters())
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = warmup_cosine_scheduler(optimizer, args.epochs, args.warmup_epochs, args.min_lr)
    best_val_acc, patience_ctr, t0 = -1.0, 0, time.time()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, args.epochs + 1):
        t_l, t_a = train_epoch(model, tr_ld, optimizer, criterion, device)
        v_l, v_a = evaluate(model, va_ld, criterion, device)
        scheduler.step()
        history["train_loss"].append(round(t_l, 4))
        history["train_acc"].append(round(t_a, 4))
        history["val_loss"].append(round(v_l, 4))
        history["val_acc"].append(round(v_a, 4))
        LOGGER.info("E%d/%d | tr %.4f/%.1f%% | va %.4f/%.1f%% | best=%.1f%%",
                   epoch, args.epochs, t_l, t_a * 100, v_l, v_a * 100,
                   best_val_acc * 100)
        if v_a > best_val_acc:
            best_val_acc = v_a; patience_ctr = 0
            torch.save({"model_state_dict": model.state_dict(),
                       "config": {"num_classes": NUM_CLASSES, "t": T, "h": H, "w": W}},
                      args.out_dir / "alphabet_classifier.pt")
            LOGGER.info("  -> Best model (val=%.1f%%)", v_a * 100)
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                LOGGER.info("Early stopping at epoch %d", epoch); break
    elapsed = time.time() - t0
    LOGGER.info("Training done in %.0fs", elapsed)
    ck = torch.load(args.out_dir / "alphabet_classifier.pt",
                     map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    te_loss, te_acc = evaluate(model, te_ld, criterion, device)
    LOGGER.info("=== TEST === loss=%.4f acc=%.2f%%", te_loss, te_acc * 100)
    pca = per_class_acc(model, te_ld, device)
    for letter in ALPHABET:
        LOGGER.info("  %s: %.1f%%", letter, pca.get(letter, 0.0) * 100)
    summary = {"test_accuracy": round(te_acc, 4), "test_loss": round(te_loss, 4),
               "epochs_run": len(history["train_loss"]),
               "best_val_accuracy": round(best_val_acc, 4),
               "elapsed_seconds": round(elapsed, 2),
               "device": str(device),
               "train_samples": len(tr), "val_samples": len(va), "test_samples": len(te),
               "per_class_accuracy": {l: round(pca.get(l, 0.0), 4) for l in ALPHABET}}
    with open(args.out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(args.out_dir / "training_curve.json", "w") as f:
        json.dump({**history, "test_accuracy": round(te_acc, 4),
                   "test_loss": round(te_loss, 4)}, f)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ep = range(1, len(history["train_loss"]) + 1)
        ax1.plot(ep, history["train_loss"], label="train")
        ax1.plot(ep, history["val_loss"],   label="val")
        ax1.set(xlabel="Epoch", ylabel="Loss"); ax1.legend()
        ax2.plot(ep, [a*100 for a in history["train_acc"]], label="train")
        ax2.plot(ep, [a*100 for a in history["val_acc"]],  label="val")
        ax2.set(xlabel="Epoch", ylabel="Accuracy (%)",
                title=f"Accuracy (test={te_acc*100:.1f}%)"); ax2.legend()
        plt.tight_layout()
        fig.savefig(args.out_dir / "alphabet_training.png", dpi=150)
        LOGGER.info("Plot saved")
    except Exception as exc:
        LOGGER.warning("Could not save plot: %s", exc)
    print("\n" + "=" * 50)
    print(f"FINAL TEST ACCURACY: {te_acc*100:.2f}%")
    print(f"BEST VAL ACCURACY:   {best_val_acc*100:.2f}%")
    print(f"TIME:                {elapsed:.0f}s ({len(history['train_loss'])} epochs)")
    print(f"PARAMS:              {model.num_parameters():,}")
    print(f"CHECKPOINT:          {args.out_dir / 'alphabet_classifier.pt'}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
