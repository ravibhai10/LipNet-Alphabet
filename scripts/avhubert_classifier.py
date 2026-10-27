"""Cache real AV-HuBERT features, train a small head, or predict an annotated clip.

Example: python scripts/avhubert_classifier.py extract --manifest clips.jsonl --output runs/features/session1
Run this with .venv-avhubert-cpu. Commands never overwrite existing artifacts.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from models.feature_classifier import TemporalClassificationHead
from pipeline.avhubert.classification import AVHubertClassifier, audit_records, load_annotations, sha256_file
from pipeline.avhubert.loader import AVHubertLoader
from src.cpu_utils import configure_cpu_runtime
from src.video_training import augment_video, warmup_cosine_scheduler
from pipeline.eval.classification import classification_report, write_classification_report

def new_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Existing artifacts are protected; choose a new output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path

def extract(args):
    if not 1 <= args.frames <= 500 or args.augment_train_copies < 0:
        raise ValueError("Use 1-500 target frames and a nonnegative augmentation copy count.")
    records = load_annotations(args.manifest)
    if getattr(args, "alphabet", False):
        validate_alphabet(records)
    loader = AVHubertLoader(args.checkpoint).load()
    output = new_directory(args.output)
    fingerprint = sha256_file(loader.ckpt_path)
    written = []
    from tracking.mouth_canonicalization import read_aligned_mouth_video
    import cv2
    for index, row in enumerate(records):
        # Read and canonicalize once; optional training augmentation shares
        # the source's speaker and split and never enters validation/test.
        if row.get("input_kind", "raw_video") == "raw_video":
            rois, fps = read_aligned_mouth_video(row["video"])
        else:
            capture = cv2.VideoCapture(row["video"]); fps = capture.get(cv2.CAP_PROP_FPS) or 25.
            rois = []
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if len(rois) >= 500:
                        raise ValueError("Video must be segmented to at most 500 frames.")
                    rois.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            finally:
                capture.release()
            if not rois:
                raise ValueError(f"Cannot decode {row['video']}")
            rois = np.stack(rois)
        copies = args.augment_train_copies if row["split"] == "train" else 0
        for copy in range(copies + 1):
            source = rois
            if copy:
                clip = torch.from_numpy(rois.astype(np.float32) / 255.)[:, None]
                source = augment_video(clip).squeeze(1).numpy()
            features = loader.encode_frames(source, source_fps=fps, target_frames=args.frames)
            name = f"{index:07d}_{copy}.npy"
            np.save(output / name, features, allow_pickle=False)
            entry = {**row, "id": row["id"] + (f"_augmentation_{copy}" if copy else ""),
                     "feature": name, "feature_sha256": sha256_file(output / name), "augmentation": bool(copy)}
            written.append(entry)
        print(f"Encoded {index + 1}/{len(records)}: {row['id']}", flush=True)
    with (output / "features.jsonl").open("x", encoding="utf-8") as stream:
        for row in written:
            stream.write(json.dumps(row) + "\n")
    metadata = {"encoder_sha256": fingerprint, "preprocessing": loader.preprocessing(),
                "target_frames": args.frames, "input_dim": loader.d_out,
                "source_manifest_sha256": sha256_file(args.manifest), "data_source": "real_camera",
                "cache_format": "avhubert-features-v1"}
    (output / "encoder.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"feature_cache": str(output), "source_clips": len(records), "feature_sequences": len(written)}))

class CachedFeatures(Dataset):
    def __init__(self, root, rows, vocabulary, metadata=None):
        self.root, self.rows = Path(root), rows
        self.indices = {label: index for index, label in enumerate(vocabulary)}
        self.metadata = metadata
        self.verified = set()
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        feature = (self.root / row["feature"]).resolve()
        if not feature.is_relative_to(self.root.resolve()):
            raise ValueError("Feature path must remain within the cache.")
        if self.metadata and self.metadata.get("cache_format") == "avhubert-features-v1" and feature not in self.verified:
            if row.get("feature_sha256") != sha256_file(feature):
                raise ValueError("Cached feature fingerprint mismatch; extract this cache again.")
            self.verified.add(feature)
        values = np.load(feature, allow_pickle=False)
        if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Invalid cached feature sequence.")
        if self.metadata and values.shape != (self.metadata["target_frames"], self.metadata["input_dim"]):
            raise ValueError("Feature dimensions or temporal length differ from the cache metadata.")
        return torch.from_numpy(values.astype(np.float32)), self.indices[row["label"]], row.get("viseme_label", -1)

def batches(samples):
    features, labels, visemes = zip(*samples)
    lengths = torch.tensor([len(item) for item in features])
    return torch.nn.utils.rnn.pad_sequence(features, batch_first=True), torch.tensor(labels), torch.tensor(visemes), lengths

def epoch(model, loader, criterion, optimizer=None, aux_weight=.2):
    model.train(optimizer is not None)
    total, loss_sum, correct = 0, 0., 0
    with torch.set_grad_enabled(optimizer is not None):
        for features, labels, visemes, lengths in loader:
            output = model.forward_with_aux(features, lengths)
            loss = criterion(output["logits"], labels)
            if output["viseme_logits"] is not None:
                if torch.any(visemes < 0):
                    raise ValueError("Auxiliary viseme training requires explicit annotated viseme_label values.")
                loss = loss + aux_weight * criterion(output["viseme_logits"], visemes)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite classification loss; check the cached features and learning rate.")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            total += len(labels); loss_sum += loss.item() * len(labels)
            correct += (output["logits"].argmax(-1) == labels).sum().item()
    if total == 0:
        raise ValueError("Cannot train or evaluate an empty partition.")
    return loss_sum / total, correct / total

def train(args):
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")
    if args.viseme_weight < 0:
        raise ValueError("Viseme loss weight must be nonnegative.")
    root = args.features
    rows = [json.loads(line) for line in (root / "features.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    audit = audit_records(rows)
    encoder = json.loads((root / "encoder.json").read_text(encoding="utf-8"))
    if encoder.get("data_source") != "real_camera":
        raise ValueError("Feature cache has no real-camera provenance.")
    vocabulary = audit["classes"]
    if len(vocabulary) < 2:
        raise ValueError("Classification training requires at least two labeled classes.")
    if args.viseme_classes:
        for row in rows:
            value = row.get("viseme_label")
            if not isinstance(value, int) or not 0 <= value < args.viseme_classes or row.get("viseme_source") != "phoneme_annotation":
                raise ValueError("Viseme labels need explicit phoneme_annotation provenance; letter names are not phonemes.")
    torch.manual_seed(args.seed)
    config = {"input_dim": encoder["input_dim"], "num_classes": len(vocabulary),
              "hidden_dim": args.hidden_dim, "viseme_classes": args.viseme_classes}
    model = TemporalClassificationHead(**config)
    loaders = {}
    for split in ("train", "valid", "test"):
        dataset = CachedFeatures(root, [row for row in rows if row["split"] == split], vocabulary, encoder)
        for index in range(len(dataset)):
            dataset[index]  # Validate the entire cache before creating a training artifact.
        loaders[split] = DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train", collate_fn=batches, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = warmup_cosine_scheduler(optimizer, args.epochs, args.warmup_epochs, 1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=.1)
    output = new_directory(args.output)
    best, history = float("inf"), []
    metadata = {**encoder, "format": "avhubert-classifier-v1", "head_config": config,
                "vocabulary": vocabulary, "evaluation_protocol": audit["protocol"],
                "frozen_encoder": True, "label_smoothing": .1,
                "training_video_sha256": sorted({row["video_sha256"] for row in rows if row["split"] == "train"}),
                "training_speakers": sorted({row["speaker_id"] for row in rows if row["split"] == "train"})}
    began = time.perf_counter()
    for number in range(args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_accuracy = epoch(model, loaders["train"], criterion, optimizer, args.viseme_weight)
        val_loss, val_accuracy = epoch(model, loaders["valid"], criterion, aux_weight=args.viseme_weight)
        history.append({"epoch": number + 1, "lr": lr, "train_loss": train_loss, "train_accuracy": train_accuracy,
                        "val_loss": val_loss, "val_accuracy": val_accuracy})
        if val_loss < best:
            best = val_loss
            torch.save({"state_dict": model.state_dict(), "metadata": {**metadata, "trained_epochs": number + 1}},
                       output / "head.pt")
        scheduler.step()
        print(json.dumps(history[-1]), flush=True)
    best_checkpoint = torch.load(output / "head.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best_checkpoint["state_dict"])
    test_loss, test_accuracy = epoch(model, loaders["test"], criterion, aux_weight=args.viseme_weight)
    summary = {"model": "AV-HuBERT frozen encoder + supervised head", "epochs": len(history),
               "num_classes": len(vocabulary), "speaker_disjoint": True,
               "train_samples": len(loaders["train"].dataset), "val_samples": len(loaders["valid"].dataset),
               "test_samples": len(loaders["test"].dataset), "test_loss": test_loss, "test_accuracy": test_accuracy,
               "history": history, "elapsed_seconds": time.perf_counter() - began, "device": "cpu"}
    best_checkpoint["metadata"].update({"validation_samples": summary["val_samples"], "test_samples": summary["test_samples"]})
    torch.save(best_checkpoint, output / "head.pt")
    report = evaluate_head(model, loaders["test"], vocabulary)
    report.update({"protocol": "speaker_disjoint", "partition": "test", "head_sha256": sha256_file(output / "head.pt"),
                   "encoder_sha256": encoder["encoder_sha256"]})
    write_classification_report(output, report)
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"head": str(output / "head.pt"), "test_accuracy": test_accuracy, "classes": len(vocabulary)}))

def predict(args):
    classifier = AVHubertClassifier(args.head, args.checkpoint).load()
    features = classifier.loader.encode(args.video, input_kind=args.input_kind, target_frames=classifier.metadata["target_frames"])
    print(json.dumps(classifier.predict_features(features), indent=2))


def validate_alphabet(rows):
    for split in ("train", "valid", "test"):
        missing = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") - {row["label"] for row in rows if row["split"] == split}
        unknown = {row["label"] for row in rows} - set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        if missing or unknown:
            raise ValueError(f"Alphabet {split} partition: missing {sorted(missing)}, unknown {sorted(unknown)}")


def validate(args):
    rows = load_annotations(args.manifest)
    if args.alphabet:
        validate_alphabet(rows)
    print(json.dumps(audit_records(rows), indent=2))


def evaluate_head(model, loader, vocabulary):
    targets, predictions = [], []
    began = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for features, labels, _visemes, lengths in loader:
            logits = model(features, lengths)
            if not torch.isfinite(logits).all():
                raise ValueError("Head produced nonfinite logits.")
            targets.extend(labels.tolist()); predictions.extend(logits.argmax(-1).tolist())
    report = classification_report(targets, predictions, vocabulary)
    report["cached_head_evaluation_seconds"] = time.perf_counter() - began
    return report


def evaluate(args):
    rows = [json.loads(line) for line in (args.features / "features.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    audit_records(rows)
    encoder = json.loads((args.features / "encoder.json").read_text(encoding="utf-8"))
    classifier = AVHubertClassifier(args.head, args.checkpoint).load()
    for key in ("encoder_sha256", "preprocessing", "target_frames", "input_dim"):
        if encoder.get(key) != classifier.metadata.get(key):
            raise ValueError(f"Evaluation cache differs from the trained head: {key}")
    test_rows = [row for row in rows if row["split"] == "test"]
    train_hashes = set(classifier.metadata.get("training_video_sha256", []))
    train_speakers = set(classifier.metadata.get("training_speakers", []))
    if not train_hashes or not train_speakers:
        raise ValueError("Head has no training-membership audit; retrain with the current pipeline.")
    if any(row["video_sha256"] in train_hashes or row["speaker_id"] in train_speakers for row in test_rows):
        raise ValueError("Evaluation data overlaps the head's training speakers or videos.")
    dataset = CachedFeatures(args.features, test_rows, classifier.metadata["vocabulary"], encoder)
    loader = DataLoader(dataset, batch_size=16, collate_fn=batches)
    report = evaluate_head(classifier.head, loader, classifier.metadata["vocabulary"])
    report.update({"protocol": "speaker_disjoint", "partition": "test", "head_sha256": sha256_file(args.head),
                   "encoder_sha256": encoder["encoder_sha256"]})
    output = new_directory(args.output)
    write_classification_report(output, report)
    print(json.dumps(report, indent=2))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    extract_parser = sub.add_parser("extract")
    extract_parser.add_argument("--manifest", type=Path, required=True)
    extract_parser.add_argument("--output", type=Path, required=True)
    extract_parser.add_argument("--checkpoint", type=Path)
    extract_parser.add_argument("--frames", type=int, default=29)
    extract_parser.add_argument("--augment-train-copies", type=int, default=0)
    extract_parser.add_argument("--alphabet", action="store_true", help="Require all A-Z labels in every partition")
    validate_parser = sub.add_parser("validate", help="Validate annotations and report class/speaker counts without loading a model")
    validate_parser.add_argument("--manifest", type=Path, required=True)
    validate_parser.add_argument("--alphabet", action="store_true")
    evaluate_parser = sub.add_parser("evaluate", help="Evaluate a trained head on the audited test cache")
    evaluate_parser.add_argument("--features", type=Path, required=True)
    evaluate_parser.add_argument("--head", type=Path, required=True)
    evaluate_parser.add_argument("--checkpoint", type=Path)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--features", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--warmup-epochs", type=int, default=5)
    train_parser.add_argument("--hidden-dim", type=int, default=0)
    train_parser.add_argument("--viseme-classes", type=int, default=0)
    train_parser.add_argument("--viseme-weight", type=float, default=.2)
    train_parser.add_argument("--seed", type=int, default=42)
    predict_parser = sub.add_parser("predict")
    predict_parser.add_argument("--head", type=Path, required=True)
    predict_parser.add_argument("--checkpoint", type=Path)
    predict_parser.add_argument("--video", type=Path, required=True)
    predict_parser.add_argument("--input-kind", choices=("raw_video", "mouth_roi"), default="raw_video")
    args = parser.parse_args()
    configure_cpu_runtime(4)
    {"extract": extract, "train": train, "predict": predict, "validate": validate, "evaluate": evaluate}[args.command](args)

if __name__ == "__main__":
    main()

