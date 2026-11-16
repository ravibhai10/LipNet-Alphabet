"""Audited LRW whole-word workflow. Independent of GRID and live alphabet selection.

Official LRW date partitions are preserved; speaker independence is UNKNOWN.
Legacy caches/checkpoints without this artifact contract are never auto-selected.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data.schema import VOCABULARY
from models.alphabet_model import AlphabetCNNBiLSTM
from src.video_training import augment_video, resample_temporal, warmup_cosine_scheduler

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "valid", "test")
PREPROCESSING = {"version": "lrw-measured-mouth-v1", "frames": 29, "channels": 1,
                 "height": 140, "width": 46, "normalization": "uint8 / 255",
                 "alignment": "measured eye-axis; mouth-centered similarity crop 96x96; margin 2.2",
                 "tracking": "at least 80% tracked; reject interior gaps over 5 frames; interpolate tracked frames",
                 "resize": "OpenCV INTER_LINEAR to width 46, height 140",
                 "quantization": "round interpolated grayscale to uint8 before cache/train/predict",
                 "temporal": "linear interpolation of full 29-frame contextual clip; no duration trimming"}
ARCHITECTURE = "AlphabetCNNBiLSTM-defaults-500-v1"
PROTOCOL = "publisher LRW date splits; speaker independence unknown"
TASK = "lrw_500_whole_word"
if len(VOCABULARY) != 500 or len(set(VOCABULARY)) != 500:
    raise ValueError("The LRW word workflow requires exactly 500 unique vocabulary entries.")


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def cache_contract(manifest_hash):
    return {"manifest_sha256": manifest_hash, "preprocessing": PREPROCESSING,
            "vocabulary": VOCABULARY, "landmarker_sha256": digest(ROOT / "frontend/public/models/face_landmarker.task")}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def fresh_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("Existing output is protected; choose a new directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_metadata(path):
    """Actual publisher example contains broadcast reference and word duration, not a label."""
    text = Path(path).read_text(encoding="utf-8-sig")
    patterns = {"disk_reference": r"^Disk reference:\s*(\d+)\s*$",
                "channel": r"^Channel:\s*(.+)$",
                "program_start": r"^Program start:\s*(.+)$",
                "clip_start": r"^Clip start:\s*([\d.]+) seconds\s*$",
                "word_duration": r"^Duration:\s*([\d.]+) seconds\s*$"}
    result = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text, re.MULTILINE)
        if len(matches) != 1:
            raise ValueError(f"Missing/ambiguous LRW annotation field {key}: {path}")
        result[key] = matches[0].strip()
    for key in ("clip_start", "word_duration"):
        result[key] = float(result[key])
        if not math.isfinite(result[key]) or result[key] < 0:
            raise ValueError("Invalid LRW annotation timing.")
    if not 0 < result["word_duration"] <= 1.16:
        raise ValueError("Word duration must fit a 29-frame/25fps LRW clip.")
    return result


def decode_check(path):
    capture = cv2.VideoCapture(str(path))
    frames, fingerprints, dimensions = 0, hashlib.sha256(), set()
    fps = capture.get(cv2.CAP_PROP_FPS)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames += 1
            if frames > 29:
                raise ValueError(f"Expected exactly 29 LRW frames: {path}")
            dimensions.add(frame.shape[:2])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fingerprints.update(cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).tobytes())
    finally:
        capture.release()
    if frames != 29 or len(dimensions) != 1 or not np.isfinite(fps) or abs(fps - 25) > .1:
        raise ValueError(f"LRW decoding requires 29 frames at 25fps: {path}")
    h, w = next(iter(dimensions))
    if min(h, w) < 96:  # Reject legacy generated 140x46 fixture ROIs.
        raise ValueError(f"Expected original LRW face video, not legacy ROI fixtures: {path}")
    return {"frames": frames, "fps": fps, "height": h, "width": w,
            "decoded_thumbnail_sha256": fingerprints.hexdigest()}


def validate_distribution(rows, vocabulary=VOCABULARY):
    counts = {split: Counter(row["word"] for row in rows if row["split"] == split) for split in SPLITS}
    for split in SPLITS:
        if set(counts[split]) != set(vocabulary):
            raise ValueError(f"{split}: all {len(vocabulary)} vocabulary classes are required.")
        for word, count in counts[split].items():
            if (split == "train" and not 800 <= count <= 1000) or (split != "train" and count != 50):
                raise ValueError(f"Incomplete official LRW distribution: {split}/{word} has {count} clips.")
    return {split: dict(counts[split]) for split in SPLITS}


def ingest(root, output, *, agreement_confirmed=False):
    """Read-only source audit; resumable decoding cache, immutable completed manifest.

    root must directly contain WORD/{train,val,test} pairs, as distributed by LRW.
    Approval/provenance is a user declaration, not something inferred from filenames.
    """
    if not agreement_confirmed:
        raise ValueError("Confirm authorized official LRW acquisition with --agreement-confirmed.")
    root, output = Path(root).resolve(), Path(output).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if (output / "dataset.json").exists():
        raise FileExistsError("Completed dataset output is protected.")
    output.mkdir(parents=True, exist_ok=True)
    contract = {"root": str(root), "vocabulary": VOCABULARY, "preprocessing": PREPROCESSING}
    progress = output / "ingestion_contract.json"
    if progress.exists() and json.loads(progress.read_text()) != contract:
        raise ValueError("Cannot resume with a different input root/vocabulary/preprocessing.")
    atomic_json(progress, contract)
    audit_cache = output / "validation_cache"
    audit_cache.mkdir(exist_ok=True)
    videos = sorted(root.rglob("*.mp4"))
    if not videos:
        raise ValueError("No official LRW MP4 files found.")
    rows, byte_hashes, decoded_hashes, references = [], {}, {}, {}
    paired_annotations = set()
    for path in videos:
        relative = path.relative_to(root)
        if len(relative.parts) != 3:
            raise ValueError("Expected official WORD/{train,val,test}/CLIP.mp4 layout.")
        word, source_split, _ = relative.parts
        split = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}.get(source_split)
        if word not in VOCABULARY or split is None or not path.stem.upper().startswith(word + "_"):
            raise ValueError(f"Unknown/mismatched LRW class, filename, or split: {relative}")
        annotation = path.with_suffix(".txt")
        if not annotation.is_file():
            raise FileNotFoundError(f"Missing original LRW annotation: {annotation}")
        metadata = read_metadata(annotation)
        paired_annotations.add(annotation.resolve())
        source_hash, annotation_hash = digest(path), digest(annotation)
        cache_path = audit_cache / (source_hash + ".json")
        check = json.loads(cache_path.read_text()) if cache_path.exists() else decode_check(path)
        atomic_json(cache_path, check)
        row_id = relative.with_suffix("").as_posix()
        for seen, key in ((byte_hashes, source_hash), (decoded_hashes, check["decoded_thumbnail_sha256"]),
                          (references, (metadata["disk_reference"], metadata["clip_start"]))):
            if key in seen:
                raise ValueError(f"Duplicate source/content/broadcast clip: {seen[key]} and {row_id}")
            seen[key] = row_id
        rows.append({"id": row_id, "video": str(path), "annotation": str(annotation), "word": word,
                     "label": VOCABULARY.index(word), "split": split, "source_sha256": source_hash,
                     "annotation_sha256": annotation_hash, "metadata": metadata, **check})
        if len(rows) % 1000 == 0:
            print(f"Validated {len(rows):,} source/annotation pairs", flush=True)
    annotations = {path.resolve() for path in root.rglob("*.txt")}
    if annotations != paired_annotations:
        raise ValueError("Orphaned/unrecognized text annotations exist; choose the direct LRW class root.")
    distribution = validate_distribution(rows)
    manifest = output / "clips.jsonl"
    temporary = output / "clips.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    if manifest.exists() and digest(manifest) != digest(temporary):
        raise FileExistsError("An existing ingestion manifest differs; preserved. Choose a new output.")
    os.replace(temporary, manifest)
    dataset = {"schema_version": 1, "task": TASK, "source": "Oxford-BBC official LRW (user declared)",
               "agreement_confirmed": True, "vocabulary": VOCABULARY, "preprocessing": PREPROCESSING,
               "protocol": PROTOCOL, "speaker_disjoint": None, "manifest_sha256": digest(manifest),
               "clips": len(rows), "distribution": distribution,
               "decoder": {"opencv": cv2.__version__, "policy": "29 decoded frames at 25fps; no replacements"}}
    atomic_json(output / "dataset.json", dataset)
    return dataset


def load_dataset(directory):
    directory = Path(directory)
    info = json.loads((directory / "dataset.json").read_text())
    if (info.get("schema_version") != 1 or info.get("task") != TASK or not info.get("agreement_confirmed")
            or info.get("vocabulary") != VOCABULARY or info.get("preprocessing") != PREPROCESSING
            or info.get("protocol") != PROTOCOL or info.get("manifest_sha256") != digest(directory / "clips.jsonl")):
        raise ValueError("Dataset contract/fingerprint mismatch; legacy manifests are not accepted.")
    with (directory / "clips.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != info.get("clips") or validate_distribution(rows) != info.get("distribution"):
        raise ValueError("Dataset count/distribution mismatch.")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate sample IDs.")
    for row in rows:
        if row["label"] != VOCABULARY.index(row["word"]):
            raise ValueError("Class mapping mismatch.")
    return info, rows


def preprocess_video(path):
    from tracking.mouth_canonicalization import read_aligned_mouth_video
    # Fail rather than feeding a full face warped into a mouth tensor.
    decode_check(path)
    crops, _ = read_aligned_mouth_video(path, max_frames=29)
    resized = [cv2.resize(crop, (46, 140), interpolation=cv2.INTER_LINEAR) for crop in crops]
    result = np.rint(resample_temporal(resized, 29)).clip(0, 255).astype(np.uint8)
    return torch.from_numpy(result[:, None].copy())


def cache_clips(dataset, output, *, limit=None, reserve_gib=20):
    info, rows = load_dataset(dataset)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    contract = cache_contract(info["manifest_sha256"])
    path = output / "cache_contract.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("Cache belongs to a different manifest/preprocessing/landmarker.")
    atomic_json(path, contract)
    remaining = sum(not (output / (row["source_sha256"] + ".pt")).exists() for row in rows)
    projected_bytes = remaining * (29 * 140 * 46 + 8192)  # uint8, not 4x larger float32 caches
    if shutil.disk_usage(output).free < projected_bytes + reserve_gib * 2**30:
        raise OSError(f"Insufficient disk: approximately {projected_bytes / 2**30:.1f} GiB cache plus reserve needed.")
    processed, start = 0, time.perf_counter()
    for row in rows:
        target = output / (row["source_sha256"] + ".pt")
        sidecar = target.with_suffix(".json")
        if target.exists() and sidecar.exists():
            meta = json.loads(sidecar.read_text())
            if meta == {"source_sha256": row["source_sha256"], "tensor_sha256": digest(target), "contract": signature(contract)}:
                continue
            raise ValueError(f"Existing cache fingerprint mismatch: {target}")
        if target.exists():
            raise ValueError(f"Unverified cache file is protected: {target}")
        if limit is not None and processed >= limit:
            break
        if digest(row["video"]) != row["source_sha256"] or digest(row["annotation"]) != row["annotation_sha256"]:
            raise ValueError("Source video or annotation changed after ingestion.")
        tensor = preprocess_video(row["video"])
        temporary = target.with_suffix(".tmp")
        torch.save(tensor, temporary)
        os.replace(temporary, target)
        atomic_json(sidecar, {"source_sha256": row["source_sha256"], "tensor_sha256": digest(target), "contract": signature(contract)})
        processed += 1
        if processed % 100 == 0:
            print(f"Aligned/cached {processed} new clips", flush=True)
    result = {"new_clips": processed, "seconds": time.perf_counter() - start,
              "total_clips": len(rows), "estimated_cache_bytes": len(rows) * (29 * 140 * 46 + 8192)}
    atomic_json(output / "cache_progress.json", result)
    return result


class CachedWords(Dataset):
    def __init__(self, rows, root, augment=False):
        self.rows, self.root, self.augment = rows, Path(root), augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        tensor = torch.load(self.root / (row["source_sha256"] + ".pt"), map_location="cpu", weights_only=True)
        if tensor.dtype != torch.uint8 or tuple(tensor.shape) != (29, 1, 140, 46):
            raise ValueError("Invalid cached tensor format.")
        tensor = tensor.float().div_(255)
        return (augment_video(tensor) if self.augment else tensor), row["label"]


def verified_cache(dataset, cache):
    info, rows = load_dataset(dataset)
    cache = Path(cache)
    contract = json.loads((cache / "cache_contract.json").read_text())
    if contract != cache_contract(info["manifest_sha256"]):
        raise ValueError("Cache and dataset contracts differ.")
    for row in rows:
        path = cache / (row["source_sha256"] + ".pt")
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta != {"source_sha256": row["source_sha256"], "tensor_sha256": digest(path), "contract": signature(contract)}:
            raise ValueError("Cached source/contract/tensor fingerprint mismatch.")
    return info, rows, contract


def word_metrics(targets, rankings, vocabulary=VOCABULARY):
    """Streaming-compatible top-k report, with every class represented including absent ones."""
    n = len(vocabulary)
    if not n or len(set(vocabulary)) != n or not len(targets) or len(targets) != len(rankings):
        raise ValueError("Unique vocabulary and nonempty paired rankings are required.")
    matrix = np.zeros((n, n), dtype=np.int64)
    top5, supported = Counter(), Counter()
    for truth, ranked in zip(targets, rankings):
        if (type(truth) not in (int, np.int64, np.int32) or not 0 <= truth < n or len(ranked) != min(5, n)
                or len(set(ranked)) != len(ranked) or any(type(i) not in (int, np.int64, np.int32) or not 0 <= i < n for i in ranked)):
            raise ValueError("Invalid target or top-k class indices.")
        matrix[truth, ranked[0]] += 1
        supported[truth] += 1
        top5[truth] += int(truth in ranked)
    return {"samples": len(targets), "top1_accuracy": float(matrix.trace() / len(targets)),
            "top5_accuracy": sum(top5.values()) / len(targets), "top_k": min(5, n),
            "vocabulary": list(vocabulary), "classes_evaluated": len(supported),
            "per_word": {word: {"correct": int(matrix[i, i]), "total": supported[i], "top5_correct": top5[i],
                                "accuracy": int(matrix[i, i]) / supported[i] if supported[i] else None}
                         for i, word in enumerate(vocabulary)},
            "confusion_matrix": matrix.tolist(), "confusion_matrix_axes": {"rows": "true", "columns": "predicted"}}


def write_metrics(output, report, predictions):
    atomic_json(Path(output) / "evaluation.json", report)
    with (Path(output) / "confusion_matrix.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true / predicted", *report["vocabulary"]])
        for label, values in zip(report["vocabulary"], report["confusion_matrix"]):
            writer.writerow([label, *values])
    with (Path(output) / "per_word.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["word", "correct", "total", "top5_correct", "accuracy"])
        for label, values in report["per_word"].items():
            writer.writerow([label, *values.values()])
    with (Path(output) / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row) + "\n")


@torch.no_grad()
def score(model, loader):
    model.eval()
    total_loss, targets, rankings = 0., [], []
    for inputs, labels in loader:
        logits = model(inputs)
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite model output.")
        total_loss += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        targets.extend(labels.tolist())
        rankings.extend(logits.topk(min(5, logits.shape[1]), dim=1).indices.tolist())
    if not targets:
        raise ValueError("Empty evaluation partition.")
    return total_loss / len(targets), targets, rankings


def load_word_model(path):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if (artifact.get("schema_version") != 1 or artifact.get("task") != TASK
            or artifact.get("architecture") != ARCHITECTURE or artifact.get("vocabulary") != VOCABULARY
            or artifact.get("vocabulary_sha256") != signature(VOCABULARY) or artifact.get("preprocessing") != PREPROCESSING
            or artifact.get("protocol") != PROTOCOL or not re.fullmatch("[0-9a-f]{64}", artifact.get("manifest_sha256", ""))
            or artifact.get("cache_contract") != cache_contract(artifact.get("manifest_sha256")) or not artifact.get("epoch", 0) >= 1
            or not math.isfinite(artifact.get("validation_loss", float("nan")))):
        raise ValueError("Not an audited LRW word artifact; legacy 500-output weights have unverified mapping/provenance.")
    model = AlphabetCNNBiLSTM(num_classes=500).eval()
    model.load_state_dict(artifact["model_state_dict"], strict=True)
    if any(not torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError("Checkpoint contains nonfinite parameters.")
    return model, artifact


def train(dataset, cache, output, *, epochs=30, batch_size=2, lr=1e-4, warmup_epochs=5, init=None, init_word=None, threads=2, seed=42):
    if epochs < 1 or batch_size < 1 or not math.isfinite(lr) or lr < 1e-6 or threads < 1 or warmup_epochs < 0:
        raise ValueError("Invalid training parameters.")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    if init and init_word:
        raise ValueError("Choose one initialization source.")
    info, rows, contract = verified_cache(dataset, cache)
    output = fresh_output(output)
    if shutil.disk_usage(output).free < 1024**3:
        raise OSError("At least 1 GiB free space is required for training artifacts.")
    model = AlphabetCNNBiLSTM(num_classes=500)
    initialization = {"kind": "random"}
    if init_word:
        model, artifact = load_word_model(init_word)
        if artifact["manifest_sha256"] != info["manifest_sha256"] or artifact["cache_contract"] != contract:
            raise ValueError("Word fine-tuning requires the same frozen splits/cache to prevent unknown test leakage.")
        initialization = {"kind": "audited whole-word warm start; optimizer/scheduler reset", "sha256": digest(init_word)}
    if init:
        artifact = torch.load(init, map_location="cpu", weights_only=True)
        state = artifact.get("model_state_dict", {})
        # Existing alphabet encoder only: all classification layers are reset.
        if tuple(state.get("fc2.weight", torch.empty(0)).shape) != (26, 128):
            raise ValueError("--init-alphabet requires the existing 26-class architecture; legacy 500 heads are not trusted.")
        expected = AlphabetCNNBiLSTM(num_classes=26)
        expected.load_state_dict(state, strict=True)
        model.load_state_dict({key: value for key, value in state.items() if not key.startswith("fc")}, strict=False)
        initialization = {"kind": "alphabet CNN+BiLSTM transfer; fc1/fc2 reset", "sha256": digest(init)}
    loaders = {split: DataLoader(CachedWords([r for r in rows if r["split"] == split], cache, split == "train"),
                                batch_size=batch_size, shuffle=split == "train", num_workers=0) for split in SPLITS}
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = warmup_cosine_scheduler(optimizer, epochs, warmup_epochs, 1e-6)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=.1)
    best, history, start = float("inf"), [], time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0., 0
        for inputs, labels in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            running += loss.item() * len(labels)
            seen += len(labels)
        validation_loss, targets, rankings = score(model, loaders["valid"])
        metrics = word_metrics(targets, rankings)
        history.append({"epoch": epoch, "train_loss_smoothed": running / seen, "validation_loss": validation_loss,
                        "validation_top1": metrics["top1_accuracy"], "validation_top5": metrics["top5_accuracy"],
                        "lr": optimizer.param_groups[0]["lr"]})
        if validation_loss < best:
            best = validation_loss
            torch.save({"schema_version": 1, "task": TASK, "architecture": ARCHITECTURE,
                        "vocabulary": VOCABULARY, "vocabulary_sha256": signature(VOCABULARY),
                        "preprocessing": PREPROCESSING, "protocol": PROTOCOL, "speaker_disjoint": None,
                        "manifest_sha256": info["manifest_sha256"], "cache_contract": contract,
                        "initialization": initialization, "epoch": epoch, "validation_loss": best,
                        "model_state_dict": model.state_dict()}, output / "word_model.pt")
        scheduler.step()
        atomic_json(output / "training_summary.json", {"task": TASK, "seed": seed, "threads": threads,
                    "epochs_requested": epochs, "batch_size": batch_size, "lr": lr, "warmup_epochs": warmup_epochs,
                    "label_smoothing": .1, "initialization": initialization, "history": history,
                    "selection": "minimum unsmoothed validation cross-entropy", "seconds": time.perf_counter() - start})
        print(json.dumps(history[-1]), flush=True)
    # Test only after validation selection; no test gradients or threshold tuning.
    return evaluate(dataset, cache, output / "word_model.pt", output / "held_out_test", threads=threads)


def evaluate(dataset, cache, checkpoint, output, *, batch_size=2, threads=2):
    if batch_size < 1 or threads < 1:
        raise ValueError("Positive batch size/threads required.")
    torch.set_num_threads(threads)
    info, rows, contract = verified_cache(dataset, cache)
    model, artifact = load_word_model(checkpoint)
    if artifact["manifest_sha256"] != info["manifest_sha256"] or artifact["cache_contract"] != contract:
        raise ValueError("Evaluation requires the original audited held-out partition and cache contract.")
    rows = [r for r in rows if r["split"] == "test"]
    loss, targets, rankings = score(model, DataLoader(CachedWords(rows, cache), batch_size=batch_size, num_workers=0))
    report = word_metrics(targets, rankings)
    report.update(task=TASK, split="test", protocol=PROTOCOL, speaker_disjoint=None, loss=loss,
                  checkpoint_sha256=digest(checkpoint), manifest_sha256=info["manifest_sha256"])
    output = fresh_output(output)
    predictions = [{"id": row["id"], "truth": row["word"], "top5": [VOCABULARY[i] for i in ranked]}
                   for row, ranked in zip(rows, rankings)]
    write_metrics(output, report, predictions)
    return {key: report[key] for key in ("samples", "top1_accuracy", "top5_accuracy", "classes_evaluated")}


def predict(checkpoint, video, *, threads=2):
    if threads < 1:
        raise ValueError("Positive threads required.")
    torch.set_num_threads(threads)
    model, artifact = load_word_model(checkpoint)
    with torch.no_grad():
        probabilities = model(preprocess_video(video).float().div(255).unsqueeze(0)).softmax(-1)[0]
    if not torch.isfinite(probabilities).all():
        raise ValueError("Nonfinite inference probabilities.")
    values, indices = probabilities.topk(5)
    return {"task": TASK, "checkpoint_sha256": digest(checkpoint), "confidence_calibrated": False,
            "top5": [{"word": VOCABULARY[i], "probability": p} for p, i in zip(values.tolist(), indices.tolist())]}
