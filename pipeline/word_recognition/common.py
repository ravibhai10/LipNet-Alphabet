from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "valid", "test")
TASK = "grid_aligned_whole_words"
LEXICAL_WORDS = {"bin", "lay", "place", "set", "blue", "green", "red", "white",
                 "at", "by", "in", "with", "zero", "one", "two", "three", "four", "five",
                 "six", "seven", "eight", "nine", "again", "now", "please", "soon"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        from pipeline.paths import relocate_row
        return [relocate_row(json.loads(line)) for line in stream if line.strip()]


def write_rows(path, rows):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    os.replace(temp, path)


def project_path(path):
    return (ROOT / path).resolve()


def load_config(path):
    config = load_json(path)
    words, preprocessing, training = config["words"], config["preprocessing"], config["training"]
    if config.get("dataset") != TASK or len(words) < 2 or len(set(words)) != len(words) or not set(words) <= LEXICAL_WORDS:
        raise ValueError("At least two unique annotated GRID lexical words are required; letter slots are excluded.")
    if preprocessing != {"version": "grid-whole-word-mouth-v1", "frames": 24, "size": 64, "grayscale": True}:
        raise ValueError("Unsupported preprocessing contract.")
    for name in ("quotas", "minimum_speakers"):
        if set(config[name]) != set(SPLITS) or any(type(n) is not int or n < 1 for n in config[name].values()):
            raise ValueError("Positive quotas/speaker counts are required for all three splits.")
    if any(config["quotas"][s] < config["minimum_speakers"][s] for s in SPLITS):
        raise ValueError("Quotas cannot be smaller than minimum speakers per class.")
    if any(type(training[k]) is not int or training[k] < 1 for k in ("epochs", "batch_size", "threads", "patience")):
        raise ValueError("Positive epochs/batch size/threads/patience required.")
    if any(not math.isfinite(training[k]) or training[k] < 0 for k in ("lr", "weight_decay", "label_smoothing", "min_delta", "max_minutes")) or training["lr"] == 0 or training["max_minutes"] == 0 or training["label_smoothing"] >= 1:
        raise ValueError("Invalid training numeric parameters.")
    if config["reserve_gib"] < 0 or config["download_max_minutes"] <= 0:
        raise ValueError("Invalid disk/time budget.")
    return config


def parser(description):
    import argparse
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    return p


def run_contract(config, run):
    run = Path(run).resolve()
    metadata = {"config": config, "config_sha256": fingerprint(config), "task": TASK}
    contract = run / "contract.json"
    run.mkdir(parents=True, exist_ok=True)
    if contract.exists() and load_json(contract) != metadata:
        raise ValueError("Existing run belongs to another config. Choose a new run directory.")
    if not contract.exists() and any(run.iterdir()):
        raise FileExistsError("Refusing to use a nonempty unrelated run directory.")
    atomic_json(contract, metadata)
    return run


def timestamp():
    return datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat()


def prepared_root(config):
    return project_path(config["processed_root"]) / fingerprint(config)[:16]


def validate_rows(rows, config, *, require_quotas=True):
    if not rows:
        raise ValueError("No actual word samples.")
    speakers, sources, segments, ids, pixel_hashes = {}, {}, set(), set(), {}
    counts = {split: Counter() for split in SPLITS}
    coverage = defaultdict(set)
    for row in rows:
        split, word, speaker = row["split"], row["word"], row["speaker"]
        if split not in SPLITS or word not in config["words"] or row["label"] != config["words"].index(word):
            raise ValueError("Invalid word vocabulary or split mapping.")
        if row.get("data_source") != "official_grid_video" or row.get("landmark_source") != "mediapipe":
            raise ValueError("Real GRID video and measured landmarks are required.")
        if not speaker or speaker in speakers and speakers[speaker] != split:
            raise ValueError("Speaker leakage across partitions.")
        speakers[speaker] = split
        source = row["source_sha256"]
        if source in sources and sources[source] != split:
            raise ValueError("Source utterance leakage across partitions.")
        sources[source] = split
        first, last = row["start_frame"], row["end_frame"]
        if type(first) is not int or type(last) is not int or first < 0 or last - first < 2:
            raise ValueError("Invalid actual word bounds.")
        key = (source, first, last)
        if key in segments or row["id"] in ids:
            raise ValueError("Repeated word segment or sample ID.")
        segments.add(key)
        ids.add(row["id"])
        pixels = row["pixels_sha256"]
        if pixels in pixel_hashes:
            raise ValueError("Duplicate decoded mouth content.")
        pixel_hashes[pixels] = split
        counts[split][word] += 1
        coverage[split, word].add(speaker)
    if require_quotas:
        for split in SPLITS:
            for word in config["words"]:
                if counts[split][word] != config["quotas"][split] or len(coverage[split, word]) < config["minimum_speakers"][split]:
                    raise ValueError(f"Incomplete class/speaker quota: {split}/{word}")
    return {"counts": {s: dict(c) for s, c in counts.items()},
            "speakers": {s: sorted(k for k, v in speakers.items() if v == s) for s in SPLITS},
            "per_class_speakers": {s: {w: sorted(coverage[s, w]) for w in config["words"]} for s in SPLITS}}


def load_dataset(run, config, *, verify_pixels=False):
    run = Path(run)
    info = load_json(run / "dataset.json")
    manifest = run / "words.jsonl"
    if (info.get("task") != TASK or info.get("config_sha256") != fingerprint(config) or info.get("manifest_sha256") != sha256(manifest)
            or info.get("vocabulary") != config["words"] or info.get("preprocessing") != config["preprocessing"]):
        raise ValueError("Word dataset fingerprint/contract mismatch.")
    model = ROOT / "frontend/public/models/face_landmarker.task"
    if info.get("landmarker_sha256") != sha256(model):
        raise ValueError("Face landmarker changed after preparation.")
    rows = read_rows(manifest)
    validate_rows(rows, config)
    if verify_pixels:
        from pipeline.word_recognition.preprocess import read_frames
        for row in rows:
            read_frames(run, row, config)
    return info, rows
