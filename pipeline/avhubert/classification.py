"""Supervised classifier contracts and data audits for frozen AV-HuBERT features."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from models.feature_classifier import TemporalClassificationHead
from pipeline.avhubert.loader import AVHubertLoader, ModelUnavailableError


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_records(records, *, require_splits=True):
    """Reject synthetic/unmarked data, duplicate clips, and speaker leakage."""
    if not records:
        raise ValueError("The annotated dataset is empty.")
    speakers, clips, identifiers, classes = {}, {}, set(), {}
    for row in records:
        if row.get("data_source") != "real_camera":
            raise ValueError("Training requires explicit real_camera provenance; fixture data is not accepted.")
        identifier, speaker, label = (str(row.get(key, "")).strip() for key in ("id", "speaker_id", "label"))
        if not all(isinstance(row.get(key), str) for key in ("id", "speaker_id", "label")):
            raise ValueError("id, speaker_id, and label must be strings.")
        split = row.get("split")
        if not identifier or not speaker or not label or split not in ("train", "valid", "test"):
            raise ValueError("Every clip needs id, speaker_id, label, and a train/valid/test split.")
        if any(row[key] != row[key].strip() for key in ("id", "speaker_id", "label")):
            raise ValueError("Identifiers and labels must not contain leading or trailing whitespace.")
        if row.get("augmentation") and split != "train":
            raise ValueError("Augmentation is permitted in training only.")
        if identifier in identifiers:
            raise ValueError(f"Duplicate clip id: {identifier}")
        identifiers.add(identifier)
        if speaker in speakers and speakers[speaker] != split:
            raise ValueError(f"Speaker leakage across splits: {speaker}")
        speakers[speaker] = split
        digest = row.get("video_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("Each clip needs its source video SHA-256.")
        if digest in clips and clips[digest] != (split, label):
            raise ValueError("Duplicate video content crosses splits or has conflicting labels.")
        clips[digest] = (split, label)
        classes.setdefault(split, set()).add(label)
    if require_splits:
        if set(classes) != {"train", "valid", "test"}:
            raise ValueError("Nonempty train, valid, and test partitions are required.")
        if (classes["valid"] | classes["test"]) - classes["train"]:
            raise ValueError("Evaluation contains classes missing from the training partition.")
    return {"clips": len(identifiers), "speakers": len(speakers),
            "classes": sorted(classes.get("train", set())), "protocol": "speaker_disjoint",
            "split_counts": {split: sum(row["split"] == split for row in records) for split in ("train", "valid", "test")},
            "label_counts": {split: {label: sum(row["split"] == split and row["label"] == label for row in records)
                                     for label in sorted(classes.get(split, set()))} for split in ("train", "valid", "test")}}


def load_annotations(manifest):
    manifest = Path(manifest).resolve()
    records = []
    for number, line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        source = manifest.parent / row["video"]
        if not source.is_file():
            raise ValueError(f"Missing video at line {number}: {source}")
        row = dict(row, video=str(source.resolve()), video_sha256=sha256_file(source))
        if row.get("input_kind", "raw_video") not in ("raw_video", "mouth_roi"):
            raise ValueError("input_kind must be raw_video or mouth_roi.")
        records.append(row)
    audit_records(records)
    return records


class AVHubertClassifier:
    def __init__(self, head_path, encoder_path=None):
        self.head_path = Path(head_path)
        self.loader = AVHubertLoader(encoder_path)
        self.head = None
        self.metadata = None

    def load(self):
        if self.head is not None:
            return self
        if not self.head_path.is_file():
            raise ModelUnavailableError("AV-HuBERT encoder weights are available, but a trained classifier head is required.")
        checkpoint = torch.load(self.head_path, map_location="cpu", weights_only=True)
        metadata = checkpoint.get("metadata", {})
        if metadata.get("format") != "avhubert-classifier-v1" or metadata.get("data_source") != "real_camera":
            raise ModelUnavailableError("The classifier is not a verified real-data AV-HuBERT head artifact.")
        if metadata.get("evaluation_protocol") != "speaker_disjoint":
            raise ModelUnavailableError("The classifier requires audited speaker-disjoint partitions.")
        if metadata.get("frozen_encoder") is not True or any(
            not isinstance(metadata.get(key), int) or metadata[key] < 1
            for key in ("trained_epochs", "validation_samples", "test_samples", "target_frames")
        ):
            raise ModelUnavailableError("The classifier needs completed training and nonempty evaluation partitions.")
        if metadata["target_frames"] > 500 or not isinstance(metadata.get("head_config"), dict):
            raise ModelUnavailableError("Classifier frame count or head configuration is invalid.")
        vocabulary = metadata.get("vocabulary", [])
        if not isinstance(vocabulary, list) or len(vocabulary) < 2 or not all(isinstance(label, str) and label for label in vocabulary) or len(set(vocabulary)) != len(vocabulary):
            raise ModelUnavailableError("Classifier vocabulary is invalid.")
        encoder_path = self.loader._resolve_ckpt()
        if metadata.get("encoder_sha256") != sha256_file(encoder_path):
            raise ModelUnavailableError("The classifier was trained with different encoder weights.")
        self.loader.load()
        if metadata.get("preprocessing") != self.loader.preprocessing():
            raise ModelUnavailableError("Classifier preprocessing does not match this encoder.")
        config = metadata["head_config"]
        if config["input_dim"] != self.loader.d_out or config["num_classes"] != len(vocabulary):
            raise ModelUnavailableError("Classifier dimensions do not match the encoder or vocabulary.")
        head = TemporalClassificationHead(**config)
        head.load_state_dict(checkpoint["state_dict"], strict=True)
        self.head, self.metadata = head.eval(), metadata
        return self

    def predict_features(self, features):
        self.load()
        features = np.asarray(features)
        if features.shape != (self.metadata["target_frames"], self.loader.d_out) or not np.isfinite(features).all():
            raise ModelUnavailableError("Features do not match the trained temporal and embedding dimensions.")
        with torch.inference_mode():
            probabilities = self.head(torch.as_tensor(features, dtype=torch.float32)[None]).softmax(-1)[0]
        if not torch.isfinite(probabilities).all():
            raise ModelUnavailableError("Classifier returned invalid probabilities.")
        labels = self.metadata["vocabulary"]
        values, indices = probabilities.topk(min(3, len(labels)))
        return {"label": labels[indices[0].item()], "confidence": values[0].item(),
                "top3": [{"letter": labels[index], "prob": value} for index, value in zip(indices.tolist(), values.tolist())],
                "probabilities": [{"letter": label, "prob": value} for label, value in zip(labels, probabilities.tolist())]}

    def predict_frames(self, rois):
        self.load()
        features = self.loader.encode_frames(rois, target_frames=self.metadata["target_frames"])
        return self.predict_features(features)
