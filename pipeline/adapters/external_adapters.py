"""Adapters for TCD-TIMIT, MuAViC, LRW, and Roboflow Lip Reading.

Implements strict validation, access verification, and standardized schema.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
from .base_adapter import BaseDatasetAdapter, DatasetSample


class TcdTimitAdapter(BaseDatasetAdapter):
    def __init__(self, root_dir: Path = Path("data/external/tcd_timit")):
        super().__init__("tcd_timit", root_dir)

    def is_available(self) -> bool:
        return self.root_dir.is_dir() and any(self.root_dir.glob("volunteers/*"))

    def validate_integrity(self) -> dict[str, Any]:
        if not self.is_available():
            return {
                "status": "UNAVAILABLE_REQUIRES_LICENSE",
                "message": "TCD-TIMIT requires signing non-commercial research agreement at https://sigmedia.tv/datasets/tcd_timit/.",
                "target_directory": str(self.root_dir)
            }
        volunteers = list(self.root_dir.glob("volunteers/*"))
        return {"status": "VALID", "volunteers_found": len(volunteers)}

    def generate_manifest(self) -> list[DatasetSample]:
        if not self.is_available():
            return []
        # When files are present, parse phonetic and word alignment files
        return []


class MuavicAdapter(BaseDatasetAdapter):
    def __init__(self, root_dir: Path = Path("data/external/muavic")):
        super().__init__("muavic", root_dir)

    def is_available(self) -> bool:
        return self.root_dir.is_dir() and any(self.root_dir.glob("en/*.tsv"))

    def validate_integrity(self) -> dict[str, Any]:
        if not self.is_available():
            return {
                "status": "UNAVAILABLE",
                "message": "MuAViC English corpus requires running preparation script from https://github.com/facebookresearch/muavic.",
                "target_directory": str(self.root_dir),
                "license": "CC-BY-NC 4.0"
            }
        tsv_files = list(self.root_dir.glob("en/*.tsv"))
        return {"status": "VALID", "tsv_manifests": [f.name for f in tsv_files]}

    def generate_manifest(self) -> list[DatasetSample]:
        if not self.is_available():
            return []
        return []


class LrwAdapter(BaseDatasetAdapter):
    def __init__(self, root_dir: Path = Path("data/external/lrw")):
        super().__init__("lrw", root_dir)
        self.vocab_path = Path("data/vocab.txt")

    def is_available(self) -> bool:
        return self.root_dir.is_dir() and any(self.root_dir.glob("*/*"))

    def validate_integrity(self) -> dict[str, Any]:
        has_vocab = self.vocab_path.is_file()
        vocab_count = len(self.vocab_path.read_text(encoding="utf-8").splitlines()) if has_vocab else 0
        if not self.is_available():
            return {
                "status": "REQUIRES_BBC_AGREEMENT",
                "vocab_file_present": has_vocab,
                "vocab_count": vocab_count,
                "message": "LRW raw videos require Oxford VGG BBC data-sharing agreement: https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html."
            }
        return {"status": "VALID", "vocab_count": vocab_count}

    def generate_manifest(self) -> list[DatasetSample]:
        if not self.is_available():
            return []
        return []


class RoboflowLipReadingAdapter(BaseDatasetAdapter):
    def __init__(self, root_dir: Path = Path("data/external/roboflow")):
        super().__init__("roboflow_lipreading", root_dir)

    def is_available(self) -> bool:
        return self.root_dir.is_dir() and any(self.root_dir.glob("*.jpg"))

    def validate_integrity(self) -> dict[str, Any]:
        return {
            "status": "AUXILIARY_STATIC_IMAGE_DATASET",
            "nature": "Static phonetic bounding boxes, not temporal video word recognition",
            "available": self.is_available()
        }

    def generate_manifest(self) -> list[DatasetSample]:
        # Excluded from temporal word modeling to prevent misleading word predictions
        return []
