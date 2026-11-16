"""Master dataset preflight, status audit, and acquisition manager.

Audits:
- GRID Corpus (33 speakers)
- MIRACL-VC1 (15 speakers)
- TCD-TIMIT (59 speakers)
- MuAViC (English AVSR)
- Roboflow Lip Reading
- Oxford LRW 500-word dataset

Generates machine-readable status report: data/dataset_status_report.json
"""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.adapters import (
    GridWordAdapter,
    MiraclAdapter,
    TcdTimitAdapter,
    MuavicAdapter,
    LrwAdapter,
    RoboflowLipReadingAdapter
)


def audit_all_datasets():
    print("=" * 70)
    print("LIPNET-AI: MULTI-DATASET PREFLIGHT & ACQUISITION STATUS AUDIT")
    print("=" * 70)

    adapters = [
        GridWordAdapter(),
        MiraclAdapter(),
        TcdTimitAdapter(),
        MuavicAdapter(),
        LrwAdapter(),
        RoboflowLipReadingAdapter()
    ]

    report = {
        "datasets": {},
        "storage": {
            "root": str(ROOT),
            "free_gib": round(shutil.disk_usage(ROOT).free / (1024**3), 2)
        }
    }

    print(f"Free Disk Storage: {report['storage']['free_gib']} GiB\n")

    for adapter in adapters:
        name = adapter.dataset_name
        is_avail = adapter.is_available()
        integrity = adapter.validate_integrity()
        report["datasets"][name] = {
            "available_locally": is_avail,
            "integrity_details": integrity
        }

        status_str = "[VERIFIED ON DISK]" if is_avail else "[REQUIRES ACCESS/DOWNLOAD]"
        print(f"Dataset: {name:<22} Status: {status_str}")
        for k, v in integrity.items():
            print(f"   {k}: {v}")
        print("-" * 70)

    out_path = ROOT / "data" / "dataset_status_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nAudit report exported to: {out_path}")
    return report


if __name__ == "__main__":
    audit_all_datasets()
