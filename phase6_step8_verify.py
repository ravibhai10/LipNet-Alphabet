"""Phase 6 Step 0+8 verification (measured, no fabrication)."""
import json
import sys
from pathlib import Path
from collections import Counter

# ---------- STEP 8: GPU VERIFICATION (PyTorch) ----------
import torch
print("=" * 60)
print("STEP 8: GPU VERIFICATION (MEASURED, NOT CLAIMED)")
print("=" * 60)
print(f"PyTorch version:      {torch.__version__}")
print(f"Built with CUDA:      {torch.version.cuda}")
print(f"CUDA available:       {torch.cuda.is_available()}")
print(f"GPU device count:     {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU device name:      {torch.cuda.get_device_name(0)}")
    print(f"Actual training device: cuda:0")
else:
    print(f"GPU device name:      none")
    print(f"Actual training device: cpu")
print()

# ---------- STEP 0: REAL DATASET AUDIT (RE-VERIFY) ----------
root = Path("alphabet_dataset/recordings")
print("=" * 60)
print("STEP 0: REAL DATASET AUDIT (RE-VERIFIED, MEASURED)")
print("=" * 60)

real = 0
synth = 0
notes_render = 0
notes_webcam = 0
sources = Counter()
landmarks = Counter()
total = 0
for meta_path in root.rglob("metadata.json"):
    total += 1
    m = json.loads(meta_path.read_text())
    src = m.get("source", m.get("data_source", "?"))
    lm = m.get("landmark_source", "?")
    sources[src] += 1
    landmarks[lm] += 1
    if src == "real_camera":
        real += 1
    else:
        synth += 1
    note = m.get("notes", "")
    if "render-speaker" in note:
        notes_render += 1
    if "webcam" in note.lower() or "real human" in note.lower():
        notes_webcam += 1

print(f"Total recordings:                  {total}")
print(f"data_source == 'real_camera':      {real}")
print(f"data_source != 'real_camera':      {synth}  (i.e. synthetic/rendered)")
print(f"notes containing 'render-speaker': {notes_render}")
print(f"notes containing 'webcam'/'real':  {notes_webcam}")
print(f"source distribution:               {dict(sources)}")
print(f"landmark_source distribution:      {dict(landmarks)}")
print()

manifest_path = Path("alphabet_dataset/manifest.jsonl")
cache_path = Path("runs/repr_cache")
print(f"Manifest exists:      {manifest_path.exists()}  ({manifest_path})")
print(f"Phase 3 cache exists: {cache_path.exists()}  ({cache_path})")
print()

print("=" * 60)
print("HONEST CONCLUSION")
print("=" * 60)
if real == 0:
    print(f"REAL RECORDINGS: 0")
    print(f"The entire alphabet_dataset/ directory is procedurally-rendered")
    print(f"synthetic data (note 'render-speaker spkNN...' in every metadata).")
    print(f"No human webcam recordings exist. Steps 1-23 are BLOCKED on")
    print(f"real-data collection; they CANNOT be executed honestly without it.")
    sys.exit(2)
else:
    print(f"REAL RECORDINGS: {real}")
    print("Proceed with Step 1+.")
    sys.exit(0)
