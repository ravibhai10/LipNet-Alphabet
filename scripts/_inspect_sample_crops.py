import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import json
import numpy as np
import cv2

root = Path("runs/word_recognition_v1")
cfg = json.loads(Path("pipeline/word_recognition/config.cpu.json").read_text())
prep_root = Path(cfg["processed_root"]) / "80ff4af3decb1bdd"

with open(root / "words.jsonl") as f:
    rows = [json.loads(line) for line in f]

train_rows = [r for r in rows if r["split"] == "train"]

print("=== INSPECTING CROP PROPERTIES ACROSS CLASSES AND SPEAKERS ===")
for word in cfg["words"]:
    w_rows = [r for r in train_rows if r["word"] == word]
    print(f"\nWord: '{word}' (N={len(w_rows)} in train)")
    for r in w_rows[:3]:
        p = prep_root / r["frames_path"]
        frames = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in sorted(p.glob("frame_*.png"))]
        clip = np.stack(frames) # [24, 64, 64]
        # Frame difference across time: measures mouth motion
        motion = np.abs(np.diff(clip.astype(float), axis=0)).mean()
        center_patch = clip[:, 20:44, 20:44]
        print(f"  Speaker {r['speaker']:4s} | ID: {r['id']:15s} | Mean={clip.mean():.1f}, Std={clip.std():.1f}, Motion={motion:.2f}, CenterMotion={np.abs(np.diff(center_patch.astype(float), axis=0)).mean():.2f}")
