import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from fastapi.testclient import TestClient
import numpy as np
import cv2
import json
import base64

from app.main import app

client = TestClient(app)

print("=== 1. TEST /api/models ===")
res = client.get("/api/models")
print(f"Status: {res.status_code}")
print(json.dumps(res.json(), indent=2))

print("\n=== 2. TEST /api/word/health ===")
res = client.get("/api/word/health")
print(f"Status: {res.status_code}")
print(json.dumps(res.json(), indent=2))

print("\n=== 3. TEST /api/predict/word WITH REAL MOUTH CROPS ===")
root = Path("runs/word_recognition_v1")
prep_root = Path("data/processed/grid_whole_words/80ff4af3decb1bdd")

with open(root / "words.jsonl") as f:
    rows = [json.loads(line) for line in f]

for target_word in ["bin", "set", "place"]:
    sample = next(r for r in rows if r["split"] == "test" and r["word"] == target_word)
    frames_dir = prep_root / sample["frames_path"]
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    
    b64_frames = []
    for fp in frame_files:
        img = cv2.imread(str(fp))
        _, enc = cv2.imencode(".jpg", img)
        b64_str = base64.b64encode(enc.tobytes()).decode("ascii")
        b64_frames.append(f"data:image/jpeg;base64,{b64_str}")
        
    res = client.post("/api/predict/word", json={"frames": b64_frames})
    print(f"\nTarget Word: '{target_word}' (Speaker {sample['speaker']})")
    print(f"  Response Status: {res.status_code}")
    print(f"  Prediction Body: {json.dumps(res.json(), indent=2)}")
