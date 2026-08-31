"""Phase 6 Step 0: Real dataset audit."""
import json
from pathlib import Path
from collections import Counter

root = Path("alphabet_dataset/recordings")
total = 0
speakers = set()
sessions = set()
letter_counts = Counter()
source_counts = Counter()
lm_counts = Counter()
issues = []

for ld in sorted(root.iterdir()):
    if not ld.is_dir() or not ld.name.isalpha():
        continue
    L = ld.name.upper()
    for spk in ld.iterdir():
        if not spk.is_dir():
            continue
        speakers.add(spk.name)
        for sess in spk.iterdir():
            if not sess.is_dir():
                continue
            sessions.add(sess.name)
            for rec in sess.iterdir():
                if not rec.is_dir():
                    continue
                total += 1
                letter_counts[L] += 1
                meta = rec / "metadata.json"
                if meta.exists():
                    m = json.loads(meta.read_text())
                    src = m.get("source", "?")
                    lm = m.get("landmark_source", "?")
                    source_counts[src] += 1
                    lm_counts[lm] += 1
                    if src != "real_camera":
                        issues.append(f"{L}/{rec.name}: source={src}")
                    if lm != "mediapipe":
                        issues.append(f"{L}/{rec.name}: lm={lm}")
                else:
                    issues.append(f"{L}/{rec.name}: no metadata")

mc = min(letter_counts.values())
mx = max(letter_counts.values())
imb = mx / mc if mc else 0

print("=== PHASE 6 STEP 0: REAL DATASET AUDIT ===")
print(f"Total recordings: {total}")
print(f"Real speakers: {len(speakers)}")
print(f"Real sessions: {len(sessions)}")
print(f"Data source dist: {dict(source_counts)}")
print(f"Landmark source dist: {dict(lm_counts)}")
print(f"A-Z distribution: {dict(sorted(letter_counts.items()))}")
print(f"Min class: {mc}  Max class: {mx}  Imbalance: {imb:.2f}x")
print(f"Issues: {len(issues)}")
for i in issues[:10]:
    print(f"  {i}")
