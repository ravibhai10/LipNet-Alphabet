"""Phase 6 Step 0: Real dataset audit + manifest generation + validation."""
import sys
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(".").resolve()))

from data.manifest import ManifestManager, ManifestConfig
from data.validate import DatasetValidator

# ── Step 0a: Audit recordings ────────────────────────────────────────────────
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
print(f"Issues (non-real_camera): {len(issues)}")
for i in issues[:5]:
    print(f"  {i}")

# ── Step 0b: Generate manifest ───────────────────────────────────────────────
manifest_path = Path("alphabet_dataset/manifest.jsonl")
cfg = ManifestConfig(dataset_root=Path("alphabet_dataset"), manifest_path=manifest_path)
mgr = ManifestManager(cfg)
entries = mgr.build_manifest()
mgr.save_manifest(entries)
print(f"\nManifest: saved {len(entries)} entries to {manifest_path}")

# ── Step 0c: Run validator ───────────────────────────────────────────────────
v = DatasetValidator(Path("alphabet_dataset"))
report = v.validate_all()
print(f"\nValidation: critical_errors={report.has_critical_errors()}")
for r in report.results:
    status = "PASS" if r.passed else "FAIL"
    print(f"  {status} [{r.severity}] {r.check_name}: {r.message[:100]}")

# ── Step 0d: Data source gate check ─────────────────────────────────────────
ds_results = [r for r in report.results if "data_source" in r.check_name]
if ds_results:
    r = ds_results[0]
    print(f"\nDATA SOURCE GATE: {'PASS' if r.passed else 'FAIL'}")
    print(f"  {r.message}")
else:
    print("\nDATA SOURCE GATE: NOT RUN (manifest was empty)")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n=== STEP 0 SUMMARY ===")
print(f"DATASET: {total} recordings, {len(speakers)} speakers, {len(sessions)} sessions")
print(f"REAL CAMERA RECORDINGS: {source_counts.get('real_camera', 0)}")
print(f"SYNTHETIC RECORDINGS: {source_counts.get('local_mp4', 0)}")
print(f"A-Z BALANCE: min={mc}, max={mx}, ratio={imb:.2f}x")
if source_counts.get("real_camera", 0) == 0:
    print("BLOCKER: No real camera recordings exist. Training cannot proceed.")
else:
    print("OK: Real camera recordings found.")
