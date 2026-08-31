# PHASE 6 — REAL ALPHABET DATASET + REAL-DATA TRAINING
Date: 2026-08-28
Status: **REAL DATASET: NOT READY — SYNTHETIC DATA WAS GENERATED — HARD DATA-SOURCE GATE NOW ACTIVE**

---

## 1. EXACT NATURE OF generate_dataset.py (audited)

**Conclusion: 100% SYNTHETIC / RENDERED. NOT real human recordings.**

Evidence (from the source code, lines 229–318):

```python
def render_frame(letter, frame_idx, total_frames, speaker, session):
    img = np.full((HEIGHT, WIDTH, 3), 210, np.uint8)   # BLANK CANVAS
    # ... procedural face via cv2.ellipse() with hardcoded skin/hair/eye colors
    cv2.ellipse(img, (cx, cy), (head_w // 2, head_h // 2), 0, 0, 360, skin, -1)
    # ... procedural eyes, eyebrows, nose, mouth using cv2 primitives
    openness, width_scale, teeth, curve = get_mouth_shape_at_time(letter, ...)
    # ... articulated mouth from PHONE_SHAPES + LETTER_PHONE_STAGES dicts
```

| Question | Answer | Evidence |
|---|---|---|
| Is the video generated synthetically? | **YES** | `render_frame()` paints a face onto a blank canvas |
| Are the faces rendered? | **YES** | `cv2.ellipse` + `cv2.fillConvexPoly` for hair, skin, eyes, nose, mouth |
| Are the speakers real people? | **NO** | 22 "speakers" are random combinations of SKIN_PALETTE, HAIR_PALETTE, EYE_PALETTE and random face-shape ratios |
| Are the mouth movements captured from real recordings? | **NO** | Mouth shape is mathematically interpolated from PHONE_SHAPES dict (openness, width_scale, teeth, curve) |
| Are landmarks generated or detected by MediaPipe? | **DETECTED** by MediaPipe — but on rendered images | The pipeline is honest about this: `landmark_source="mediapipe"` because MediaPipe *does* run on the rendered images. But the SOURCE VIDEO IS NOT REAL. |
| Are frames generated procedurally? | **YES** | All 90 frames per recording are drawn by `render_frame()` |

**Honest classification**: The `generate_dataset.py` script produces a **synthetic dataset** where the only "real" part is that MediaPipe FaceLandmarker runs on rendered images. The video source is fully procedural. This is acceptable for **unit tests, pipeline smoke tests, debugging, and architecture connectivity tests** — but it is **NOT** real human data.

---

## 2. WERE THE 1,144 RECORDINGS REAL OR SYNTHETIC?

**100% SYNTHETIC.** All 26 letters × 22 speakers × 2 sessions = 1,144 recordings in `alphabet_dataset/recordings/` were produced by `generate_dataset.py`, which procedurally renders face images.

These recordings **MUST NOT** be used as the primary real-data training, validation, or test dataset.

---

## 3. WAS THE GENERATION STOPPED?

**YES — by background-task teardown.**
- Background task `bb0suonbi` (initial full generation): **STOPPED** (no completion record, process ended)
- Background task `ble8g9i0u` (continuation for letters N–Z): **completed (exit 0)** before this audit — generated the final 13 letters of synthetic data

The 1,144 synthetic recordings remain on disk in `alphabet_dataset/recordings/`. They have been left in place (per §4: "do not delete anything without confirming whether the generated data is needed for tests") but they are **quarantined** by the new hard data-source gate (see §4 below).

---

## 4. DATA-SOURCE SAFETY IMPLEMENTATION

### Schema additions (`data/schema.py`):
```python
DATA_SOURCE_REAL_CAMERA = "real_camera"           # Production-only
DATA_SOURCE_SYNTHETIC_TEST = "synthetic_test"     # Test-only, rejected in production
REAL_DATA_ONLY_SOURCES = {DATA_SOURCE_REAL_CAMERA}
TEST_SYNTHETIC_SOURCES = {DATA_SOURCE_SYNTHETIC_TEST}
```

### Recording metadata (`RecordingMetadata`):
New required field `data_source: str = "real_camera"` — production training/validation/test MUST use `"real_camera"`.

### Manifest entry (`ManifestEntry`):
Same field propagated from metadata; manifest validator enforces it.

### Validator (`data/validate.py`):
New check `_check_data_source(entries)` that:
- **REJECTS** `data_source == "synthetic_test"` in any production dataset
- **REJECTS** missing or unknown `data_source` values
- Returns **CRITICAL ERROR** if any non-real recording is found

This check runs in `validate_all()` immediately after the landmark-source check. It cross-checks both the manifest entry AND the per-recording `metadata.json`.

### Result when the validator runs on `alphabet_dataset/`:
The 1,144 synthetic recordings currently on disk do **NOT** have `data_source` set in their metadata (they were generated before this field existed). The validator will flag every one of them. They cannot enter the production training/validation/test pipeline.

---

## 5. REAL DATASET CURRENTLY AVAILABLE

| Letter | Real recordings | Real speakers | Real sessions |
|---|---|---|---|
| A | 2 | 2 (spk01, spk02) | 1 (session_001) |
| B | 2 | 2 | 1 |
| C | 2 | 2 | 1 |
| D–Z | **0** | 0 | 0 |

**Total: 6 real recordings** (only A/B/C in `alphabet_dataset_smoke/recordings/`).
The 1,144 recordings in `alphabet_dataset/recordings/` are synthetic and DO NOT COUNT.

---

## 6. REAL SPEAKERS: 2 (spk01, spk02)

These two speakers are also procedural render-templates (the smoke was generated by the same `generate_dataset.py` logic). However, the smoke's `metadata.json` files DO pass the real-MediaPipe landmark test because MediaPipe DID run on the rendered images — but the source video is still synthetic.

**Strictly honest: there are currently ZERO real human recordings in the project.**

---

## 7. REAL RECORDINGS: 0 (no real human video collected)

---

## 8. PER-LETTER COUNTS (real): A=2, B=2, C=2 (all from synthetic smoke; 0 truly real)

---

## 9. SPEAKER-INDEPENDENT SPLIT

The frozen `manifest.create_splits()` enforces 70/15/15 by speaker, sessions never span splits. This machinery exists and is correct, but **cannot be applied to a non-existent real dataset**.

---

## 10. REAL-DATA SANITY RESULT

`alphabet/sanity.py::real_data_sanity()` returned **PASS** when run on the synthetic smoke dataset (3-letter, 2-speaker). The pipeline (Phase 2 → Phase 3 cache → Phase 4 → Phase 5) works end-to-end.

**This pass is on synthetic data.** It demonstrates pipeline wiring. It is NOT evidence of real-data learning.

---

## 11. TINY-REAL-OVERFIT RESULT

**NOT EXECUTED.** Spec §21: tiny-real-overfit must pass on a TINY REAL dataset. There is no real dataset, so this step cannot be run. **Training is BLOCKED.**

---

## 12. FULL TRAINING STATUS

**BLOCKED — no real human recordings exist.**
- Spec §20: Step 5 (full training) requires Step 4 (tiny real overfit) to PASS
- Spec §21: failure policy is strict — STOP, do not fabricate
- No synthetic dataset can substitute for the real overfit gate

---

## CURRENT PHASE STATUS (§14)

> **REAL DATASET: NOT READY**
> **PIPELINE: VERIFIED** (wiring correct on synthetic smoke)
> **SYNTHETIC TEST DATA: VERIFIED** (1,144 generated, quarantined by data-source gate)
> **REAL TRAINING: BLOCKED UNTIL REAL RECORDINGS EXIST**

---

## REQUIRED NEXT STEPS (when real recordings are available)

1. **Collect real human recordings**: A–Z × 3–5 speakers × 3–5 repetitions = 234–650 real recordings minimum
2. Each recording: ONE spoken letter, onset/articulation/release preserved
3. Genuinely different speakers (no fake ID inflation)
4. Multiple sessions per speaker if possible
5. Run `python -m alphabet validate-dataset alphabet_dataset` — must pass with `data_source="real_camera"` for every entry
6. Re-run `real_data_sanity` on a real subset (3–5 recs/letter)
7. Run `tiny_overfit(use_synthetic=False)` on real data — must reach loss≈0, accuracy=1.0
8. Only THEN run `train_decoder` with frozen Phase 2–5

---

## FINAL HONEST RESPONSE (§22)

| Field | Value |
|---|---|
| Real recordings used | **0** |
| Synthetic recordings used | **1,144 (quarantined)** + 6 (smoke) |
| Pipeline end-to-end (synthetic) | **PASS** |
| Real-data sanity on real data | **N/A — no real data** |
| Tiny real overfit | **NOT EXECUTED** |
| Full training | **NOT STARTED** |
| Best val loss / test accuracy | **N/A** |
| Confusion pairs | **N/A** |
| Hardware device | **N/A (not entered training)** |
| Training throughput / inference latency | **N/A** |

**This is NOT a "production-ready" pipeline. It is a verified pipeline that has never been trained on real human data, because no real human data has been collected.**

---

## HARD DATA-SOURCE GATE (now active)

The `data/validate.py::_check_data_source` validator now rejects any production dataset entry that is not `data_source="real_camera"`. The 1,144 synthetic recordings cannot enter training/validation/test even if accidentally present.
