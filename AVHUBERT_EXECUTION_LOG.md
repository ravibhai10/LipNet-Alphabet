# AV-HuBERT Execution Log

**Project:** LipNet-Alphabet  
**Root:** C:\Users\Dell\Downloads\LipNet-Alphabet-main\LipNet-Alphabet-main  
**Started:** 2026-10-25

## Baselines Recorded

| Asset | Path | SHA-256 | Size |
|-------|------|---------|------|
| alphabet_classifier.pt | models/alphabet_classifier.pt | 7c00d5117b7fae9a76c05980ca70edd680e79be4c8a152ea2595c98ded90c4c3 | 3,664,980 |
| grid_manifest.jsonl | data/external/grid/s1/grid_manifest.jsonl | ddd117696b50a20954aa2f81b3e895f8b8382e40bdf53d1d1f4dba99cf6ada5f | 408,660 |

### Environments
- `.venv`: Python 3.11.9, torch 2.2.2+cpu, opencv 4.11.0 — NO fairseq
- `.venv-avhubert`: Python 3.11.9, torch 2.2.2+cpu, opencv 4.10.0 — NO fairseq

### GRID Corpus Inventory
- `data/external/grid/s1/corpus/T01/`
  - `video/s1/`: 1001 .mpg files (~400KB each)
  - `audio/25kHz/s1/`: 1000 .wav files
  - `alignment/align/`: 1000 .align files (space-delimited: `start_ms end_ms word`)
- One extra video file (orphan without matching audio/alignment)
- `data/external/grid/s1/archives/`: s1_align.tar, s1_audio_25khz.tar, s1_video_normal.mpg_vcd.zip

### Manifest Issue
- `grid_manifest.jsonl` contains literal `\n` text instead of real newline characters — all ~1000 records concatenated on ONE physical line
- First sample: T01_Ubbaf2n, 75 frames, 25 FPS, transcript "bin blue at f two now"

## Work In Progress

### Step 1: Fix GRID manifest JSONL — COMPLETE
- Fixed literal `\n` → real newlines in `data/external/grid/s1/grid_manifest.jsonl`
- Wrote atomically (temp file → validate → replace)
- Old SHA-256: ddd117696b50a20954aa2f81b3e895f8b8382e40bdf53d1d1f4dba99cf6ada5f
- New SHA-256: 440a107ab2026e55c2d86a54b62356755442ac4cbe0905ff034f003879cf4d29
- 1000 records, each on its own physical line, all json.loads()-validated
- 1000 unique utterance IDs, all stems match, all files exist on disk
- First sample: T01_Ubbaf2n, 75 frames, 25.0 FPS, "bin blue at f two now" ✓
- Added regression test `test_persisted_manifest_uses_real_newlines` (14/14 tests pass)

### Step 2: AV-HuBERT / Fairseq compatibility — BLOCKED (external requirement)
- `.venv-avhubert`: Python 3.11.9, torch 2.2.2+cpu, opencv 4.10.0 — NO fairseq
- `pip install fairseq` → `Microsoft Visual C++ 14.0 or greater is required` (no MSVC build tools)
- `pip install "fairseq @ git+..."` → `git` not installed
- WSL2 not installed (requires admin + reboot — external requirement)
- No conda/mamba available
- All fairseq PyPI versions (0.6.1–0.12.2) are sdists requiring compilation → blocked
- **Resolution requires**: user to install git + MSVC Build Tools, OR install WSL2, OR provide fairseq wheel

### Step 3: Obtain AV-HuBERT checkpoint — BLOCKED (external requirement)
- `https://dl.fbaipublicfiles.com/av_hubert/avhubert_base.pt` → HTTP 403 Forbidden from this sandbox
- Official page: `https://facebookresearch.github.io/av_hubert/` requires clicking "I Accept" on Meta license agreement before downloads are available
- License: AV-HuBERT License Agreement (Meta Platforms, non-commercial research only)
- **Resolution requires**: user to manually accept license at official page, then download checkpoint and place at `models/external/avhubert/avhubert_base.pt`
- Download destination prepared: `models/external/avhubert/` directory exists (empty, gitignored)
- Verification command ready: `sha256sum models/external/avhubert/avhubert_base.pt`

## Real-Corpus Validation — COMPLETE
- Builder: 1000 entries from T01 corpus (1001 video files, 1 orphan without audio/alignment)
- Validator: 1000/1000 OK, 0 failed
- All stems match, all files exist on disk
- Fixed `pipeline/grid/validate.py`: added missing `_mark()` method, fixed frozen dataclass mutation (uses `dataclasses.replace`)

## Unit Tests
- 14/14 pass (pipeline/grid/tests/test_ingestion.py)
- 6/6 pass (pipeline/avhubert/tests/test_adapter.py)
- verify_phase2.py: ALL 11 modules PASSED (schema, manifest, ingestion, tracking, appearance, geometry, motion, fusion, validate, inspect, preprocess)

## Environment Setup Complete
- CPU-only Python environment: .venv-avhubert-cpu
- PyTorch CPU build installed
- All required packages: torch, sentencepiece, hydra-core, omegaconf, scipy, python-levenshtein, fairseq
- 345/345 project tests pass
- verify_phase2.py: ALL 11 modules PASSED
- Schema validation: TEST_SYNTHETIC_SOURCES accepted

## Cleanup
- Removed temp scripts: fix_manifest.py, validate_corpus.py

### Step 2: AV-HuBERT / Fairseq compatibility
- Windows/Python 3.11 cannot load fairseq (blocked)
- Need WSL2 Linux environment or alternative

### Step 3: Obtain AV-HuBERT checkpoint
- No checkpoint present; `dl.fbaipublicfiles.com` returns 403 from this sandbox
- Manual download required per AVHUBERT_PLAN.md

## 2026-10-27: Windows CPU encoder verified; training pipeline ready

This update supersedes the historical Fairseq/checkpoint blockers above.

- Downloaded the official `base_vox_iter5.pt` to `models/external/avhubert/` (1,240,185,329 bytes). Locally computed SHA-256: `dee8b0651dbea3a21cda0f2c52b956e3a878d5f6ee503d427d6a4044bb95c9cc`.
- Fixed package imports in AV-HuBERT and OmegaConf 2.3 checkpoint conversion in vendored Fairseq. Added `python_speech_features==0.6` to the CPU environment.
- Loaded the actual official encoder on Windows CPU and encoded a real GRID video using measured landmark alignment and the official grayscale/88-pixel preprocessing. Result: finite `[29,768]` features. Report: `.qa/avhubert-real-smoke.json`.
- Added real-data annotation audits, frozen feature caching, linear/MLP head training, best-validation checkpoint selection, held-out testing, and artifact-checked prediction.
- Added live mouth activity gating, aligned crops, stride-four sliding windows, full-vector EMA, and two-window confirmation while preserving the first-page palette.
- User chose: **finish the training pipeline; add data later**. No real classifier head was trained on local LRW fixtures or the single-speaker GRID subset. Current browser inference retains the original alphabet checkpoint.
- Dataset schema, commands, verification, and activation instructions: [ACCURACY_PIPELINE.md](ACCURACY_PIPELINE.md).

## 2026-10-27: Completion verification

- Rehashed both original checkpoints; both match the recorded SHA-256 values.
- Repeated real GRID video extraction in the existing CPU environment: finite `[29,768]` features, 11.14 seconds after encoder loading while other verification work was active. The earlier 4.75-second result remains a historical single-clip observation.
- Added exact feature shape/fingerprint checks, complete A–Z annotation validation, explicit recording-speaker assignment, and per-class classification report/confusion-matrix exports. Later test evaluation rejects the head's training speakers/content.
- Current inventory: 0 registered authentic alphabet recordings, 1,000 GRID utterances from one speaker, 45 generated LRW clips, no AV-HuBERT head. Authentic training and independent comparison remain blocked by missing data and original baseline provenance.
- Completed six-scene visitor story, retained technical workspace/exhibits, and separated service health from classifier/head/encoder/sentence readiness.
- Final checks: 424 Python tests passed, 3 optional-fixture tests skipped; 10 frontend checks passed; 52 browser checks passed; production build and actual HTTP/WebSocket/CPU smoke tests passed.
- Current handoff: [PROJECT_STATUS.md](PROJECT_STATUS.md), [VALIDATION_REPORT.md](VALIDATION_REPORT.md), and [MODEL_AVAILABILITY.md](MODEL_AVAILABILITY.md).
