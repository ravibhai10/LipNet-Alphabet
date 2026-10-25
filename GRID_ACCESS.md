# GRID Corpus Access & Acquisition Plan

**Date:** 2026-10-25  
**Official source:** https://spandh.dcs.shef.ac.uk/gridcorpus/  
**License / access terms:** "Freely available for research use." Per-talk files are downloadable separately from the official site (audio, video, word alignments). **No aggregate unlock button is present** — each talker's archive is fetched individually.

## Access procedure (official)

1. Visit https://spandh.dcs.shef.ac.uk/gridcorpus/
2. Choose a talker (1-34). Note: **talker 21 has no video** (technical oversight).
3. For each talker, download the three components individually:
   - **50 kHz raw audio** (~300 MB/tlk)
   - **25 kHz endpointed audio** (to resample to 16 kHz for AV-HuBERT/FAIRseq pipelines)
   - **Normal-quality video** 360×288 (~480 MB/tlk), or **high-quality video** 720×576 (~1.2 GB/tlk)
   - **Word alignments** (380 KB/tlk, ASCII grid-aligned text)
4. Transcribed sentences are of the form: **"put red at G9 now"** (color, letter, number, adverb, verb). 1000 sentences × 34 talkers.

## Storage plan

Only the **endpointed 25 kHz audio** is needed for AV-HuBERT (resampled to 16 kHz). 34 talkers × ~100 MB ≈ **3.4 GB** for audio-only, plus optional high-quality video (~40 GB). **Storage is not a blocker** (401 GB free), but manual authorization/download is required per-talk, so the corpus cannot be auto-fetched.

## Existing project status

- `data/external/grid/` has been created but **is empty** — no GRID files exist anywhere in the repo.
- `alphabet_dataset/` holds real webcam recordings only; it is **not** a GRID subset.
- `data/schema.py`, `data/manifest.py`, `data/validate.py`, `data/preprocess.py` already implement a **recording-centric** manifest (letter/speaker/session/recording directories) and a GRID-friendly alignment parser will be added under `pipeline/grid/`.

## Validation coverage (expected)

Once actual GRID files are placed under `data/external/grid/`:
1. Verify the final directory tree (talk / audio / video / alignment files).
2. Parse word-alignment/CSV alignment files and validate utterance IDs.
3. Build a `grid_manifest.jsonl` with: speaker_id, utterance_id, video_path, audio_path, frame_count, fps, transcript, alignment_path, validation_status.
4. Run `pipeline/grid/validate.py` — missing files, malformed alignments, duplicate IDs, corrupt media.

## Summary (verified: 2026-10-25)

- Official URL confirmed: https://spandh.dcs.shef.ac.uk/gridcorpus/
- License: non-commercial research use, freely available.
- No GRID files present in the project; per-talk manual download required.
- Storage safe (401 GB free); corpus to be placed in `data/external/grid/` without touching `alphabet_dataset/`.
- Access blocker: **manual per-talk download** required — no automated authorized endpoint exists in this sandbox.
