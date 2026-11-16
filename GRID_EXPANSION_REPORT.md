# GRID corpus expansion and speaker-independent dataset

Client date: **2026-10-29**. Acquisition and dataset preparation are complete. No real model training or recognition-accuracy evaluation was launched as part of this acquisition task.

Subsequent separate work trained and evaluated a four-command **whole-word** experiment using the word alignments, not the letter-slot manifest. See [AUTOMATED_WORD_PIPELINE_REPORT.md](AUTOMATED_WORD_PIPELINE_REPORT.md). Its 8/32 test accuracy is chance-level and does not establish a useful recognizer. The acquisition and letter-slot results below are unchanged.

## Acquisition

Downloaded **32 additional authentic speakers**. Normal-quality video and matching word alignments are now present for **s1–s20 and s22–s34**, covering all **33 speakers with available video**. s21 was excluded because the [official GRID distribution](https://spandh.dcs.shef.ac.uk/gridcorpus/) states that its video is unavailable. Existing s1 data and audio were preserved; no additional audio was downloaded.

The downloader parsed actual normal-video and alignment links from the publisher's index. It did not construct unverified source URL patterns. All **66 video/alignment archives** passed verification: HTTP status/length and HTML rejection on acquisition, ZIP CRC checks, TAR member/annotation checks, SHA-256 calculation, and extraction checks. Interrupted video transfers demonstrably resumed using **HTTP 206** with matching byte ranges. No failed archives or incomplete `.part` downloads remained at acquisition completion.

Source provenance:

- Publisher: [The GRID audiovisual sentence corpus](https://spandh.dcs.shef.ac.uk/gridcorpus/).
- Conditions: the publisher describes the corpus as freely available for research use. No broader commercial or redistribution grant is inferred.
- Exact downloaded links, byte lengths, computed SHA-256 values, HTTP metadata, and status: `data/external/grid/download_manifest.json`.
- Saved publisher index and its fingerprint: `data/external/grid/official-index.html` and the acquisition manifest's `index_sha256`.
- No publisher archive checksums were supplied for comparison; recorded SHA-256 values are locally computed fingerprints.

Archive bytes, all 33 speakers: **13,592,536,622**. Extracted video/alignment bytes recorded by the archives: **13,873,648,010**. Additional file data excluding existing s1: **26,610,918,505 bytes** across new archives and extracted members (approximately 24.78 GiB of logical file data). Archives are retained for integrity checks and reproducible extraction. These totals exclude the pre-existing s1 audio and filesystem allocation overhead.

The machine has an Intel i3-1115G4 CPU with two physical cores/four logical processors. Acquisition began with approximately 355 GiB free. The post-acquisition free-space snapshot is retained in `.qa/grid-acquisition-summary.json`; full CPU feature extraction/training was deliberately deferred.

## Verified dataset and split results

**33,000 source videos; 32,891 usable video/annotation pairs; 109 excluded for premature decoder termination.** No word annotations are missing or orphaned. All 33 available speakers are represented. No additional available speakers remain to download.

All 32,891 usable source hashes and decoded-thumbnail hashes are unique; no duplicate clips remain across partitions. 15 complete streams with underestimated container frame counts were retained using actual decoded lengths. All exclusions remain in video_inventory.jsonl. No source video was deleted or replaced.

| Speaker | Videos | Usable | Excluded | Split |
| --- | ---: | ---: | ---: | --- |
| s1 | 1000 | 1000 | 0 | train |
| s2 | 1000 | 1000 | 0 | train |
| s3 | 1000 | 1000 | 0 | train |
| s4 | 1000 | 999 | 1 | train |
| s5 | 1000 | 1000 | 0 | train |
| s6 | 1000 | 1000 | 0 | train |
| s7 | 1000 | 1000 | 0 | train |
| s8 | 1000 | 892 | 108 | train |
| s9 | 1000 | 1000 | 0 | train |
| s10 | 1000 | 1000 | 0 | train |
| s11 | 1000 | 1000 | 0 | train |
| s12 | 1000 | 1000 | 0 | train |
| s13 | 1000 | 1000 | 0 | train |
| s14 | 1000 | 1000 | 0 | train |
| s15 | 1000 | 1000 | 0 | train |
| s16 | 1000 | 1000 | 0 | train |
| s17 | 1000 | 1000 | 0 | train |
| s18 | 1000 | 1000 | 0 | train |
| s19 | 1000 | 1000 | 0 | train |
| s20 | 1000 | 1000 | 0 | train |
| s22 | 1000 | 1000 | 0 | train |
| s23 | 1000 | 1000 | 0 | train |
| s24 | 1000 | 1000 | 0 | train |
| s25 | 1000 | 1000 | 0 | valid |
| s26 | 1000 | 1000 | 0 | valid |
| s27 | 1000 | 1000 | 0 | valid |
| s28 | 1000 | 1000 | 0 | valid |
| s29 | 1000 | 1000 | 0 | valid |
| s30 | 1000 | 1000 | 0 | test |
| s31 | 1000 | 1000 | 0 | test |
| s32 | 1000 | 1000 | 0 | test |
| s33 | 1000 | 1000 | 0 | test |
| s34 | 1000 | 1000 | 0 | test |

Exact assignments after validation and duplicate screening:

- **train**: s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s22, s23, s24 - 22,891 clips, 23 speakers.
- **valid**: s25, s26, s27, s28, s29 - 5,000 clips, 5 speakers.
- **test**: s30, s31, s32, s33, s34 - 5,000 clips, 5 speakers.

The 25 observed classes are A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, X, Y, Z. W is absent from all partitions. Each validation/test class occurs in training. Speakers, sample IDs, and source content do not overlap between partitions.

Frozen outputs:

- `data/processed/grid_multispeaker_v1/DATASET_INVENTORY.md` and `inventory.json`: per-speaker counts, class distributions, exact splits, and validation policy.
- `data/processed/grid_multispeaker_v1/grid_manifest.jsonl`: existing sentence schema; paths relative to `data/external/grid`.
- `data/processed/grid_multispeaker_v1/avhubert_letters.jsonl`: combined speaker-disjoint, frame-bounded training manifest.
- `data/processed/grid_multispeaker_v1/train.jsonl`, `valid.jsonl`, `test.jsonl`: partition manifests for inspection.
- `video_inventory.jsonl`, `duplicates.jsonl`, `smoke_annotations.jsonl`: validation details, duplicate findings, bounded smoke samples.

Combined annotation manifest SHA-256: `843cb6c50cc97d588136ed1c79f6d1bc4324990d5dc83679c75ee1dbfac5e637`.

## Task and annotation boundaries

The supervised task is the **annotated GRID letter slot in coarticulated sentences**, not sentence recognition or general-purpose English words. GRID word alignments use **25,000 timestamp units per second**. Silence/pause tokens `sil` and `sp` are removed before selecting the fourth content token. No label is inferred from a model prediction or a filename.

`avhubert_letters.jsonl` points to the original full-face video with integer `start_frame` and exclusive `end_frame` bounds. The source video is not copied, trimmed on disk, or re-encoded. The CLI tracks only the selected source interval, then applies the existing eye-axis/mouth alignment and 29-frame interpolation. Encoder crop, mean/std, dimension, and checkpoint compatibility remain unchanged.

Observed GRID letter labels omit W. A trained head for this dataset is therefore a **25-class GRID experiment** and must not be advertised as a complete A–Z model. Do not use `--alphabet`, which correctly requires all 26 letters. The existing live browser endpoint intentionally rejects a head whose vocabulary is not exactly A–Z.

Speaker-independent partitions apply to future training runs using these explicit assignments. They do not prove that any historical checkpoint's undocumented training speakers were disjoint from these held-out speakers. No accuracy improvement is established by acquiring data or obtaining finite encoder features.

## Validation and reproducibility

- Full Python regression suite before the frame-metadata refinement: **437 passed, 3 skipped**, 61.82 seconds. Skips require the absent optional `synthetic_dataset` fixture. Log: `.qa/grid-expansion-tests.txt`.
- A further regression specifically checks that a fully decoded 75-frame stream is not rejected for a 74-frame container estimate, while premature EOF remains an error. The final targeted suite passed **59 tests** in 41.17 seconds; log: `.qa/grid-expansion-final-tests.txt`.
- Real preflight extraction succeeded for **s3, s4, and s5** through MediaPipe alignment and the official frozen AV-HuBERT CPU encoder. Each returned finite **29 × 768** features. Cache: `runs/features/grid_expansion_smoke_20261029`; these are execution checks, not training or accuracy results.
- The existing `GridManifestEntry` schema and `GridValidator` loaded genuine rows from **all 33 speakers** without requiring audio. Final schema, content uniqueness, bounds, preservation, and split counts are recorded in `.qa/grid-final-integrity.json`.
- The actual classifier CLI rehashed and validated **all 32,891** source annotations successfully. It confirmed 33 separate speakers, the 25-letter vocabulary, and 22,891/5,000/5,000 train/validation/test rows. Both evaluation partitions contain exactly 200 examples per class. Raw result: `.qa/grid-final-annotation-validation.json`.
- The existing corpus inventory reader now reports **32,891 utterances across 33 speakers**, matching the frozen manifest. Four newly generated interim speaker summaries were refreshed after the frame-count refinement; their earlier derived copies are retained in `.qa/grid-interim-speaker-manifests`. The original s1 manifest remains unchanged. Integration proof: `.qa/grid-integration-evidence.json`.
- A second real CPU smoke used the **final** training/validation/test speakers: s11/T11_Usrwa8n (train), s28/T28_Upgia7n (valid), s31/T31_Upwba8p (test). All three yielded finite 29 x 768 features. Proof: `.qa/grid-final-smoke.json`; cache: `runs/features/grid_expansion_final_smoke_20261029`. Neither smoke trained a classifier or measured recognition.
- Existing model checkpoints, s1 source archives, and the original s1 manifest have preservation fingerprints in `.qa/grid-preservation-before.json`. All nine recorded files matched exactly after expansion, including all five recognition/representation checkpoints. Proof: `.qa/grid-final-integrity.json`.
- Passing OpenCV decoding means the clip is accepted by this runtime. MPEG decoder concealment warnings were observed; archive CRC success is distinct from visual or linguistic quality assurance. All exclusion reasons remain inspectable rather than being replaced with synthetic data.

New/updated source files: `scripts/download_grid_official.py`, `scripts/prepare_grid_dataset.py`, `tracking/mouth_canonicalization.py`, `pipeline/avhubert/classification.py`, `pipeline/avhubert/loader.py`, `scripts/avhubert_classifier.py`, `tests/test_grid_expansion.py`, `pipeline/grid/MULTISPEAKER.md`, `GRID_ACCESS.md`, and `.gitignore`. Existing frontend work and backend model selection were left untouched.

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/download_grid_official.py --parallel 8 --max-minutes 45

.\.venv\Scripts\python.exe scripts/prepare_grid_dataset.py --output data/processed/grid_multispeaker_v1 --workers 4

.\.venv\Scripts\python.exe scripts/avhubert_classifier.py validate --manifest data/processed/grid_multispeaker_v1/avhubert_letters.jsonl
```

Download reruns reuse verified archives and resume partial files. Preparation and all training/evaluation outputs are protected: choose a new output directory when repeating an already completed run.

## Exact future extraction, training, and held-out evaluation commands

These commands are provided for an explicit future run; **full extraction and training have not been started**. Full extraction can take many hours on this CPU. Validation/test features receive no random training augmentation.

```powershell
.\.venv-avhubert-cpu\Scripts\python.exe scripts/avhubert_classifier.py extract --manifest data/processed/grid_multispeaker_v1/avhubert_letters.jsonl --output runs/features/grid_letters_v1 --frames 29

.\.venv-avhubert-cpu\Scripts\python.exe scripts/avhubert_classifier.py train --features runs/features/grid_letters_v1 --output runs/avhubert/grid_letters_v1 --epochs 30 --batch-size 16 --lr 0.001 --warmup-epochs 5

.\.venv-avhubert-cpu\Scripts\python.exe scripts/avhubert_classifier.py evaluate --features runs/features/grid_letters_v1 --head runs/avhubert/grid_letters_v1/head.pt --output runs/evaluation/grid_letters_v1
```

The training command selects the head by validation loss, then evaluates the held-out test partition. The separate evaluation command reports only held-out test rows, checks encoder/preprocessing consistency and training speaker/content leakage, and writes class metrics and a confusion matrix. No backend environment variable should be changed to activate this 25-class experiment as the existing 26-class browser model.
