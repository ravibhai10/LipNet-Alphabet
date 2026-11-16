# Official GRID visual-only, speaker-independent dataset

Acquire authentic normal-quality MPG video and matching word alignments from the links actually published on the [official GRID index](https://spandh.dcs.shef.ac.uk/gridcorpus/). The publisher describes the corpus as freely available for research use; no broader redistribution or commercial grant is inferred. Speaker s21 has no video. Audio is not required by the project's frozen video-only AV-HuBERT path.

Run from the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts/download_grid_official.py --parallel 8 --max-minutes 45
```

The time limit applies to each invocation. Repeat the command to resume unfinished assets. Completed archives are checked and reused; partial archives use HTTP Range/If-Range when supported. A server ignoring Range causes a restart of the partial file, never an append of a full response to old bytes. HTML, unexpected HTTP statuses, size mismatches, failed ZIP CRCs, and invalid TAR alignments are rejected. Existing differing extracted files are preserved and reported as conflicts.

The acquisition manifest is `data/external/grid/download_manifest.json`; it records source/final URLs, byte counts, computed SHA-256 values, HTTP metadata when downloaded, archive verification, extraction status, and failures. The exact official index snapshot is retained beside it. No URL pattern is synthesized for acquisition.

Each speaker keeps its own `archives` and `corpus/TNN` directory. Existing s1 files are preserved. No new audio is downloaded. New-speaker `grid_manifest.jsonl` files use paths relative to that speaker's `corpus`, matching the existing structure. The old Zenodo-layout downloader is a separate legacy tool.

After acquisition finishes, choose a new output directory:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_grid_dataset.py --output data/processed/grid_multispeaker_v1 --workers 4
```

Preparation only snapshots speakers whose acquisition records show both assets verified and extracted. It fully decodes every source video, rejects premature EOF relative to the reported frame count, hashes source bytes, matches annotation stems, validates annotation ordering and six-token grammar, checks annotation duration against actual decoded video duration, and calculates decoded 64×64 grayscale sequence fingerprints. Container frame estimates that undercount a complete stream are retained; actual decoded counts control bounds. Source-hash-bound validation caches speed subsequent runs without changing sources.

Invalid videos and annotations remain in place and are listed in `video_inventory.jsonl`. Byte duplicates are screened globally. Matching decoded-thumbnail groups are conservatively screened as possible visual duplicates; they are not claimed to be identical at native resolution. Conflicting duplicate labels are quarantined. Identical transcripts spoken by different people are legitimate class overlap, not identical video content.

Output files:

- `grid_manifest.jsonl`: usable sentence rows in the existing `GridManifestEntry` schema, with paths relative to `data/external/grid`. The empty audio path explicitly describes a visual-only sample.
- `video_inventory.jsonl` and `duplicates.jsonl`: validation results and exclusion reasons.
- `inventory.json` and `DATASET_INVENTORY.md`: per-speaker counts, class coverage, exact speaker assignments, and totals.
- `avhubert_letters.jsonl`: real source-video annotations with `start_frame` and exclusive `end_frame`; paths are relative to this manifest's directory.
- `train.jsonl`, `valid.jsonl`, `test.jsonl`: separate partition manifests for inspection. The combined `avhubert_letters.jsonl` is the input to the current CLI, whose integrity check requires all three partitions.
- `smoke_annotations.jsonl`: one authentic A segment from each partition for feature-extraction execution checks only.

GRID's `.align` timestamps are in **1/25000-second units**. The annotated fourth content token is the letter. Silence and short pauses (`sil`, `sp`) are removed before selecting this token; no model prediction supplies a label. Frame bounds use floor(start × FPS) and ceil(end × FPS). The source videos are not copied, trimmed on disk, or re-encoded.

The measured frame interval is read and tracked before interpolation to the existing 29-frame AV-HuBERT input. Eye-axis alignment, mouth crop, 88×88 encoder crop, mean/std, and encoder fingerprint checks are preserved. Optional bounds are rejected for already-cropped ROI inputs until that path has an explicitly implemented segment reader.

Speakers are sorted numerically. Approximately 15% are held out for validation and another 15% for testing, with at least one speaker in each; training receives the rest. Assignment occurs after validation/deduplication. All evaluation labels must be present in training. Speaker IDs remain compatible with s1's convention (`grid_T01`, `grid_T02`, etc.). The exact selected IDs are recorded rather than inferred from file order during training.

Validate and run a small CPU smoke before committing to full extraction:

```powershell
.\.venv\Scripts\python.exe scripts/avhubert_classifier.py validate --manifest data/processed/grid_multispeaker_v1/avhubert_letters.jsonl

.\.venv-avhubert-cpu\Scripts\python.exe scripts/avhubert_classifier.py extract --manifest data/processed/grid_multispeaker_v1/smoke_annotations.jsonl --output runs/features/grid_smoke_new --frames 29
```

Full extraction, supervised-head training, and held-out evaluation are explicit separate commands:

```powershell
.\.venv-avhubert-cpu\Scripts\python.exe scripts/avhubert_classifier.py extract --manifest data/processed/grid_multispeaker_v1/avhubert_letters.jsonl --output runs/features/grid_letters_v1 --frames 29

.\.venv-avhubert-cpu\Scripts\python.exe scripts/avhubert_classifier.py train --features runs/features/grid_letters_v1 --output runs/avhubert/grid_letters_v1 --epochs 30 --batch-size 16 --lr 0.001 --warmup-epochs 5

.\.venv-avhubert-cpu\Scripts\python.exe scripts/avhubert_classifier.py evaluate --features runs/features/grid_letters_v1 --head runs/avhubert/grid_letters_v1/head.pt --output runs/evaluation/grid_letters_v1
```

All output directories must be new. Full CPU extraction can take many hours; no full extraction or training is launched by acquisition or preparation. Training selects the head using validation loss and reports the held-out test partition separately; evaluation writes per-class metrics and a confusion matrix.

Do **not** pass `--alphabet`: observed GRID letter slots omit W, so this is a 25-letter, coarticulated GRID classification task. A trained 25-class head is intentionally incompatible with the existing browser A–Z endpoint. Data availability and this workflow do not supply general-purpose English word recognition, sentence recognition, or a measured accuracy improvement.
