# Pipeline: GRID corpus ingestion & manifest builder

Owner scope: new files under `pipeline/grid/` (and `pipeline/grid/tests/fixtures/`).

**Goal:** consume authorized GRID corpus files (audio/video/alignment per talker) into a machine-readable manifest without modifying the read-only source corpus.

## Input format (official GRID, from GRID_ACCESS.md)

Each talker's archive contains:
- `audio/25kHz/` — endpointed audio, 25 kHz mono (resample to 16 kHz for AV-HuBERT)
- `audio/raw/` — 50 kHz raw audio (optional, not needed for this project)
- `video/` — normal 360×288 (~480 MB) or high 720×576 (~1.2 GB)
- `alignment/` — word alignments (380 KB), one file per talker

Transcripts: sentences of the form `"put red at G9 now"` (color, letter, number, adverb, verb). 1000 sentences per talker × 34 talkers.

## Output: `grid_manifest.jsonl`

One JSON object per utterance (screening/validation entry). Fields:

| Field | Type | Description |
|---|---|---|
| `talk_id` | str | GRID talker id, e.g. `T01` |
| `speaker_id` | str | derived speaker, `grid_t01` |
| `utterance_id` | str | GRID utterance id, e.g. `T01_U0001` |
| `video_path` | str | relative path to video file |
| `audio_path` | str | relative path to resampled audio (16 kHz) |
| `alignment_path` | str | relative path to alignment file |
| `frame_count` | int | video frame count (0 / null if unavailable) |
| `fps` | float | video FPS (0 / null if unavailable) |
| `transcript` | str | cleaned sentence transcript |
| `alignment_status` | str | `ok` \| `missing` \| `malformed` |
| `validation_status` | str | `ok` \| `missing_video` \| `missing_audio` \| `missing_alignment` \| `corrupt_media` |
| `value` | str | parsed semantic value, e.g. `put red at 9 now` |

## Modules

- `pipeline/grid/manifest.py` — `GridManifest`, `GridManifestBuilder`, `GridAlignmentParser`, `GridValidationReport`.
- `pipeline/grid/validate.py` — reconstruction of per-file validation (missing files, malformed alignments, duplicate IDs, corrupt/unreadable media) — mirrors `data/validate.py` style.
- `pipeline/grid/tests/fixtures/` — synthetic GRID-like fixture trees (audio/video/alignment files written as tiny placeholder files) + a `conftest.py` that builds them in a temp dir.

## Ingestion strategy
- Source corpus files are **never modified**. A `processed/` sibling record (never touching source) tracks success.
- Manifest builder emits `grid_manifest.jsonl`; no download is performed by code (download is manual/authorized).

## Tests (no complete GRID corpus required)
- Synthetic fixture registry in `tests/fixtures/` + unit tests for `GridAlignmentParser`, `GridManifestBuilder`, duplicate-ID detection, missing-file detection, and `validation_status` assignment.
- Real-data validation is best-effort pending authorized downloads.
