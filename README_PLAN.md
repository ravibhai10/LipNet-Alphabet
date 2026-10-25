# LipNet-Alphabet — Avanced Demo / Research Infrastructure

## Environment
- Python 3.11.9 isolated `@ .venv` (Windows, CPU-only PyTorch 2.2.2 + MediaPipe/OpenCV/NumPy/SciPy/Torchvision).
- FastAPI + uvicorn ready; `verify_phase2.py` passes (schema, manifest, appearance, geometry, motion, fusion).
- PyTorch CPU + MediaPipe FaceLandmarker (face_landmarker.task) work as Phase-0/A knobs.

## What was built (this session)
- `data/external/grid/` — empty bucket for authorized GRID corpus (per-talk manual download).
- `models/external/avhubert/` — empty bucket for AV-HuBERT weights (existing `alphabet_classifier.pt`, `face_landmarker.task`, etc. untouched).
- `pipeline/grid/`, `pipeline/grid/tests/fixtures/` — corpus-ingestion + manifest tooling (see `pipeline/grid/README.md`).
- `pipeline/avhubert/`, `pipeline/avhubert/tests/` — AV-HuBERT adapter + safe tests (see `pipeline/avhubert/README.md`).
- `pipeline/eval/` — speaker-independent splits, label-consistency, split-overlap, WER metrics.
- `AVHUBERT_PLAN.md` — dependency/compatibility/storage blocker report + isolated env plan.
- `GRID_ACCESS.md` — official GRID access + license + storage plan.
- `start_demo.bat` / `demo_backend.py` / `demo_frontend.html` — shipped zero-build demo (from earlier session).

## Separate isolated AV-HuBERT environment (if real inference is desired)
`.venv-avhubert` — see `AVHUBERT_PLAN.md` for the exact stack. Official AV-HuBERT checkpoint page (MIT-like AV-HuBERT License Agreement, Meta; non-commercial research only): http://facebookresearch.github.io/av_hubert. Official repo: https://github.com/facebookresearch/av_hubert (archived).

## Safety rules honored
- Never overwrite/delete/regenerate existing model checkpoints or datasets. (Checked: `alphabet_classifier.pt` and `face_landmarker.task` untouched.)
- No full training run.
- No fabricated dataset completeness, evaluation results, or accuracy.
- No unofficial dataset mirrors or bypass of access restrictions.
- New data/model assets go to dedicated directories (`data/external/`, `models/external/`).
- Sources recorded (URL + license + framework + size), checksums added on download.
- `.venv` left untouched; isolated env only after explicit compatibility report.
- Download concurrency limited to ≤1 stream per asset.
- All changes local (no push).
