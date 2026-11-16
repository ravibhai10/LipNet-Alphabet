# Reproduction and API guide

Run from the nested repository directory, not its parent Downloads folder.
Original checkpoints and evaluations are protected. Choose a new output path
for every experiment/audit; do not overwrite completed results.

## Start the evaluated word path

```powershell
.\start_word_demo.ps1 -Port 8001
```

Open `http://127.0.0.1:8001/demo/live`. Start Camera → Start Word → visibly speak
one supported word → Finish Word → Next Word or Retry. Camera permission is
requested only after clicking. Words: choose, connection, hello, navigation,
next, previous, start, stop. Collection labels do not change the model vocabulary.

The launcher uses `.venv-avhubert-cpu`, preloads the selected models once, and
restores its environment variables on exit. It never stops an existing server.
Use an unused port. Use `LIPNET_MODEL_BACKEND=alphabet` for the preserved alphabet
model; `LIPNET_WORD_BACKEND=avhubert` selects the separate word model. A custom
head must be explicitly selected with `LIPNET_AVHUBERT_WORD_HEAD`; no personal
head is silently preferred. Missing/incompatible models report unavailable.

## API

| Route | Purpose |
|---|---|
| GET `/health` | Backend and alphabet readiness; availability is not accuracy |
| GET `/api/word/health` | Selected word vocabulary, historical evaluation, actual policy source and policy-file discrepancy |
| GET `/api/recognition/capabilities` | Separate task evidence; sentence recognition unavailable |
| POST `/api/predict` | Alphabet mouth-sequence inference |
| POST `/api/predict/word` | Independently selected legacy or AV-HuBERT word inference |
| POST `/api/word/train-personal` | Validate and save recordings only; HTTP 202, training/evaluation still required |
| GET `/docs` | Generated OpenAPI documentation |

Prediction request: `{"frames": ["base64 JPEG", "..."]}`. Supply 8–76 ordered
frames, each at most 200,000 encoded characters. The AV-HuBERT word path requires
measured eye-axis-aligned grayscale-compatible 96×96 mouth JPEGs. Passing eight
copies of one image is not a meaningful speech clip. Browser quality gating is
required; accepting JPEGs alone does not prove face visibility or speech.

Existing candidate fields are preserved for compatibility. The additive common
schema is:

```text
schema_version: 1
status: success | uncertain
recognition_task: alphabet_recognition | word_recognition
prediction: accepted label or null
confidence: measured model score (uncalibrated)
top_k: [{label, score}, ...]
latency_ms: measured inference/request duration
model_version: checkpoint identifier
vocabulary_id: ordered vocabulary identifier
confidence_calibrated: false
```

The word compatibility field `task` remains `isolated_word`; `predicted_word`
is the candidate even when rejected. Use `prediction`/`word` and `uncertain` for
accepted output. Diagnostic candidates must not be displayed as confirmed words.
AV-HuBERT diagnostics include agreement of three overlapping feature windows;
they are correlated views, not three independent utterances. Alphabet display
also requires consecutive accepted responses. Health and success mean that a
model ran, not that a prediction is correct.

Responses: 422 invalid/missing input; 429 CPU busy (do not queue retries); 503
model unavailable; 504 inference deadline; 500 sanitized internal failure.
The model deadline is 30 seconds. A native CPU job can outlive the HTTP timeout;
its lock remains held and its response is discarded. Retry a fresh capture once
the service is ready. Frontend timeouts/cancellation also discard stale responses.

Recording intake accepts the existing confirmed `WordRecording` structures
(IDs, labels, speaker/session/split, consent, timestamps, hashes, crop contract).
It stores `data/personal_recordings/<bundle_id>.json` using exclusive creation.
`success: false`, `model_activated: false` and `validation_required` explicitly
mean no model was trained. Export real collections and follow the existing
personal cross-session protocol:

```powershell
.\.venv-word-recognition\Scripts\python.exe scripts/train_personal_words.py --help
```

Personal cross-session evaluation is not speaker-independent evaluation. The
existing eight-word workflow requires real train/validation/test recordings,
separate sessions, completeness and paired comparison before explicit activation.
It refuses unsupported expanded vocabularies rather than adding neutral classes.

## Audit and experiments

```powershell
.\.venv-word-recognition\Scripts\python.exe scripts/audit_recognition_dataset.py data/processed/multidataset/miracl_v1.jsonl --decode --out .qa/new-miracl-audit.json
.\.venv-word-recognition\Scripts\python.exe scripts/audit_recognition_dataset.py data/processed/grid_multispeaker_v1/avhubert_letters.jsonl --decode --limit 100 --out .qa/new-grid-sample-audit.json
.\.venv-word-recognition\Scripts\python.exe -m pipeline.word_recognition.temporal_experiment --out runs/new-temporal-experiment --epochs 25
.\.venv-word-recognition\Scripts\python.exe scripts/evaluate_preserved_words.py --out .qa/new-validation-reproduction.json
.\.venv-avhubert-cpu\Scripts\python.exe scripts/benchmark_recognition.py --out .qa/new-cpu-probe.json
```

The benchmark compares the archived service and updated service through the
same ASGI route, sharing one encoder. It requires this task's targeted source
archive. It does not measure browser capture, physical-camera tracking or
network latency. Avoid other CPU workloads during timing. Its alphabet word-ROI
probes measure runtime only. The `--include-historical-test` evaluation option
explicitly reopens the already-used historical test; never use that output to
retune a model or describe it as a fresh test population.

For new alphabet data, use the existing recording/preprocessing tools before
`scripts/train_alphabet.py`. The trainer expects finite 30×140×46 clips in the
original [0,255] storage convention and manifest speaker identities. Full A–Z
coverage is required by default. `--allow-incomplete-vocabulary` is an explicit
partial research exception; unsupported classes remain missing in saved metadata.
Do not compare such a result with complete 26-class recognition.

## Verification and environment

```powershell
.\.venv-word-recognition\Scripts\python.exe -m pytest tests pipeline/eval/tests pipeline/grid/tests pipeline/avhubert/tests
npm --prefix frontend run test:live
npm --prefix frontend run test:math
npm --prefix frontend run build
```

Local word test runtime was completed with FastAPI 0.115.12, PyYAML 6.0.2 and
httpx 0.28.1. Exact resolved versions are recorded in
[word_environment_freeze.txt](.qa/reliability_20261109/word_environment_freeze.txt)
and [encoder_environment_freeze.txt](.qa/reliability_20261109/encoder_environment_freeze.txt).
The encoder environment retains the existing Fairseq compatibility patch;
use `scripts/patch_fairseq_meta.py` to verify it. Do not copy the blocked
scikit-learn DLL from `.venv` into the working test environment.

Browser QA uses installed Chrome, a genuine saved AFTERNOON video replay and
explicit API fixtures. Run a temporary server, set `LIPNET_QA_BASE`, then execute
the saved `.qa/reliability_20261109/browser_qa.mjs`. It is a software integration
test; physical webcam recognition still needs user testing and labelled captures.

## Artifacts and rollback

| Requested deliverable | Location/status |
|---|---|
| Audit, risks and initial plan | `RELIABILITY_AUDIT.md` |
| Architecture, comparisons, limitations | `RELIABILITY_RESULTS.md` |
| Dataset stats and per-record audits | `.qa/reliability_20261109/{miracl,grid}_audit.json` |
| Reproduced frozen baseline, per-class metrics, errors | `.qa/reliability_20261109/final_frozen_evaluation.json` |
| Confusion matrices and class metrics | Same directory, `test_confusion.csv`, `test_per_class.csv` |
| Candidate training config and label vocabulary | `runs/reliability_20261109/temporal_v2/config.json` |
| Candidate best/final weights, history, metrics | `temporal_v2/bilstm_t{15,29}/` |
| Preserved deployable word head | `runs/word_accuracy/avhubert_miracl8_v1/head.pt` |
| New alphabet checkpoint | Not created: complete labelled A–Z data unavailable |
| CPU timings and real API results | `.qa/reliability_20261109/cpu_benchmark_controlled.json` |
| Full test XML/text, browser report | `.qa/reliability_20261109/verified_tests.*`, `live-assistant-browser.json` |
| Artifact preservation | `.qa/reliability_20261109/artifact_integrity_after.json` |
| Initial source + primary checkpoint backup | `.qa/reliability_20261109/source_before.zip` |
| Exact targeted source rollback | `.qa/reliability_20261109/targeted_source_before.zip` |
| Pre-existing Git edits and task-only changes | Same directory: `git_diff_before.patch`, `task_source_changes.patch` |

Review individual archived files before restoring. The repository had extensive
user edits before this task; a bulk Git reset would discard unrelated work.
No commit, deployment, external publication or dataset redistribution was made.
