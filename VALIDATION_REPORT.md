# Completion validation report

Executed locally on Windows CPU on 2026-10-27 (Asia/Calcutta). Node 24, npm 11, Python 3.11.9, and the existing `.venv` / `.venv-avhubert-cpu` environments were used. No GPU, WSL, Linux, Docker, checkpoint redownload, or real training run was required.

## Regression and build

| Check | Before editing | Final result |
| --- | --- | --- |
| Python suite: `tests` plus AV-HuBERT/GRID/evaluation tests | 413 passed, 3 skipped, 46.39 seconds | 424 passed, 3 skipped, 27.82 seconds |
| Live stabilization / alignment / socket lifecycle | Existing checks retained | 10 passed |
| TypeScript and Vite | Passed | Passed, 122 modules transformed |
| Browser | Prior report retained | 66 checks passed in the navigation/motion follow-up; no uncaught exceptions or failed resource responses |

The three skipped tests require an absent `synthetic_dataset` fixture directory; they are not failed tests or evidence of hardware testing. No additional lint configuration exists in the frontend; `npm run build` runs the available TypeScript project check before bundling.

New tests cover service/model readiness separation, missing AV-HuBERT head, safe unavailable errors, CORS and invalid inputs, legacy WebSocket capacity/cleanup and bounded buffering, authentic-source declarations and speaker/content leakage, exact feature shapes/fingerprints, per-class metric arithmetic, annotation preparation, crop rejection, and intentional socket disconnects. Temporary generated fixtures verify software behavior only.

The stricter cache checks initially caught a temporal-length mismatch in the old unit fixture; it was corrected to its declared 29-frame format. A TypeScript numeric/string mismatch in the separately added science component was fixed without replacing that work. The HTTP harness was corrected to accept CORS's plain-text preflight response. Required missing test/runtime packages `httpx==0.28.1` and `websockets==15.0.1` were installed; the latter was also added to the AV-HuBERT CPU environment. No broad package upgrade was performed.

## Actual backend requests

`scripts/verify_local.py --http` ran against the real Uvicorn service at `http://127.0.0.1:8000` and wrote `.qa/completion-http.json`:

- `GET /health`: 200, service ready, selected `alphabet` backend, CPU CNN-BiLSTM inference ready.
- `GET /api/model/availability`: 200; encoder file present, supervised AV-HuBERT head absent/unready, sentence recognition unavailable, WebSocket classification unavailable.
- `GET /api/evidence`: 200, current corpus counts, independent benchmark accuracy `null`.
- Local-origin CORS preflight: 200 with the correct allowed origin.
- Unknown API route: 404 JSON, not SPA HTML.
- Empty sequence and malformed JPEG sequence: 422.
- Real saved-checkpoint execution from an eight-JPEG synthetic transport fixture: 200, genuine full 26-class softmax summing to one, three ranked candidates, received frames 8, model frames 30.
- Measured transport round trip: 181.68 ms; backend-reported model execution: 176.12 ms for that single fixture. This is not a benchmark or a real-person measurement.
- Four concurrent real HTTP prediction requests returned `[429, 429, 200, 429]`, verifying CPU backpressure without concurrent forward passes.
- Actual WebSocket handshake reported lifecycle-only/unready classification; ping/pong worked, a JSON array was rejected, a frame received explicit `model_unavailable`, and disconnect completed cleanly. ASGI tests also exercised the 30-frame buffer bound and subsequent reconnection after cleanup.
- Missing AV-HuBERT head was separately tested through real ASGI routes: health remains 200 with inference unready; prediction is 503 with no fabricated label or private path.

Synthetic crop predictions demonstrate transport and real model execution, not speech understanding. The live product gates requests on measured mouth movement and has never been validated for real-person accuracy in this task.

## Browser and visual verification

The installed Chrome browser was controlled through its DevTools protocol using `node scripts/frontend-qa.mjs`. Screenshots were actually inspected, including desktop/mobile landing pages and the live scene; this verification is not inferred from a successful build or HTTP response.

The `.qa/browser-report.json` now records 66 passed checks after the navigation/motion follow-up (the original completion run passed 52). No uncaught exceptions or failed resource responses were reported. Tests cover:

- All nine workspace routes at 1366 × 768, 900 × 1024, and 390 × 844 without horizontal overflow, unwanted camera prompts, or application errors.
- Six story scenes, complete anchor targets, preserved charcoal/ivory/mint palette, no main-story sidebar, collapsed diagnostics, and no placeholder predictions.
- Desktop/mobile screenshots of the idea, method, live demo, evidence, and journey.
- Story and workspace camera denial, simulated stream start, stop, cancelled late permission response, disconnect, and route-unmount cleanup.
- Offline backend usability, missing endpoints, 503, invalid JSON, timeout, reduced-motion behavior, and no invented prediction.
- Preserved `/research` and `/science` routes.
- The compiled production bundle served at `http://127.0.0.1:8000/`, with same-origin API readiness and no camera access on load.
- Direct header links to Science and the full Live Demo, all 12 routes in the All pages directory at desktop/mobile widths, opaque menu/pointer behavior, Escape/focus cleanup, and Science/research links in the workspace sidebar.
- Reused original landmark-face particles changing over time, actual pause behavior with a stable canvas, and a reduced-motion fallback that stops the face animation and reveals all content.

The navigation follow-up also rechecked the production build (124 modules) and all 10 frontend tests. Page changes reset to the beginning of the destination, so Science does not inherit the story's scroll position. Same-page anchors retain smooth scrolling; direct scene anchors are resolved after lazy content mounts.

Physical camera permission and real-person recognition were not tested. The simulated streams are explicit lifecycle fixtures. Browser artifacts and source snapshots live in ignored `.qa`; they are not training or evaluation data.

## Pretrained encoder recheck

The official checkpoint was loaded again in `.venv-avhubert-cpu` and encoded the authentic GRID video `data/external/grid/s1/corpus/T01/video/s1/bbaf2n.mpg`:

| Result | Value |
| --- | --- |
| Output features | `[29,768]`, all finite |
| Device | CPU |
| Frame/crop preprocessing | 29 interpolated frames; grayscale 88 × 88; mean 0.421; std 0.165 |
| Extraction after model load | 11.14 seconds in this verification run, while other checks were active |
| Prior recorded measurement | 4.75 seconds for the same clip after model load |

Timing includes decoding, landmark tracking/alignment, and feature extraction after loading. These are individual observations under different system load, not real-time guarantees. The encoder supplies representations, not alphabet labels. Detailed artifact: `.qa/completion-encoder.json`.

## Data and recognition evidence

| Available source | Count | Speaker/provenance limitation |
| --- | --- | --- |
| Registered authentic alphabet recordings | 0 | All 26 labels are missing from the local collection |
| GRID utterance manifest | 1,000 | One speaker, sentence-level speech, no isolated A–Z annotations |
| Local LRW clips | 45 (15 train, 15 valid, 15 test) | Generated development fixtures; excluded from real-data evidence |
| Trained AV-HuBERT alphabet head | 0 | Cannot activate pretrained classification |

No new eligible baseline evaluation, per-class recognition benchmark, confusion matrix from real data, head accuracy, or accuracy gain was produced. The saved historical `models/training_summary.json` contains 12% test accuracy over 100 samples (800 training/100 validation/100 test), with 20% best validation accuracy, but its original data membership and speaker/source provenance are unavailable. It is not an independently verified baseline, and the story does not promote it as one.

The completed pipeline writes precision/recall/F1/support, macro F1, accuracy, confusion matrix CSV, model/encoder fingerprints, and measured cached-head evaluation time when authentic labeled data is supplied. Test data does not select checkpoints. Training speakers/content hashes are recorded so later evaluation can reject overlap. Threshold calibration and a same-sample baseline comparison remain dependent on eligible data and audited training membership.

## Protected checkpoint hashes

Both actual files were hashed before editing and again by the final CPU/network smoke scripts. Before and after values are identical:

| File | SHA-256 before = after |
| --- | --- |
| `models/alphabet_classifier.pt` | `7c00d5117b7fae9a76c05980ca70edd680e79be4c8a152ea2595c98ded90c4c3` |
| `models/external/avhubert/base_vox_iter5.pt` | `dee8b0651dbea3a21cda0f2c52b956e3a878d5f6ee503d427d6a4044bb95c9cc` |

The official encoder remains 1,240,185,329 bytes. The AV-HuBERT digest is locally computed; no publisher checksum comparison is claimed.

## Reproduction commands

See [README.md](README.md#windows-cpu-setup) for backend/frontend launch and [ACCURACY_PIPELINE.md](ACCURACY_PIPELINE.md) for data preparation and training/evaluation. From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests pipeline/avhubert/tests pipeline/grid/tests pipeline/eval/tests -q --junitxml=.qa/completion-final-tests.xml
```

```powershell
.\.venv\Scripts\python.exe scripts/verify_local.py --http --output .qa/completion-http.json
```

```powershell
.\.venv-avhubert-cpu\Scripts\python.exe scripts/verify_local.py --encoder-smoke --output .qa/completion-encoder.json
```

```powershell
node scripts/frontend-qa.mjs
```

Run `npm run test:live` and `npm run build` individually in `frontend`. Existing local Vite and backend servers remain available; no external publication or deployment was performed. [PROJECT_STATUS.md](PROJECT_STATUS.md) lists files changed and exact remaining data-dependent actions.
