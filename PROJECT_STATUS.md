# Project completion status

Implementation checklist (2026-10-27, Asia/Calcutta):

- [x] Preserve user changes and record checkpoint hashes; baseline: 413 tests passed, 3 skipped, production build passed.
- [x] Inspect active model, routes, training pipeline, and available data.
- [x] Replace the default presentation with six connected story scenes; retain technical tools.
- [x] Separate backend health, classifier readiness, encoder presence, head readiness, and sentence availability.
- [x] Repair connection lifecycle defects and verify actual HTTP/WebSocket contracts.
- [x] Harden feature validation and prepare reproducible per-class evaluation outputs.
- [x] Run final regression, CPU, HTTP, browser, and production checks.
- [x] Update setup, model availability, and validation documentation.

Real A–Z training/evaluation is blocked by missing authentic labeled data with disjoint train/validation/test speakers. The single-speaker GRID sentences and generated LRW fixtures do not meet this requirement. The original alphabet CNN-BiLSTM remains active.

## Implemented architecture

The main experience is React/Vite/TypeScript at `/`, with six scenes from the question to the research journey. Its live demonstration uses the existing camera/MediaPipe pipeline, measured eye-axis mouth alignment, activity gating, sliding windows, real HTTP prediction, EMA, and two-window confirmation. Diagnostics are collapsed by default. The charcoal `#0a0a0c`, ivory `#edebe4`, and mint `#8fdfcb` palette is preserved.

Navigation and animation follow-up: the header now directly links Live Demo and Science Lab, with an All pages directory for all 12 existing routes. Four visible destination links and a visible full-studio link supplement the inline demo. The workspace sidebar also links Science and the original research exhibits. The main hero reuses the earlier animated landmark face and particles, with GSAP entrance/parallax, staggered frames, processing-line motion, and journey reveals. Animation pause, offscreen/tab cleanup, and reduced-motion fallbacks remain available.

The backend remains FastAPI on Windows CPU. Its selected classifier is the saved 26-class CNN-BiLSTM, receiving grayscale `[B,30,1,140,46]`. AV-HuBERT is a verified frozen feature encoder with a separate supervised-head workflow; it is not an active classifier. Sentence recognition and WebSocket classification remain explicitly unavailable. The optional WebSocket transport is lifecycle-only; the product uses bounded HTTP windows.

`/demo` retains all nine technical workspace pages, `/research` preserves the earlier detailed exhibits, and the separately developed `/science` route is retained. The new main story is a distinct visitor experience, not a recolored workspace.

## Files changed in this completion pass

| Area | Files and purpose |
| --- | --- |
| Story presentation | `frontend/src/story/StoryExperience.tsx`, `styles/story.css`, `App.tsx`, `ResearchApp.tsx`, `router.tsx`, `frontend/index.html`: six-scene story, simple real demo, disclosed diagnostics, evidence from API, preserved routes and accurate metadata |
| Backend contract | `app/main.py`, `app/services/browser_inference.py`, `project_evidence.py`: explicit readiness layers, CORS configuration, real corpus counts, safe model errors, truthful legacy WebSocket state |
| Connection cleanup | `frontend/src/lib/inferenceSocket.ts`, `hooks/useWebSocketInference.ts`, `lib/api.ts`, `hooks/useBackend.tsx`: bounded retries, correct socket URLs, no reconnect after cleanup, readiness fields |
| Mouth quality | `live-demo/mouthUtils.ts`, `useDemoPipeline.ts`: reject invalid/off-image crops and reset stale buffers when the mouth crop is lost |
| Training/evaluation | `pipeline/avhubert/classification.py`, `scripts/avhubert_classifier.py`, `pipeline/eval/classification.py`: frame/dimension/fingerprint checks, training-membership metadata, complete A–Z validation, held-out per-class reports and confusion matrix |
| Data handoff | `scripts/prepare_alphabet_annotations.py`: explicit speaker assignment and safe real-recording manifest preparation |
| Verification | `scripts/verify_local.py`, `scripts/frontend-qa.mjs`, Python contract/annotation/accuracy tests and Node socket/live tests: actual network, CPU, browser, and regression checks |
| Dependencies | `requirements.txt`, `pyproject.toml`, `frontend/package.json`: add pinned HTTP test client and WebSocket runtime; expose extended Node checks; no broad dependency upgrades |
| Compatibility | One numeric comparison in the concurrently added `science/components/MathematicsScene.tsx` fixed a TypeScript error; other science work was preserved |
| Documentation | `README.md`, this status file, `VALIDATION_REPORT.md`, `MODEL_AVAILABILITY.md`, `ACCURACY_PIPELINE.md`, and `FRONTEND_REDESIGN.md`: current setup, scope, data requirements, protocols, results, and limitations |

No datasets, trained model weights, or existing user changes were reset, deleted, or replaced. Working assets were reused. The completion pass did not redownload either checkpoint or run real training. Before-edit source snapshots are in ignored `.qa/completion-before`.

## Remaining requirements and exact next actions

| Requirement | Missing input | Next action |
| --- | --- | --- |
| Real A–Z supervised head | Authentic correctly labeled A–Z clips from separate train/validation/test speakers | Collect to `alphabet_dataset/recordings`; use the preparation, validation, extraction, and training commands in `ACCURACY_PIPELINE.md` |
| Independent accuracy, per-class failure analysis, threshold calibration | Untouched labeled evaluation speakers and a validation set | Select settings on validation, then run the documented `evaluate` command on test data; inspect every class, not accuracy alone |
| Certified comparison with original CNN | Auditable CNN training membership plus the same eligible test samples | Recover original source/speaker/split provenance or train an auditable baseline on the new training split; do not certify the historical summary as independent |
| AV-HuBERT live activation | Compatible complete A–Z `head.pt` | Set `LIPNET_MODEL_BACKEND=avhubert` and `LIPNET_AVHUBERT_HEAD`, then restart using `.venv-avhubert-cpu`; inspect `/health` |
| Physical-camera behavior and real-person recognition | User-operated camera session and labeled utterances | Open the story demo and start the camera explicitly; camera permission was simulated during automated verification |

The minimum schema is all 26 letters in each of three disjoint speaker partitions (78 clips). This is not a sufficient scientific sample-size recommendation. Collect repeated utterances and more speakers before making generalization claims. Current recorded inventory is 0 registered alphabet recordings, 1,000 GRID utterances from 1 speaker, and 45 LRW development fixtures.

## Launch and reproduce

[README.md](README.md#windows-cpu-setup) is the Windows CPU setup guide. [MODEL_AVAILABILITY.md](MODEL_AVAILABILITY.md) defines HTTP/WebSocket contracts and activation. [ACCURACY_PIPELINE.md](ACCURACY_PIPELINE.md) provides recording, preparation, annotation validation, feature extraction, head training, and evaluation commands. [VALIDATION_REPORT.md](VALIDATION_REPORT.md) records exact executed checks, limitations, and both checkpoint hashes before/after.
