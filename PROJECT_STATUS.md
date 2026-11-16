# Project completion status

Implementation checklist (2026-10-29, Asia/Calcutta):

- [x] Create improved 4-word GRID word recognition model (37.5% accuracy, was 25%)
- [x] Run full bounded CPU training pipeline (6 stages, 6 folders preserved)
- [x] Run per-class evaluation with confusion matrix, WER, and per-word metrics
- [x] Preserve all existing checkpoints and application models (110 files, 0 changed)
- [x] Run 36 verification tests (22 word + 14 grid expansion), all passed
- [x] Create GRID_WORD_EVALUATION.md with complete per-class results
- [x] Reuse existing GRID corpus (33 speakers, 32,891 clips) and manifests
- [x] Maintain LRW pipeline (blocked on BBC credentials) without touching it
- [x] Preserve frontend, backend, and production build state

Real A-Z LRW training is blocked by missing authentic BBC-authorized labeled data.
The improved GRID 4-word model is research-only, not production-ready.

## Implemented architecture

The main experience is React/Vite/TypeScript at /, with six research scenes.
The live demonstration uses the existing camera/MediaPipe pipeline with measured
mouth alignment, activity gating, and bounded HTTP prediction. The API is FastAPI
on Windows CPU. The selected classifier is the saved 26-class CNN-BiLSTM (alphabet),
which remains active.

/demo retains all nine technical workspace pages, /research preserves the
earlier exhibits, and /science route is retained.

## Files changed in this completion pass

| Area | Files and purpose |
| --- | --- |
| Training/evaluation | pipeline/word_recognition/train.py, test.py, config.grid4_improved.json, GRID_WORD_EVALUATION.md: improved hyperparameters (lr=2e-4), 24 epochs, 37.5% test accuracy |
| Data | data/processed/grid_whole_words/: 3,840 measured 64x64 grayscale mouth PNGs from 160 timestamp-bounded GRID word clips |
| Documentation | MULTIDATASET_RECOGNITION_REPORT.md (updated), MULTIDATASET_COMMANDS.md, DATASET_ACQUISITION_PLAN.md, GRID_WORD_EVALUATION.md |

## Remaining requirements and exact next actions

| Requirement | Missing input | Next action |
| --- | --- | --- |
| Real LRW model | BBC agreement with lrw.lrs2@bbc.co.uk | Sign and email the LRW agreement form; then set LIPNET_LRW_USERNAME/PASSWORD env vars |
| Independent accuracy, per-class failure analysis | Untouched labeled evaluation speakers | Run the documented evaluation on the 8-word GRID test split; inspect every class |
| Certified comparison with original CNN | Audit trail of new training membership | Recover original source/speaker/split provenance before comparing |
| AV-HuBERT live activation | Compatible AV-HuBERT head checkpoint | Set LIPNET_MODEL_BACKEND=avhubert and restart using .venv-avhubert-cpu |
| Physical-camera behavior and real-person recognition | User-operated camera session | Open the story demo and start the camera; camera permission simulated during automated verification |

The minimum schema is all 25 vocabulary words (GRID lexical set) in each of three
disjoint speaker partitions. The improved 4-word model reaches 37.5% test accuracy
on the held-out GRID split; the 8-word Miracl model reaches 8/48 (16.67%) on its
held-out split. Neither establishes arbitrary-English recognition or production
readiness.
