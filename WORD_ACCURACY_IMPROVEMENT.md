# Measured whole-word accuracy improvement

> Subsequent integration, 2026-10-29: the eight-word model remains frozen and is the default fallback in the combined demo at http://127.0.0.1:8002/demo/live. A separate publisher 500-word research API is now connected; its small diagnostic shows substantial limitations and does not inherit the eight-word score. See [current features, measurements and start commands](EXPANDED_WORD_RECOGNITION.md). The original experiment below is preserved.

Date: **2026-10-29, Asia/Calcutta**. A real frozen AV-HuBERT encoder plus a supervised eight-word head achieved **182/192 correct (94.79%)**, compared with **24/192 (12.50%)** from the preserved scratch model on the **same source clips**: an **82.29 percentage-point increase**. This is an eight-class MIRACL experiment, not a general-English or physical-webcam accuracy result.

Open the separate local research demo at **http://127.0.0.1:8001/demo/live**. It serves the built frontend and real word API; camera capture starts only after the user clicks. The original backend and alphabet selection were not replaced. Restart this separate server, when needed, with `./start_word_demo.ps1` from the repository root. Port 8001 must be free before restarting.

## Scope and results

The supported words are **choose, connection, hello, navigation, next, previous, start, stop**. Inputs are silent mouth video; audio is not used. GRID letter-slot supervision remains separate.

| Measure | Actual result |
| --- | --- |
| Preserved scratch model, same 192 clips | 24/192; 12.50% top-1 |
| New pretrained model, same 192 clips | 182/192; 94.79% top-1 |
| Top-5 | 191/192; 99.48% |
| Macro F1 | 0.94668 |
| Single-word WER | 10/192; 5.21% (substitution rate, not sentence WER) |
| Validation used for selection | 228/240; 95.00% |
| Accepted known test words after rejection | 137/192; 71.35% coverage, 137 correct |
| Phrase out-of-vocabulary diagnostic | 8/60 phrases falsely accepted; 13.33% |

| Word | Correct / total |
| --- | --- |
| choose | 24/24 |
| connection | 23/24 |
| hello | 24/24 |
| navigation | 23/24 |
| next | 22/24 |
| previous | 18/24 |
| start | 24/24 |
| stop | 24/24 |

Held-out speaker results: F10 62/64, F11 57/64, M08 63/64. The weakest class is **previous**, with four errors assigned to **start**. Full precision/recall/F1, counts, predictions and matrix are saved in [evaluation.json](runs/word_accuracy/avhubert_miracl8_v1/evaluation.json), [test_predictions.jsonl](runs/word_accuracy/avhubert_miracl8_v1/test_predictions.jsonl), [confusion.csv](runs/word_accuracy/avhubert_miracl8_v1/confusion.csv), [confusion.png](runs/word_accuracy/avhubert_miracl8_v1/confusion.png), and [interactive confusion matrix](runs/word_accuracy/avhubert_miracl8_v1/confusion.html).

The accepted-word result does **not** establish perfect recognition: rejection removes 55 known test clips, and phrase false accepts remain. Unseen isolated English words, accent/camera variation, continuous sentences and live webcam accuracy remain unverified. Softmax confidence is not a calibrated probability of correctness.

## Dataset and selection protocol

The author-provided MIRACL archive and original RGB frames were reused. Full source manifest: `data/processed/multidataset/miracl_v1.jsonl`, SHA-256 `4c6150dcd58736b9620807c9d00df2ff94b410dda3413ec84b614b8053152352`. Word labels with unresolved author naming conflicts remain excluded rather than silently renamed.

- Training: F01, F02, F04, F05, F06, F07, M01, M02, M04; 720 planned word utterances, **704 usable**. Sixteen measured tracking failures were recorded and excluded from training.
- Validation: F08, F09, M07; **240/240 usable** word utterances.
- Test: F10, F11, M08; repetitions **03–10**, **192/192 usable** word utterances, 24 per class.
- The 48 test utterances from repetitions 01–02 used by the earlier pilot were explicitly quarantined. The test speakers themselves had been observed in that pilot; this is a fresh-utterance test on the original held-out speakers, not an untouched new population.
- Separate phrase diagnostics: 60 validation and 60 test phrases. They never become training word labels. Phrase rejection is not a substitute for testing unseen isolated words.

The frozen encoder produced layer 6, 9 and 12 representations in one forward pass per clip. Exactly 27 layer/pooling/regularization candidates were selected using word validation results only. The winning head and rejection policy were frozen before the test evaluation. Test results did not select a candidate. See [protocol.json](runs/word_accuracy/avhubert_miracl8_v1/protocol.json), [selection.json](runs/word_accuracy/avhubert_miracl8_v1/selection.json) and [rejection_policy.json](runs/word_accuracy/avhubert_miracl8_v1/rejection_policy.json).

The paired comparison ran the original scratch model with its actual 24-frame, 64-pixel preprocessing on the same 192 source RGB utterances; it did not approximate that input by resizing the new feature cache. Independent workspace work changed the shared scratch definition, so its exact original definition was preserved as `pipeline/word_recognition/legacy_word_model.py`, matching the original recorded source SHA-256 `7d517dbd6d97bf2885749cacf73b32461340c2ba9339c1b9b0abc11855648ec9`. The original checkpoint was not overwritten. See [paired_comparison.json](runs/word_accuracy/avhubert_miracl8_v1/paired_comparison.json).

## Architecture and CPU execution

The existing [official AV-HuBERT](https://github.com/facebookresearch/av_hubert) Base checkpoint is frozen. Measured MediaPipe landmarks align the eye axis and mouth center; the mouth is sampled at 96 × 96 grayscale, interpolated to 29 frames, center-cropped to 88 × 88, scaled to [0,1], then normalized by mean 0.421/std 0.165. MIRACL native frame rate is unknown and is recorded as unknown; fixed-length interpolation does not invent a native FPS.

The selected layer is **9**. Temporal mean and standard deviation concatenate into 1,536 features, followed by a training-only StandardScaler and eight-way logistic classifier, C=1. The classifier has **12,296 learned coefficients/biases**; the encoder is not fine-tuned. The artifact contains the vocabulary, scaler, layer, preprocessing and encoder fingerprints.

The i3-1115G4 has 7.74 GiB RAM and no CUDA. Memory-mapped checkpoint loading plus meta-device construction and strict tensor assignment avoid allocating a second encoder or loading optimizer tensors unnecessarily. A reproducible Fairseq compatibility patch skips random initialization only for meta tensors. The normal loader remains the default for other callers.

Measured stages, one CPU thread: tracking/preparation **626.28 s**, feature extraction **856.29 s** for 1,256 usable word/phrase clips, and head selection/training **19.29 s**. Each cache item holds finite `[3,29,768]` features with source/preprocessing/encoder integrity receipts. These are measured local runtimes, not latency guarantees.

Important fingerprints:

| Artifact | SHA-256 |
| --- | --- |
| New head `runs/word_accuracy/avhubert_miracl8_v1/head.pt` | `bc07a9eca0979e556af573a387733396a891fe41faa10715c9f4d323e9c4cc74` |
| Frozen AV-HuBERT encoder | `dee8b0651dbea3a21cda0f2c52b956e3a878d5f6ee503d427d6a4044bb95c9cc` |
| Feature manifest | `b264901023a5ec9fef6211c88e57bbb66f86c3e409a78d835de375267b10c6c8` |
| Preserved scratch checkpoint | `3c7c420d30ac111d4b752322318d92c3b9dfa2077ba2f3d04bc6a71b6aa937ed` |

Use the publisher's research license. Neither successful encoding nor these eight-class results grant commercial deployment rights.

## Live research integration

`app/services/pretrained_word_inference.py` is selected only with `LIPNET_WORD_BACKEND=avhubert`; `LIPNET_AVHUBERT_WORD_HEAD` selects this word head. This is separate from `LIPNET_MODEL_BACKEND` and the alphabet endpoint. The eight-word head must not be passed to the 26-class alphabet adapter.

The word service checks fingerprints, dimensions and evaluation evidence before reporting ready, performs a discarded warmup for CPU initialization, and permits one inference request at a time. Wrong spatial preprocessing returns HTTP 422; concurrent inference returns 429. Rejection uses validation-selected confidence 0.95/margin 0.20. The frontend requires two agreeing accepted windows within the current utterance; stop, face loss and mode changes clear confirmation. This reduces flicker but has no independently measured live accuracy gain.

A genuine RGB clip labeled **connection** was tracked, encoded as JPEG mouth ROIs, sent through the actual FastAPI word route, and correctly accepted at confidence 0.99838. The selected-layer runtime exactly matched cached hook features (maximum absolute error 0). This single API execution is distinct from the 192-clip evaluation. Evidence: `.qa/word-improvement-real-api.json`.

The separate private server on port 8001 serves `/demo/live`, `/api/word/health`, and `/api/predict/word`. No previous server was stopped. Camera use requires the user's button click; no physical webcam recognition benchmark was performed.

## Genuine 500-word publisher model: separate research path

The [author's LRW model repository](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks), commit `0cc99fab046bd959578f2baac52e654746da4825`, supplies a genuine video-only **ResNet18 + DC-TCN**, without word-boundary inputs. Its publicly linked `resnet18_dctcn_video.pth` was downloaded with a recorded acquisition receipt: **210,597,729 bytes**, SHA-256 `806ea23bc082a9c8a3c08aafbc1cab0f0f8aaeef70a4c824989ebd17bf245b5f`. No publisher checksum was available for comparison. Ordered author labels match all 500 entries of `data/vocab.txt`.

The isolated `pipeline/lrw_tcn` module strictly loads all weights and produces 500 logits. On the existing genuine public Oxford **AFTERNOON** clip it predicted **AFTERNOON**, confidence 0.95890, inference 3.09 s. This is **one smoke test**, not local LRW accuracy. The fixed crop used for this test does not reproduce the publisher's full mean-face alignment. The [author license](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/blob/master/LICENSE) limits use to research/private study/comparative prototypes; no production or commercial use is asserted.

This model remains a separate research CLI. It is not the eight-word live model and does not inherit its 94.79% score. A complete authorized LRW dataset, publisher-equivalent alignment and a real held-out 500-class evaluation remain missing. Downloading public author weights does not grant LRW corpus access or bypass the BBC agreement.

## Commands executed and repeatable entry points

From the repository root, the actual model stages were:

```powershell
.\.venv-word-recognition\Scripts\python.exe -m pipeline.word_recognition.pretrained init --run runs/word_accuracy/avhubert_miracl8_v1
.\.venv-word-recognition\Scripts\python.exe -u scripts/improve_word_accuracy.py --run runs/word_accuracy/avhubert_miracl8_v1 --through train
.\.venv-word-recognition\Scripts\python.exe -u scripts/improve_word_accuracy.py --run runs/word_accuracy/avhubert_miracl8_v1 --through evaluate
.\.venv-word-recognition\Scripts\python.exe scripts/compare_word_accuracy.py --run runs/word_accuracy/avhubert_miracl8_v1
.\.venv-word-recognition\Scripts\python.exe -m pipeline.lrw_tcn.inference --lrw-centered-video .qa/lrw-public-example/AFTERNOON.mp4
```

The comparison itself was executed by `.qa/word_improvement_baseline_driver.py`; the native command and paired output are recorded in `.qa/word-improvement-paired-baseline.txt` and the run's `paired_comparison.json`. The comparison refuses to overwrite its completed result. Stage commands, environment, timestamps and zero exit codes are preserved in `runs/word_accuracy/logs/*-commands.json`. Completed outputs are reused by the orchestrator; keep this frozen evaluation rather than repeatedly tuning against its test partition.

To use the completed model:

```powershell
.\start_word_demo.ps1
```

To run another dataset/experiment, choose a new run directory and establish a fresh evaluation protocol first. See [pretrained workflow](pipeline/word_recognition/PRETRAINED.md) and [publisher LRW integration](pipeline/lrw_tcn/README.md).

## Validation and preservation

- Full Python suite: **482 passed, 4 skipped** in 199.37 s; `.qa/word-improvement-full-tests.txt`. Skips concern optional fixtures/dependencies, not successful recognition evidence.
- Frontend: production TypeScript/Vite build passed; **14 live logic tests + 26 mathematics tests passed**. Logs: `.qa/word-improvement-{frontend-build,live-tests,math-tests}.txt`.
- Actual port-8001 browser: **8 checks passed** covering model readiness, word selection, visible measured results/vocabulary, mode buttons, camera remaining off until clicked, mobile overflow and absence of uncaught exceptions. No mocked model responses; `.qa/word-improvement-browser-check.json` and `.qa/word-improvement-live-demo.png`.
- Real API smoke: accepted the labeled connection clip, verified 422/429 behavior and selected-layer equivalence; `.qa/word-improvement-real-api.json`.
- Real 500-class publisher-model smoke: AFTERNOON correctly predicted; `.qa/word-improvement-lrw-pretrained.json`. No local 500-class evaluation exists.
- Original protected checkpoints and GRID inputs remain fingerprinted; final verification is saved in `.qa/word-improvement-final-integrity.json`. Other independent workspace work was preserved; its changing experimental checkpoints are not claimed as changes made by this task.

Main inspected/implemented files: `pipeline/avhubert/{loader,low_memory}.py`, vendored Fairseq sentence encoder initializer, `scripts/patch_fairseq_meta.py`, `pipeline/word_recognition/{pretrained,pretrained_inference,legacy_word_model}.py`, `scripts/{improve_word_accuracy,compare_word_accuracy,download_lrw_tcn}.py`, `pipeline/lrw_tcn/`, `app/main.py`, `app/services/{pretrained_word_inference,recognition_capabilities}.py`, `frontend/src/{lib/api,hooks/useBackend,live-demo/useDemoPipeline,live-demo/wordConsensus}.ts*`, workspace LiveDemo/RecognitionEvidence, scoped live CSS, and the new pretrained-word/LRW/service tests. Landing colors and unrelated frontend work were preserved.
