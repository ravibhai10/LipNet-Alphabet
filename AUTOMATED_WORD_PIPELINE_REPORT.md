# Automated whole-word pipeline: execution and diagnosis

**2026-10-29, Asia/Calcutta.** The modular CPU workflow was implemented, provisioned, and executed against genuine annotation-bounded GRID word segments. **The resulting research model failed recognition-quality checks: 8/32 correct (25%), with every prediction equal to “place.”** It is not ready for production and was not activated in the application. No improvement over existing recognition models is established.

Implementation and runnable commands: [pipeline/word_recognition/README.md](pipeline/word_recognition/README.md).

## What was actually run

The task recognizes four complete spoken English command words: **bin, lay, place, set**. It uses the first annotated lexical word in actual GRID sentences; no GRID letter-slot labels or alphabet clips are used. Original word timing, not filenames or predictions, supplies the supervision and segmentation.

| Partition | Samples per word | Total | Speakers |
| --- | ---: | ---: | --- |
| Training | 24 | 96 | 23: s1–s20, s22–s24 |
| Validation | 8 | 32 | 5: s25–s29 |
| Test | 8 | 32 | 5: s30–s34 |

All 160 source videos and annotations were rehashed against the previously verified inventory. Source utterances/speakers do not cross partitions. All four classes occur in every partition, each class covers every available speaker in that partition, and no candidate was excluded during this bounded preparation. The pipeline generated **3,840 measured, grayscale 64 × 64 mouth PNGs** at 24 interpolated frames per word. No synthetic images, inferred labels, or fallback zero tensors entered this run.

The corpus was already present. All **66 existing official video/alignment archive SHA-256 values and lengths** matched the acquisition records. No new restricted dataset acquisition was attempted. The [official GRID publisher](https://spandh.dcs.shef.ac.uk/gridcorpus/) supplies the research-use corpus and word alignments; the existing downloader uses links parsed from that publisher. Recorded archive SHA-256 values are local fingerprints, not checksums published by GRID. The existing downloader's ZIP/TAR integrity, HTTP resume, and safe extraction behavior was reused.

This is **fixed-vocabulary recognition of pre-segmented, coarticulated words**. It does not implement automatic word boundaries, unrestricted English, sentence decoding, or an LRW-500 recognizer. The earlier LRW access/data blockers remain; see [LRW_WORD_PIPELINE.md](LRW_WORD_PIPELINE.md).

## Environment and model

Provisioned a dedicated `.venv-word-recognition` using Python 3.11.9, without selecting a different application backend or installing into the existing application/AV-HuBERT environments. Verified core versions: torch **2.2.2+cpu**, torchvision **0.17.2+cpu**, torchaudio **2.2.2+cpu**, MediaPipe **0.10.14**, OpenCV contrib **4.10.0**, NumPy **1.26.4**, Pandas **2.2.3**, tqdm **4.67.1**, scikit-learn **1.5.2**, and imageio-ffmpeg **0.6.0**. `pip check` passed. The bundled native **FFmpeg 7.1** binary executed successfully; the system PATH was not changed. Setup log: `.qa/word-recognition-setup.txt`.

The new **83,660-parameter** network has three Conv3D blocks with 8/16/32 channels, GroupNorm/SiLU, spatial pooling, a **two-layer bidirectional GRU with hidden size 48 per direction**, temporal mean pooling, and a four-class linear head. It was trained from scratch. Existing alphabet, LRW-sized, and AV-HuBERT weights were not reinterpreted or overwritten.

Real micro-batch geometry:

```text
input            [2, 1, 24, 64, 64]
Conv3D block 1   [2, 8, 24, 32, 32]
Conv3D block 2   [2, 16, 24, 16, 16]
Conv3D block 3   [2, 32, 24, 8, 8]
frame features   [2, 24, 32]
BiGRU            [2, 24, 96]
logits           [2, 4]
```

The smoke matrix used two genuinely tracked words, checked finite logits/loss, and found finite nonzero gradients for every trainable parameter. Total gradient L2 norm was **3.1785**. A complete two-clip optimization step took **1.2646 seconds** in that check; it is a single cold/runtime measurement, not a throughput guarantee. Full details and per-parameter norms are in the run's `smoke.json`.

## Training and actual held-out results

Configuration: seed 42, CPU/two threads, batch 4, AdamW learning rate 0.001, weight decay 0.0001, label smoothing 0.1, clipping at 5, ReduceLROnPlateau, and early-stopping patience 4. Only training clips received coherent spatial/brightness jitter, mirroring, and simulated frame loss. Validation/test preprocessing was deterministic.

Training requested at most 12 epochs and **early-stopped after epoch 6**. Validation loss selected **epoch 2**, loss **1.3865673**. Every recorded validation accuracy was 25%. Learning rate reduced to 0.0005; recorded pre-clipping gradient norms remained finite, approximately **0.337–2.911**. The optimization loop took **65.54 seconds**, including validation; whole train-stage time, including imports/data checks, was **132.54 seconds**.

The held-out test contains **32 genuine words from five test speakers**:

| Word | Correct / total | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| bin | 0 / 8 | 0.00 | 0.00 | 0.00 |
| lay | 0 / 8 | 0.00 | 0.00 | 0.00 |
| place | 8 / 8 | 0.25 | 1.00 | 0.40 |
| set | 0 / 8 | 0.00 | 0.00 | 0.00 |

**Word accuracy: 25% (8/32). Segmented-word WER: 0.75 (24 substitutions / 32 references). Macro F1: 0.10.** A one-word classifier always emits one word for one segmented reference, so insertion/deletion counts are structurally zero and WER equals 1 − accuracy. This is not continuous-speech/sentence WER. Top-k at k=4 necessarily includes the entire vocabulary and is uninformative; it is explicitly labeled that way in the metrics.

Every test prediction was **place**, so performance equals the balanced majority-class baseline. Finite gradients and successful checkpoint loading did not establish useful word learning. The artifacts do not establish the cause: dataset scale, crop content, initialization, architecture, augmentation, and optimization require investigation on training/validation data. No post-test hyperparameter search or replacement evaluation was performed to obtain a better-looking score.

`diagnostics.json` records `recognition_quality_failed`, `deployment_ready: false`, single-class collapse, majority-baseline failure, and zero recall for the other observed classes. The diagnostic stage is now included in future wrapper runs. For the already completed initial five-stage run, this newly added diagnostic module was explicitly executed afterward and its command/result were retained. Computational completion is separate from model-quality approval; no model deployment is triggered.

## Inspectable artifacts

Run directory: `runs/word_recognition/grid_commands_cpu_20261029`.

- Selected checkpoint: `best.pt`, SHA-256 **`86735179f16fc8044b23232433810b33a1ab231996d559d0dcc6f5d81c42e5fa`**.
- Resume checkpoint: `last.pt`, including optimizer/scheduler and loader/PyTorch RNG state at the last complete epoch.
- Frozen word manifest: `words.jsonl`, SHA-256 **`00db5d7ed880441a76c6bdfeaf148ec4e2ddb80db785e4ada6a67807e17b572e`**.
- Configuration fingerprint: **`80ff4af3decb1bddbb2af01c9e155042de03fa1670b4de2d7fd1ae2d47a7581a`**.
- Canonical PNG root: `data/processed/grid_whole_words/80ff4af3decb1bdd/<speaker>/<word>/<sample>/`.
- Source/data proof: `acquisition.json`, `dataset.json`, `excluded_candidates.jsonl` (empty), `resources.json`, and `smoke.json`.
- Training/quality proof: `training_history.json`, `training_summary.json`, `diagnostics.json`, and checkpoint metadata.
- Evaluation folder: `evaluation/86735179f16fc804/`, containing `metrics.json`, `per_word.csv`, `predictions.jsonl`, `confusion_matrix.csv`, and **offline interactive `confusion_matrix.html`**.
- Matrix screenshot: `.qa/word-confusion-matrix.png`; browser interaction proof: `.qa/word-matrix-browser-check.json`.
- Execution/preservation: `commands.json`, individual stage logs, `protected_before.json`, `integrity.json`, and `complete.json`.

The interactive matrix passed four headless Chrome checks: 16 correct cells rendered, cell selection reported the actual 8/8 confusion, row-percentage display showed 100%, and no uncaught browser errors occurred. It requires no hosted application or external chart assets.

Native prediction also ran on the real `bbaf2n` **bin** interval, source frames 23–30, using measured alignment and the selected checkpoint. It returned **place**, which is incorrect. Its successful native process exit and input shape `[1,1,24,64,64]` establish execution only; this training-speaker example is not an additional accuracy benchmark. Proof: `.qa/word-prediction-check.json` and `.qa/word-prediction-native.json`.

## Validation and exact commands

Executed from `C:\Users\Dell\Downloads\LipNet-Alphabet-main\LipNet-Alphabet-main`:

```powershell
.\.venv\Scripts\python.exe pipeline/word_recognition/setup.py
.\.venv-word-recognition\Scripts\python.exe -m pytest tests/test_word_recognition.py -q
.\run_all.ps1 -Run runs/word_recognition/grid_commands_cpu_20261029
.\.venv-word-recognition\Scripts\python.exe -u -m pipeline.word_recognition.diagnostics --config pipeline/word_recognition/config.cpu.json --run runs/word_recognition/grid_commands_cpu_20261029
.\.venv\Scripts\python.exe -u -m pytest -q
.\.venv-word-recognition\Scripts\python.exe -u -m pytest tests/test_word_recognition.py -q --durations=3
.\.venv-word-recognition\Scripts\python.exe .qa/smoke-word-prediction.py
.\.venv-word-recognition\Scripts\python.exe .qa/check-word-final-integrity.py
node .qa/check-word-matrix.mjs
.\.venv-word-recognition\Scripts\python.exe -m compileall -q pipeline/word_recognition
```

The helper `.qa/smoke-word-prediction.py` records its full native prediction command. The first direct PowerShell prediction capture produced valid JSON but a nonzero shell status because MediaPipe's warning stderr was converted into `NativeCommandError` during redirection. The subprocess helper captured stdout/stderr directly and verified the Python **exit code 0**; warning text is retained separately. It did not change model outputs or hide an inference exception.

Results:

- Initial word checks: **17 passed**. An expanded intermediate run passed **20** checks.
- Existing application regression suite plus the then-current word checks: **460 passed, 3 skipped**, 122.63 seconds; `.qa/word-recognition-regression-tests.txt`. The three skips require an absent optional synthetic dataset fixture.
- Final word checks after diagnostics/cache additions: **22 passed**, 34.47 seconds; `.qa/word-recognition-final-tests.txt`. These cover invalid task/geometry, missing quotas, speaker/source leakage, pixel tampering, gradient propagation, deterministic evaluation, checkpoint mismatch/protection, early stopping, timeout behavior, exact completed-epoch optimizer/RNG resumption, quality-collapse detection, and bounded cache reuse.
- The isolated `.venv-word-recognition` setup passed `pip check` and native FFmpeg execution. Python module compilation passed.
- Root `run_all.sh` and setup Bash scripts passed `bash -n` using Git Bash; PowerShell wrapper parsing passed. The actual end-to-end run used the PowerShell wrapper, not Bash execution.
- All five actual orchestration stages returned **exit 0**. Timings were approximately: archive verification 54.13 s; preprocessing 228.40 s; smoke/import/input checks 171.64 s; training/import/input checks 132.54 s; held-out evaluation/import/input checks 62.18 s. Total stage time was approximately **10.8 minutes**. Separate diagnostics returned exit 0 while reporting failed recognition quality.
- Free disk was approximately **307.41 GiB** at final inspection, above the 20 GiB reserve. Large 26-word training was not launched; only its complete 36,400-segment configuration was validated.

## Preservation and remaining work

The run compared **110 protected files**, including all five existing recognition/representation checkpoints and frontend source/public assets. **No protected hash changed**, and backend-selection environment values matched before/after. A final independent rehash after prediction, diagnostics, and browser checks also matched all 110 files; `.qa/word-final-integrity.json` records this and the selected/resume checkpoint hashes. No backend server was restarted or replaced. The new checkpoints are confined to their own run; the default application alphabet model and LRW/AV-HuBERT workflows remain separate.

New implementation files are the `pipeline/word_recognition/` package, pinned requirements, two complete configurations, setup/run wrappers, and `tests/test_word_recognition.py`. `.gitignore` excludes the new isolated environment and generated licensed corpus frames. Dated cross-links were added to the earlier GRID/model/accuracy reports so their historical statements are not mistaken for current experiment results. Frontend source code was not edited.

The feasible implementation and bounded execution are complete. Remaining recognition work is substantive: improve train/validation learning using a larger genuine word cohort or justified pretrained representations; inspect crop/motion quality; reserve an appropriate fresh evaluation protocol before adapting to the exposed test result; then measure independent-speaker and real webcam performance/calibration. The supplied larger configuration is a way to collect a larger experiment, **not evidence that it will improve accuracy**. Production word recognition, automatic segmentation, general English/LRW-500 accuracy, and live deployment remain unverified.
