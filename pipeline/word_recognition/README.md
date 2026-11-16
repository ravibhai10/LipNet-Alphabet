# Automated recognition of aligned whole words

This independent research workflow trains a new 3D CNN + bidirectional GRU on **complete annotated spoken words** from the real GRID recordings already in this repository. It never uses the GRID letter slot, synthetic LRW fixtures, or an alphabet checkpoint as a word recognizer. Existing application models, frontend files, and backend environment selection are fingerprinted and preserved.

The default runnable experiment recognizes **bin, lay, place, set**. It prepares 24 training, 8 validation, and 8 test examples per word, across separate speakers. These small quotas make an actual end-to-end CPU run practical. `config.grid26.json` is a complete larger configuration for the 26 lexical GRID words, excluding spoken alphabet tokens. Neither configuration is unrestricted English, LRW-500, continuous speech, or sentence recognition. Annotation timing supplies the word boundaries; the model does not discover them.

## Run the whole workflow

From the repository root in Windows PowerShell:

```powershell
.\run_all.ps1
```

In Bash, including Git Bash on Windows:

```bash
bash run_all.sh
```

Both wrappers provision `.venv-word-recognition` when needed, then execute acquisition → preprocessing → real smoke/gradient diagnostics → training → held-out testing → learning-quality diagnostics. Structural failures stop subsequent stages. A completed computation can still produce a failed recognition-quality assessment; it never automatically activates a model. The Python entry point is also available:

```powershell
.\.venv-word-recognition\Scripts\python.exe -u -m pipeline.word_recognition.run_all
```

Each invocation receives a fresh run directory under `runs/word_recognition`. An explicit run and configuration can be supplied. To resume the existing dated experiment, including its optimizer, scheduler, loader RNG, and PyTorch RNG at the last completed epoch:

```powershell
.\run_all.ps1 -Run runs/word_recognition/grid_commands_cpu_20261029 -Resume
```

This resumes an interrupted epoch from the preceding completed epoch, discarding partial epoch updates. It does not claim mid-batch restoration. Changing the vocabulary, quotas, model, preprocessing, or training configuration requires a **new run**, because the contract is fingerprinted. Repeating evaluation of an unchanged selected checkpoint reuses identical report artifacts; do not tune architecture or thresholds against test results.

The larger experiment is executable but **has not been launched**:

```powershell
.\run_all.ps1 -Config pipeline/word_recognition/config.grid26.json
```

That configuration requests 26,000 training + 5,200 validation + 5,200 test word segments, 30 epochs, and a 60-minute training budget per invocation. It is a separate experiment. Its source validation and crop preparation can take hours on this machine. The budget stops incomplete training before test evaluation and leaves completed epoch checkpoints for explicit resumption. It does not impose a deadline on integrity verification or preprocessing.

## Physical modules and contracts

| File | Responsibility |
| --- | --- |
| `setup.py`, `setup.sh`, `setup.ps1`, `requirements.txt` | Dedicated Python 3.11 environment, compatible CPU torch/torchvision/torchaudio, MediaPipe/OpenCV, NumPy/Pandas/tqdm/scikit-learn, bundled FFmpeg, and dependency checks |
| `download.py` | Actual official GRID acquisition when needed; comparison of existing archive lengths and SHA-256 fingerprints; validated source inventory |
| `preprocess.py` | Real annotation-bounded words, measured facial alignment, canonical grayscale mouth PNGs, deterministic speaker-balanced selection, exclusion log, and leakage/quota checks |
| `data.py` | Verified lossless input loader, bounded uint8 RAM cache, and training-only coherent crop/brightness/flip/frame-drop augmentation |
| `model.py` | New Conv3D → two-layer bidirectional GRU → word softmax classifier; no legacy weight reinterpretation |
| `smoke.py` | Two real differently labeled samples, layer geometries, finite outputs/loss, all-parameter gradient diagnostics, dependency/FFmpeg checks, CPU timing |
| `train.py` | AdamW, smoothed cross-entropy, ReduceLROnPlateau, clipping, early stopping, atomic best/last checkpoints, complete-epoch resumption, and time budget |
| `test.py` | Selected checkpoint on test speakers only; accuracy/WER, precision/recall/F1, counts, per-sample predictions, CSV and interactive HTML confusion matrices |
| `diagnostics.py` | Majority-baseline comparison, single-class-collapse and zero-recall detection; separates computational success from recognition quality |
| `predict.py` | Offline inference on one explicitly bounded real word using this run's own selected checkpoint |
| `run_all.py`, root `run_all.sh` and `run_all.ps1` | Strict orchestration, command/stage logs, resource evidence, and protected-file integrity checks |

Direct module invocation is supported using `python -m pipeline.word_recognition.MODULE`. For example, after a run has prepared its frozen dataset:

```powershell
.\.venv-word-recognition\Scripts\python.exe -m pipeline.word_recognition.smoke --config pipeline/word_recognition/config.cpu.json --run runs/word_recognition/grid_commands_cpu_20261029
.\.venv-word-recognition\Scripts\python.exe -m pipeline.word_recognition.train --config pipeline/word_recognition/config.cpu.json --run runs/word_recognition/grid_commands_cpu_20261029 --resume
.\.venv-word-recognition\Scripts\python.exe -m pipeline.word_recognition.test --config pipeline/word_recognition/config.cpu.json --run runs/word_recognition/grid_commands_cpu_20261029
```

The original real GRID `bbaf2n` alignment marks **bin** at 23–30 source frames (zero-based, exclusive end). An executable prediction example for this trained four-command model is:

```powershell
.\.venv-word-recognition\Scripts\python.exe -m pipeline.word_recognition.predict --config pipeline/word_recognition/config.cpu.json --run runs/word_recognition/grid_commands_cpu_20261029 --video data/external/grid/s1/corpus/T01/video/s1/bbaf2n.mpg --start-frame 23 --end-frame 30
```

That example is from a training speaker; its prediction must not be presented as held-out accuracy. The actual test report records only the separately assigned test speakers.

## Data, acquisition, and integrity

The [official GRID distribution](https://spandh.dcs.shef.ac.uk/gridcorpus/) makes the corpus available for research use. It supplies video and word alignments; speaker 21 has no available video. This workflow uses the 33 available speakers already acquired here and respects those research conditions. It does not assume that LRW, TCD-TIMIT, MIRACL-VC1, or MuAViC access conditions are interchangeable, and does not obtain restricted LRW data.

The existing official downloader parses actual publisher links, retries and resumes HTTP ranges, rejects HTML/truncated downloads, verifies ZIP CRCs/TAR annotation members, and safely extracts archives while preserving existing files. Its ZIP/TAR handling uses the standard libraries' format detection, including compressed TAR readers. No made-up corpus URLs are added. GRID publishes no archive SHA-256 values; recorded hashes are **local fingerprints**. The new stage compares current archive bytes against those recorded fingerprints rather than describing them as publisher-signed checksums. Changed archives stop the run and are never silently overwritten.

The validated source inventory records the decoded frame count, byte and thumbnail hashes, strict alignment grammar, and excluded damaged source videos. The new pipeline rehashes each selected original video and annotation before tracking. Word labels come from the actual `.align` token and bounds, not filenames, letter-slot labels, or a prediction. GRID alignment units are **25,000 per second**. Word intervals use floor(start × fps / 25000) and exclusive ceil(end × fps / 25000), retaining at least two real frames. They may contain coarticulation at boundaries.

Selection is deterministic and rotates through speakers for each class. The existing source split keeps s1–s20/s22–s24 in training, s25–s29 in validation, and s30–s34 in testing. No speaker or source utterance may cross partitions. Repeated segments, duplicate decoded mouth pixels, missing classes, wrong annotations, or insufficient class/speaker quotas stop completion. Failed tracking candidates are logged and replaced from the **same split** before any training; no fake crops or zero tensors are substituted.

Canonical PNGs are stored as:

```text
data/processed/grid_whole_words/<config fingerprint>/<speaker>/<word>/<sample>/frame_0000.png
```

Sample contracts bind source/annotation/landmarker hashes, word bounds, configuration, and decoded pixels. Completed cached samples can be verified and reused; partial PNG writes recover only when existing frames match. Source recordings and existing letter manifests are not re-encoded or rewritten.

## Preprocessing and architecture

Measured MediaPipe eye landmarks establish rotation; the measured mouth establishes a centered 96 × 96 similarity crop. At least 80% of source frames must track successfully, and an interior gap longer than five frames is rejected. Crops become grayscale **64 × 64** pixels and are linearly interpolated to **24 frames**, with uint8 rounding before storage. Model input is float32 `[B,1,24,64,64]` in `[0,1]`.

Three Conv3D blocks use 8/16/32 channels, GroupNorm and SiLU, reducing spatial size 64 → 32 → 16 → 8 while retaining time. Spatial pooling produces `[B,24,32]`; two bidirectional GRU layers with hidden size 48 per direction produce `[B,24,96]`. Temporal mean pooling and a linear head produce one logit per configured word. The four-word model has **83,660 parameters**. This is a new model trained from scratch; neither old CNN–BiLSTM nor frozen AV-HuBERT weights are loaded.

Training applies one spatial shift and photometric adjustment coherently across the clip, optional horizontal mirroring, and simulated frame loss by holding the preceding frame for 2–3 samples. Validation/test are deterministic. A bounded 128-clip uint8 cache reduces repeated PNG I/O without loading an entire large corpus into RAM. Checkpoints embed vocabulary/order, architecture/configuration, data and landmarker fingerprints, selected epoch, optimizer/scheduler, and RNG state. Loading rejects an incompatible experiment.

## Diagnostics and metrics interpretation

The smoke report includes every Conv3D/GRU shape, per-parameter gradient norm/nonzero count, total gradient norm, and timing on two **real prepared words**. Finite nonzero gradients for that batch do not guarantee that optimization will never stall; training continues to reject nonfinite loss/gradient values and logs norm ranges.

Only validation loss drives learning-rate scheduling, checkpoint selection, and early stopping. Test samples never enter training/selection forward passes. Test output includes word accuracy, per-class correct/total and precision/recall/F1, ranked predictions, and confusion counts with rows=truth and columns=prediction. The HTML matrix works offline, supports count/row-percentage display, and provides keyboard/click cell details.

For one reference word and exactly one predicted word per pre-segmented sample, **WER = substitutions / samples = 1 − accuracy**. Insertions/deletions are structurally zero in this task. This is not continuous-speech or sentence WER. Top-k uses k=min(5,vocabulary size); with four words it necessarily includes all classes and is uninformative. A confusion matrix records word errors, not phoneme/viseme identities; those require additional annotations. Scores are uncalibrated.

This is an executable, tested training/evaluation workflow. Small GRID word results do not establish production recognition, general English performance, live boundary detection, webcam robustness, or an improvement over historical models. Real run results and preservation evidence are recorded separately in the root `AUTOMATED_WORD_PIPELINE_REPORT.md`.
