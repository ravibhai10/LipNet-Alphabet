# LRW whole-word pipeline: implementation and evidence

Client date: **2026-10-29**. Direct classification of one complete word per contextual video clip is implemented as an offline workflow. **A genuine, evaluated LRW-trained checkpoint is not verified in this repository.** Full licensed LRW acquisition, real training, and held-out recognition evaluation have not run. No browser model was activated.

## What exists

`data/vocab.txt` has **500 unique nonempty uppercase words**, in exactly the same order as the publisher's public vocabulary. It starts ABOUT, ABSOLUTELY, ABUSE and ends YEARS, YESTERDAY, YOUNG. Class indices follow file order. SHA-256: `c882e2ffcbf301d4145a33b753d3c70af36a1d9e08a0a1cd1f93c01145c421fb`. The complete vocabulary is in that file and the earlier `MODEL_CAPABILITIES_AUDIT.md`; its presence does not establish training data or a trained recognizer.

| Artifact / mode | Exact architecture and output | Existing checkpoint | Verified execution | Evidence and limitations |
| --- | --- | --- | --- | --- |
| Alphabet clip classifier | `AlphabetCNNBiLSTM`, 913,274 parameters, 26 logits | `models/alphabet_classifier.pt` | Strict load and finite CPU forward; current `/api/predict` adapter | A–Z clip classification, not word recognition or CTC. Historical alphabet metrics do not evaluate LRW. |
| Legacy word-sized export | Same model definition with 500 outputs, 974,420 parameters | `runs/finetune/lrw_500words/alphabet_classifier.pt` | Strict load; real public word example gives finite `[1,500]` logits | No embedded class order, vocabulary fingerprint, input preprocessing contract, or verifiable real LRW training provenance. Not accepted by the new word predictor. |
| Legacy word-sized best model | Same 500-output definition | `runs/finetune/lrw_500words/best_model.pt` | Same real-input execution check | Stored epoch 1 and validation accuracy 0.0666667 are historical, unaudited values, not a real LRW benchmark. Different tensors from the export above. |
| Legacy word-sized epoch checkpoint | Same 500-output definition plus optimizer state | `runs/finetune/lrw_500words/checkpoint_epoch_1.pt` | Same real-input execution check | Model tensors match `best_model.pt`; no independently verified dataset/evaluation provenance. |
| AV-HuBERT representation encoder | Official Base encoder, 768-dimensional temporal features | `models/external/avhubert/base_vox_iter5.pt` | Previously verified CPU encoding on genuine GRID video | Representation checkpoint, not a 500-word classifier. No supervised word head is present. Its pretraining clusters are not English word classes. |
| New audited whole-word artifact | Existing CNN–BiLSTM, exactly 500 logits; one word per clip | Future new-run `word_model.pt` | Temporary software tests exercise training, selection, save/load, prediction, and held-out report generation | No real artifact or recognition result exists yet. Prediction CLI and evaluation require the new metadata contract. |

The two `face_landmarker.task` assets are landmark detectors, not recognition checkpoints. The 90 `.pt` files under `data/processed/lrw/` are cached input tensors, not trained models. No compatible trained CTC decoder or sentence-recognition checkpoint was found.

Architecture is taken from `models/alphabet_model.py` and strict checkpoint loading: grayscale per-frame Conv2D channels 16/32/64 with batch normalization, adaptive/max pooling, 1,024 frame features; one bidirectional LSTM layer with hidden size 96 per direction; final time-step projection 192 → 128 → 500. This is a clip classifier, **not a 3D-CNN and not CTC**. `models/alphabet_arch.txt` contains stale architecture information. New word inputs are `[B,29,1,140,46]`, float32 in `[0,1]`; width/height are intentionally retained from the existing architecture.

## Actual local data and real-input smoke

`data/lrw/` contains **45 synthetic development MP4s**, 15 each in train/valid/test and only 15 represented words. All 45 have the same source SHA-256, `33be06600bab2ee0d28aeca31c467074278e83ad31311e1526e2687e2412db8c`, with conflicting folder labels and content repeated across splits. There are **zero original annotation TXT files** there. `scripts/setup_lrw_dataset.py:create_mock_video` explains these fixtures. Existing manifests/caches cannot establish LRW recognition or split integrity.

The [official Oxford LRW page](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html) publishes a separate public AFTERNOON example. The audit downloaded that publicly linked example and its metadata to `.qa/lrw-public-example/`, outside training inputs. It is a genuine **256 × 256, 29-frame, 25fps** clip. Its SHA-256 is `6d47e997e8c28d8a05c6903890d67e0171554ac7e66fd22d215a72de2d66df5d`. Real measured MediaPipe alignment produced the expected input tensor; all three legacy 500-output checkpoints strictly loaded and returned finite logits. Their historical preprocessing and label mapping remain unverified, so this establishes execution only. No successful word recognition or accuracy is claimed. Raw evidence: `.qa/lrw-word-audit.json`.

The same public page supplies the [official class list](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/vocab.txt) and an [annotation example](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/data/AFTERNOON.txt). The metadata records broadcast reference, channel, program start, clip start, and word duration. The new importer takes labels from the official class directory and verifies the filename; it does not invent a label field in the original TXT.

## Acquisition and ingestion

The publisher requires approval under the applicable BBC Research & Development data-sharing agreement for the full corpus; see the [BBC dataset page](https://www.bbc.co.uk/rd/projects/lip-reading-datasets) linked by Oxford. BBC's page could not be fetched by the browsing tool in this session; Oxford's access instructions were verified directly. User clarification: no approved URL/credentials are available right now. No protected corpus download was attempted, and no application status was inferred.

`scripts/acquire_lrw.py` accepts the **actual approved Oxford HTTPS URL** with credentials held only in local `LIPNET_LRW_USERNAME` / `LIPNET_LRW_PASSWORD` variables, or an existing local full archive. It does not discover passwords, bypass a gate, construct download URL patterns, or download third-party checkpoints. Signed URL values/credentials are not written into acquisition reports. Cross-origin and HTTPS downgrade redirects are rejected; if the publisher changes its approved host or transport, verify that change before extending the allowlist.

Downloads retain `.part` files with range/entity metadata, validate HTTP status, Content-Range and total length, reject HTML/login bodies, and resume using Range/If-Range. A server ignoring Range restarts from byte zero rather than appending. The full archive must match Oxford's published MD5 **`0dee65af9c2087d22602e6bc2b91a744`** before extraction; SHA-256 is additionally recorded as a local fingerprint, not a publisher-provided checksum. This command supports the complete `lrw-v1.tar.bz2`, not independently guessed multipart URLs.

Extraction performs a streaming policy/space pass before writing, rejects traversal, Windows drive/ADS paths, symlinks/devices, and duplicate/case-colliding paths. Existing extracted bytes must match the archive. Interrupted member files resume only after their prefix matches the archive. It never overwrites a differing existing member. Whole-archive checksum validation and two bzip2 passes can themselves be expensive; they have not been run against the absent 70 GB corpus.

`scripts/lrw_words.py ingest` expects the direct official class root: `WORD/{train,val,test}/WORD_*.mp4` with matching original TXT files. It records source/annotation hashes, actual decoding, decoded-thumbnail fingerprints, annotations, class order, distributions, and explicit partitions. `val` is normalized to `valid`; source partitions are not randomized. Duplicated source/content/broadcast references, missing or orphaned annotations, wrong frame rates/lengths, unknown classes, and incomplete distributions stop completion. The full official distribution must contain every class: 800–1,000 training and 50 validation/test clips per word. Failures remain in the source; no substitute clips are fabricated. Decoder checks are cached by source SHA for resumption; completed dataset outputs are protected.

The protocol is **publisher date partitions, speaker independence unknown**. No fake per-clip speaker IDs are manufactured to satisfy the separate GRID/AV-HuBERT speaker-disjoint gate. Provenance and approved acquisition are explicit user declarations; file layout alone cannot authenticate arbitrary substituted video.

## Preprocessing, training, and evaluation

The old LRW reader/preprocessor resized the entire decoded frame into 140 × 46; its “mouth ROI” wording did not implement actual cropping. New `pipeline/lrw/whole_word.py` instead shares measured eye-axis alignment and mouth-centered 96 × 96 crops with the existing tracker, requires at least 80% tracking coverage, rejects interior gaps longer than five frames, resizes to the existing 140 × 46 model input, linearly interpolates tracked frames to 29, rounds to uint8, and uses `/255` normalization identically during training and prediction. It retains the contextual word clip rather than treating letter-slot bounds as word supervision. This new preprocessing must be trained and evaluated; no compatibility with undocumented legacy preprocessing is asserted.

Cache entries include source, tensor, preprocessing, vocabulary, and landmarker fingerprints. Existing unaudited `data/processed/lrw` tensors are not used. Compact uint8 tensors reduce disk use; no fake/zero tensors replace decoding or tracking failures. The legacy preprocessor now raises on failed decoding, protects existing outputs, and runs once instead of its former duplicate main invocation. Legacy `train_lrw_500words.py` permits explicitly requested mock tests only and redirects real training to this audited workflow, preventing generic letter manifests from being passed off as LRW.

Training uses the existing 500-output architecture, CPU and Windows-safe `num_workers=0`, configurable threads, coherent training-only spatial/photometric/temporal augmentation, label smoothing 0.1, AdamW, warmup/cosine learning rates, clipping, and deterministic seed recording. It selects the checkpoint using minimum unsmoothed validation cross-entropy; the held-out test is scored after selection. New output directories are required. Initialization is random by default; `--init-alphabet` optionally transfers the existing 26-class CNN/LSTM while resetting both classification layers, with its source hash recorded. `--init-word` accepts only a new audited artifact with the **same frozen dataset/cache partitions**, recording that optimizer/scheduler state resets. It is a warm start, not exact interrupted optimizer resumption.

New artifacts embed architecture ID, all 500 class names/order and fingerprint, preprocessing, manifest/cache/landmarker identity, selected epoch, validation loss, and initialization. The predictor rejects legacy weights, reordered vocabulary, mismatched preprocessing, invalid tensor shapes, and nonfinite weights/outputs. Confidence is explicitly uncalibrated.

`evaluate` is restricted to the original audited test partition/cache contract. It writes `evaluation.json`, `predictions.jsonl`, `per_word.csv` (correct, total, top-5-correct, accuracy for all words), and a **500 × 500 confusion matrix CSV** with rows=true and columns=predicted. Top-1/top-5 use actual evaluated counts. An unobserved class gets `accuracy: null`; a partial software fixture is not treated as full vocabulary evaluation. Real evaluation remains blocked by data and training availability.

## Resources and executable commands

Detected: Intel i3-1115G4, 2 physical cores / 4 logical processors, **7.74 GiB installed RAM**. Free disk snapshot: **317.75 GiB**. A four-batch shape-only CPU training check (first batch excluded from timing), using torch 2.2.2+cpu and two threads, measured about **0.304 seconds per training clip**. Extrapolating gives **33.8–42.2 hours per epoch** for 400,000–500,000 training clips, before tracking, I/O, validation, or thermal/load changes. This is a rough throughput estimate, not a completed epoch or recognition result. Thirty epochs could take many weeks on this machine.

At the upper 550,000 total clips, float32 input caches alone would need about **382.65 GiB**. The implemented uint8 cache estimate including per-tensor allowance is **99.86 GiB**, in addition to the publisher's approximately 70 GB archive, actual extracted files, filesystem overhead, validation sidecars, and a default **20 GiB reserve**. Extraction uses actual TAR sizes; caching checks its projected remaining size. The brief timing result is `.qa/lrw-word-cpu-preflight.json`. No large acquisition, full cache, or training job was launched.

Run from `C:\Users\Dell\Downloads\LipNet-Alphabet-main\LipNet-Alphabet-main`. Both existing Python environments have the necessary CPU dependencies; the main `.venv` was used for the new workflow tests. **Only after approved access exists**, set the real URL and credentials locally, then use:

```powershell
# Set LIPNET_LRW_APPROVED_URL to the actual supplied URL, plus the username/password
# variables if required. Do not paste credentials into source files or reports.
.\.venv\Scripts\python.exe scripts/acquire_lrw.py --url $env:LIPNET_LRW_APPROVED_URL --archive data/external/lrw_official/lrw-v1.tar.bz2 --agreement-confirmed --max-minutes 10

# Re-run the same download command to resume. Once the complete archive is verified:
.\.venv\Scripts\python.exe scripts/acquire_lrw.py --archive data/external/lrw_official/lrw-v1.tar.bz2 --extract-to data/external/lrw_official/extracted --agreement-confirmed

# Inspect the extracted root: use the actual directory directly containing ABOUT, etc.
$lrwClassRoot = 'data/external/lrw_official/extracted/lipread_mp4'
.\.venv\Scripts\python.exe scripts/lrw_words.py ingest --root $lrwClassRoot --output data/processed/lrw_official_v1 --agreement-confirmed

# First process at most 10 new clips. Re-run without --limit for the complete cache.
.\.venv\Scripts\python.exe scripts/lrw_words.py cache --dataset data/processed/lrw_official_v1 --output runs/cache/lrw_words_v1 --limit 10
.\.venv\Scripts\python.exe scripts/lrw_words.py cache --dataset data/processed/lrw_official_v1 --output runs/cache/lrw_words_v1

# Recheck timing before explicitly starting the long job. Choose fresh outputs.
.\.venv\Scripts\python.exe scripts/lrw_words.py preflight --output .qa/lrw-word-preflight-next.json
.\.venv\Scripts\python.exe scripts/lrw_words.py train --dataset data/processed/lrw_official_v1 --cache runs/cache/lrw_words_v1 --output runs/words/lrw_v1 --epochs 30 --batch-size 2 --threads 2

.\.venv\Scripts\python.exe scripts/lrw_words.py evaluate --dataset data/processed/lrw_official_v1 --cache runs/cache/lrw_words_v1 --checkpoint runs/words/lrw_v1/word_model.pt --output runs/evaluation/lrw_words_v1
.\.venv\Scripts\python.exe scripts/lrw_words.py predict --checkpoint runs/words/lrw_v1/word_model.pt --video .qa/lrw-public-example/AFTERNOON.mp4
```

`$lrwClassRoot` is an example expected package layout, **not a claim that extraction has occurred**; choose the real class root after authorized extraction. A second held-out evaluation is reproducibility tooling, not permission to tune repeatedly against test results. Standalone word prediction is CLI only. `/api/predict` remains the protected alphabet endpoint; no functioning HTTP word endpoint or automatic frontend model switch is claimed. GRID's 25-letter-slot manifests and features must not be supplied to this workflow.

## Inspection and validation record

Inspected existing files: `data/vocab.txt`, `data/schema.py`, `models/alphabet_model.py`, `models/alphabet_arch.txt`, `models/training_summary.json`, all five recognition/representation checkpoints in the table, `runs/finetune/lrw_500words/training_summary.json`, `pipeline/lrw/loader.py`, `data/lrw_dataset.py`, `scripts/setup_lrw_dataset.py`, `scripts/preprocess_lrw_dataset.py`, `scripts/train_lrw_500words.py`, `src/video_training.py`, `tracking/mouth_canonicalization.py`, `pipeline/avhubert/classification.py`, `pipeline/eval/classification.py`, `scripts/download_grid_official.py`, `scripts/audit_model_capabilities.py`, `app/main.py`, `app/services/browser_inference.py`, `app/services/inference_service.py`, `.gitignore`, and the LRW/accuracy/real-data-gate tests. Local corpus directories/manifests and existing audit/GRID evidence were inspected as well. No additional `AGENTS.md` was found for the project. Initial probes for `pipeline/lrw/__init__.py`, `app/api/inference.py`, and `pipeline/model` found no such paths; the actual inference services listed above were inspected.

New implementation: `pipeline/lrw/whole_word.py`, `scripts/lrw_words.py`, `scripts/acquire_lrw.py`, `scripts/audit_lrw_words.py`, `tests/test_lrw_whole_word.py`, and this report. Existing edits for this task: the legacy LRW preprocessing/training safeguards, `.gitignore` for new licensed-data outputs, and runtime audit dating in `scripts/audit_model_capabilities.py`. Frontend files, checkpoint bytes, and backend selection were preserved.

Functional commands executed (read/search commands used `Get-Content`, `rg`, `Get-Item`, `Get-PSDrive`, `Get-CimInstance`, and `git status` on the paths above):

```powershell
.\.venv\Scripts\python.exe scripts/audit_model_capabilities.py --in-process --output .qa/whole-word-before.json
.\.venv\Scripts\python.exe scripts/lrw_words.py preflight --output .qa/lrw-word-cpu-preflight.json
.\.venv\Scripts\python.exe scripts/audit_lrw_words.py --public-example --output .qa/lrw-word-audit.json
.\.venv\Scripts\python.exe -m pytest tests/test_lrw_whole_word.py tests/test_lrw_loader.py tests/test_lrw_vocab.py tests/test_accuracy_pipeline.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest tests/test_lrw_whole_word.py pipeline/grid/tests pipeline/avhubert/tests pipeline/eval/tests -q
.\.venv\Scripts\python.exe -m pytest tests/test_lrw_whole_word.py tests/test_lrw_loader.py tests/test_lrw_vocab.py tests/test_accuracy_pipeline.py pipeline/grid/tests pipeline/avhubert/tests pipeline/eval/tests -q
.\.venv\Scripts\python.exe .qa/check-lrw-preservation.py
.\.venv\Scripts\python.exe -m py_compile scripts/acquire_lrw.py scripts/audit_lrw_words.py scripts/lrw_words.py pipeline/lrw/whole_word.py scripts/audit_model_capabilities.py
.\.venv-avhubert-cpu\Scripts\python.exe scripts/lrw_words.py --help
.\.venv\Scripts\python.exe scripts/acquire_lrw.py --help
.\.venv\Scripts\python.exe scripts/lrw_words.py predict --checkpoint runs/finetune/lrw_500words/best_model.pt --video .qa/lrw-public-example/AFTERNOON.mp4
```

- Initial targeted run: **50 passed**, 28.77 seconds. Subsequent expanded checks cover the later additions.
- Main configured regression suite: **440 passed, 3 skipped**, 99.74 seconds; `.qa/lrw-word-regression-tests.txt`. Skips require the absent optional synthetic dataset fixture.
- Expanded targeted run: **56 passed**, 27.81 seconds; `.qa/lrw-word-targeted-tests.txt` (29 word tests plus 27 GRID/AV-HuBERT/evaluation checks).
- Final targeted run after the last code changes: **81 passed**, 19.83 seconds; `.qa/lrw-word-final-tests.txt`. This includes initialization partition checks and the legacy script safeguards. OpenCV's “moov atom not found” message comes from the deliberately invalid-video rejection test.
- The explicit legacy `predict` command failed closed as intended: exit 2 with “Not an audited LRW word artifact”; the encompassing PowerShell batch reported a nonzero exit. It did not emit fabricated word labels.
- In-process ASGI audit returned HTTP 200 on the alphabet path and confirmed unchanged model/backend selection. It is not evidence of a running external web server. That helper's pre-existing fixed audit-date field was corrected to runtime UTC dating after the first output; `.qa/lrw-word-audit.json` has the actual 29 October timestamp.
- Real public word decoding, measured mouth preprocessing, publisher vocabulary comparison, and all three legacy 500-output finite forwards passed. All five original model SHA-256 values matched before/after that audit.
- The final preservation check passed: all five original checkpoint hashes match and in-process model availability returns HTTP 200 with `active_backend: alphabet`; evidence is `.qa/lrw-word-final-integrity.json`. Python compilation of all five listed implementation/audit modules also passed.
- Temporary training fixtures exercise software only; their metrics are **not** real-word recognition evidence.

Remaining blockers: approved BBC access and actual full LRW files; full source validation and measured-mouth cache completion; a new real training run; untouched held-out evaluation; and, if subsequently requested and justified by that evidence, a separately designed word HTTP/live integration. No accuracy improvement is established by this work.
