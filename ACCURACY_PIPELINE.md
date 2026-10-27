# AV-HuBERT training and live stabilization

The training pipeline is ready for labeled data. Real head training is blocked by missing authentic isolated A–Z recordings with independent speakers. The active live backend still uses the existing alphabet CNN-BiLSTM checkpoint. There is no new measured recognition accuracy or guaranteed accuracy increase. See [PROJECT_STATUS.md](PROJECT_STATUS.md) and [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for the latest completion audit; the original validation counts below describe the earlier milestone.

## Verified pretrained encoder

The [official AV-HuBERT project](https://github.com/facebookresearch/av_hubert) supplies a pretrained representation model. Classification requires a separately trained supervised head. The publisher's [checkpoint page](https://facebookresearch.github.io/av_hubert/) provides the Base VoxCeleb/LRS3 checkpoint used here:

- Download: [base_vox_iter5.pt](https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/base_vox_iter5.pt)
- Local file: `models/external/avhubert/base_vox_iter5.pt`
- File size: 1,240,185,329 bytes; ignored by Git.
- Locally computed SHA-256: `dee8b0651dbea3a21cda0f2c52b956e3a878d5f6ee503d427d6a4044bb95c9cc`. No publisher checksum was supplied for comparison.
- Follow the publisher's license, including its noncommercial research restriction.

The real GRID video `data/external/grid/s1/corpus/T01/video/s1/bbaf2n.mpg` was decoded, tracked with real MediaPipe landmarks, aligned, and encoded on Windows CPU. It produced finite **29 × 768** features. The measured 4.75 seconds includes decoding/tracking/feature extraction after encoder loading; it is one clip measurement, not a real-time performance guarantee. Report: `.qa/avhubert-real-smoke.json`.

`pipeline/avhubert/loader.py` reads the checkpoint's actual task preprocessing: grayscale `[0,1]`, center crop 88 × 88, mean 0.421, standard deviation 0.165, video tensor `[B,1,T,H,W]`. Its frozen video-only path calls the official `extract_finetune(..., audio=None, mask=False)` method. The pretraining `extract_features` method requires audio in this version.

Use the existing `.venv-avhubert-cpu` environment. Verified: Python 3.11.9, PyTorch 2.2.2+cpu, editable vendored Fairseq, OmegaConf 2.3.1, and `python_speech_features==0.6`. MediaPipe and backend dependencies are also present there. The regular `.venv` does not include Fairseq. This task added the speech-features dependency to the CPU environment and optional CPU package dependencies.

The vendored AV-HuBERT modules now distinguish package imports using `__package__`, and the vendored Fairseq checkpoint loader supports OmegaConf 2.3's object configuration API. Preserve these compatibility changes when refreshing `external/av_hubert` or its nested Fairseq checkout.

## Add labeled clips

Copy `examples/avhubert_annotations.example.jsonl` to your own dataset directory and replace every example path and identifier with real recordings. The example describes a two-class format; it contains no videos and is not training data.

Each nonempty JSONL line describes one segmented letter or word:

```json
{"id":"s01-a-001","video":"clips/a001.mp4","label":"A","speaker_id":"speaker01","split":"train","data_source":"real_camera","input_kind":"raw_video"}
```

- `video` is resolved relative to the manifest. Recordings must contain the labeled utterance; trim long silence before extraction.
- `id`, `speaker_id`, and `label` are nonempty strings. Labels are case sensitive.
- `split` is `train`, `valid`, or `test`. All three partitions must be nonempty and have separate speakers: at least three speakers are needed. Every evaluation class must occur in training.
- `data_source: real_camera` is an explicit provenance declaration. Only use it for authentic camera/corpus video. Local generated LRW development fixtures are unsuitable. The currently downloaded GRID subset contains only one speaker and cannot meet this evaluation protocol.
- `input_kind: raw_video` uses measured eye-axis alignment and a mouth-centered 96 × 96 similarity crop. Use `mouth_roi` only for already aligned mouth ROI video following the same crop convention. Full-face video must use `raw_video`.
- Source SHA-256 hashes are calculated from video bytes. Repeated content cannot cross partitions or have conflicting labels. A speaker cannot cross partitions. Missing inputs, unmarked/synthetic sources, and unknown evaluation classes fail before extraction.
- Raw clips are limited to 500 frames; tracking must cover at least 80% of frames, with no tracked interior gap longer than five frames.

For the existing browser alphabet endpoint, train all 26 labels `A` through `Z`. Word heads work through the prediction CLI; that endpoint deliberately rejects them.

## Extract, train, and predict

### Collect and prepare authentic alphabet recordings

Current inventory: 0 registered real alphabet recordings, 1,000 GRID utterances from one speaker, and 45 generated LRW fixtures (15 per split). The historical `models/training_summary.json` reports 12% test accuracy over 100 samples, but lacks the original speaker/source/test-membership audit. It is not a newly measured or eligible independent baseline.

Collect each speaker in a separate session directory. Use at least three distinct people, assigning an entire person to a single partition; all 26 letters must appear in each partition for a complete alphabet benchmark. One clip per class per partition (78 total) is only the schema minimum, not enough evidence of generalization. Collect repeated utterances, more people, and varied lighting/camera conditions. Keep the test speakers untouched while selecting model settings or thresholds.

Example collection command (opens a physical camera only when you run it):

```powershell
.\.venv\Scripts\python.exe scripts/record_alphabet_dataset.py --output alphabet_dataset --speaker speaker01 --session session01 --samples-per-letter 5
```

Repeat for `speaker02` and `speaker03`. These files follow `alphabet_dataset/recordings/<LETTER>/<speaker>/<session>/<recording>/video.mp4` plus `metadata.json`. Keep only correctly labeled, visible utterances; segment long silence. The annotation helper preserves original files and never overwrites an existing manifest:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_alphabet_annotations.py --recordings alphabet_dataset --train-speakers speaker01 --valid-speakers speaker02 --test-speakers speaker03 --output data/my_dataset/clips.jsonl
```

Audit the manifest without loading an encoder or starting training:

```powershell
.\.venv\Scripts\python.exe scripts/avhubert_classifier.py validate --manifest data/my_dataset/clips.jsonl --alphabet
```

This checks authentic-source declarations, existing videos and their hashes, complete A–Z coverage, disjoint speakers, duplicate content, and split/class counts. A provenance declaration must reflect real recordings; the validator cannot establish authenticity from a declaration alone.

### Cache and train

Run in the repository directory containing `app`, `models`, and `frontend`. Choose new output directories for each run; existing nonempty outputs are protected.

```powershell
.\.venv-avhubert-cpu\Scripts\python.exe scripts/avhubert_classifier.py extract --manifest data/my_dataset/clips.jsonl --output runs/features/my_dataset_v1 --frames 29 --augment-train-copies 2 --alphabet

.\.venv-avhubert-cpu\Scripts\python.exe scripts/avhubert_classifier.py train --features runs/features/my_dataset_v1 --output runs/avhubert/my_head_v1 --epochs 30 --batch-size 16 --lr 0.001 --warmup-epochs 5

.\.venv-avhubert-cpu\Scripts\python.exe scripts/avhubert_classifier.py predict --head runs/avhubert/my_head_v1/head.pt --video data/my_dataset/held_out_clip.mp4
```

Extraction caches real `[T,768]` features once, keeping the large encoder frozen. Active clips are linearly interpolated to 29 frames. Optional augmented copies enter the training partition only: one spatial jitter up to ±3 pixels for the whole sequence, coherent contrast 0.8–1.2 and brightness ±0.1, and a contiguous 2–3-frame temporal mask. Validation/test extraction has no random augmentation. These augmentations operate on the normalized mouth clip before frozen feature extraction.

Training defaults to masked temporal mean pooling plus a linear classifier. Add `--hidden-dim 128` for an MLP. The head uses label smoothing 0.1, AdamW, five warmup epochs followed by cosine decay to 1e-6, and gradient clipping. Short runs shorten warmup automatically. The best head is chosen by validation loss; the held-out test partition is evaluated after selection. Padding is excluded from pooling.

Outputs are `head.pt` and `training_summary.json`, including the actual vocabulary, encoder fingerprint, preprocessing, best epoch, per-epoch losses/accuracy, and held-out test metrics. Loading rejects an encoder fingerprint/preprocessing mismatch, invalid vocabulary, or an incomplete training/evaluation artifact.

Training also writes `classification_report.json` (precision, recall, F1, support, macro F1, and overall accuracy) and `confusion_matrix.csv` (true classes in rows, predictions in columns). New caches fingerprint each feature file and validate both embedding width and temporal length before training. The head records training speakers and video hashes for later overlap checks.

Reproduce test evaluation from the audited feature cache after training:

```powershell
.\.venv-avhubert-cpu\Scripts\python.exe scripts/avhubert_classifier.py evaluate --features runs/features/my_dataset_v1 --head runs/avhubert/my_head_v1/head.pt --output runs/evaluation/my_head_v1_test
```

The evaluation command verifies encoder/preprocessing identity and rejects test speakers or videos used to train that head. Cached-head timing excludes encoder extraction and is labeled accordingly. A fair baseline comparison must use the same eligible test samples and an audited training-membership record for both models. The original CNN checkpoint lacks that record; no certified independent comparison is possible from its historical summary alone.

Optional auxiliary viseme supervision requires `viseme_label` (integer cluster index) and `viseme_source: phoneme_annotation` on **every** annotation, then `--viseme-classes N --viseme-weight 0.2` during training. Provide phoneme-aligned, domain-appropriate labels. Spoken letter names are not single consonant phonemes, so the tool does not invent a B/P/M grouping from alphabet class names. The optional head supervises pooled frozen encoder representations; it does not fine-tune the encoder's CNN.

## Activate a trained alphabet head

After training a complete A–Z head, restart the backend from a PowerShell terminal using the CPU environment:

```powershell
$env:LIPNET_MODEL_BACKEND = 'avhubert'
$env:LIPNET_AVHUBERT_HEAD = (Resolve-Path 'runs/avhubert/my_head_v1/head.pt').Path
.\.venv-avhubert-cpu\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Stop the existing backend process before starting a replacement on port 8000. Inspect `/health` for the selected model and actual inference readiness. A missing head reports unavailable; the encoder alone cannot return class predictions. The current running backend was left on the protected CNN-BiLSTM until real head training is possible.

To select the existing model again, set `$env:LIPNET_MODEL_BACKEND = 'alphabet'` and restart. Its original checkpoint requires 30 frames, so the adapter linearly resamples incoming windows to 30. New AV-HuBERT heads use their recorded 29-frame setting.

The workspace training form continues to run the alphabet CNN-BiLSTM trainer. The AV-HuBERT feature/head workflow above is a separate CLI, as explained in the interface. Legacy alphabet/LRW training tools also use the shared coherent augmentations and warmup/cosine schedule. LRW real-data training now requires an audited annotation manifest; synthetic fixtures require an explicit testing option and are marked accordingly.

## Live stabilization

`frontend/src/live-demo/stabilization.ts` and `useDemoPipeline.ts` now provide:

- Eye-axis alignment before mouth cropping, matching the Python alignment convention.
- A 29-frame FIFO at approximately 15 capture samples/second, with inference considered every four captured frames.
- Visual activity gating based on measured mouth-aperture range and recent motion energy. A static open mouth does not count as speaking.
- An 8-frame preroll/minimum window, a 350 ms movement hold, and an end-of-speech window. The complete selected segment is linearly resampled by the backend.
- One active request with a latest-window pending slot for CPU backpressure.
- EMA with alpha 0.6 over the full returned class probability vector, followed by two consistent top-1 windows before display. Older responses without a full vector use top-1 consensus only.
- Resets on new utterances, face loss, stop/unmount, and backend unavailability. Session history records confirmed results only.

The activity thresholds are heuristics needing held-out calibration. Smoothed confidence remains uncalibrated. Physical webcam recognition and accuracy gains have not been measured.

The original front-page colors remain charcoal `#0a0a0c`, ivory `#edebe4`, and mint `#8fdfcb`.

## Validation

- 130 Python checks passed, 2 skipped across the changed pipeline, browser inference, training/preprocessing, LRW loader, backend services, CPU runtime, corpus ingestion, and evaluation integrity tests.
- 6 live-stabilization tests passed using `npm run test:live` in `frontend` (Node 24 with TypeScript stripping).
- Production TypeScript/Vite build passed.
- 41 Chrome checks passed: nine routes at three viewport sizes, camera permission/lifecycle behavior, offline and failed API responses, reduced motion, and landing palette. No uncaught browser errors or failed resource responses.
- Real official-checkpoint CPU encoding passed on the GRID clip described above.
- Legacy alphabet checkpoint SHA-256 remains `7c00d5117b7fae9a76c05980ca70edd680e79be4c8a152ea2595c98ded90c4c3`.

The temporary feature/head fixtures in unit tests verify training and artifact plumbing; their metrics are not recognition results. No real head or new accuracy benchmark has been produced.
