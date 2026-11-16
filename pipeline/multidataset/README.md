# Silent-video recognition workflows

These adapters preserve dataset/task boundaries. The new MIRACL pilot is an
eight-word research experiment. It does not replace the browser alphabet model,
the older GRID four-word experiment, or any LRW checkpoint. A CTC decoding helper
does not constitute a trained continuous recognizer.

From the project root, use the existing isolated Windows CPU environment:

```powershell
.\.venv-word-recognition\Scripts\python.exe scripts/multidataset.py preflight --output .qa/my_preflight.json
.\.venv-word-recognition\Scripts\python.exe scripts/multidataset.py acquire-miracl --minutes 10
.\.venv-word-recognition\Scripts\python.exe scripts/multidataset.py ingest --dataset miracl --root data/external/miracl/original --output data/processed/multidataset/miracl_v1.jsonl
.\.venv-word-recognition\Scripts\python.exe scripts/multidataset.py validate --manifest data/processed/multidataset/miracl_v1.jsonl --verify-sources
```

The transfer follows the author site's public download and its ordinary Google
confirmation form. It rejects login/error HTML, bounds transfer size/time, retains
identified partials, verifies ranges/entities, records SHA-256, checks ZIP CRC,
and extracts without traversal, symlinks, case collisions, or replacing changed
files. No publisher checksum was available for this MIRACL archive: its SHA-256
is a local fingerprint, not a comparison with a published digest. Existing
archives and extracted files are reverified on reruns. Download time limits do
not include the finite archive-integrity and extraction passes.

If the author's public file becomes unavailable, configure an accepted Kaggle
account locally using its documented API token or legacy `~/.kaggle/kaggle.json`,
then use `kaggle datasets download -d apoorvwatsky/miraclvc1 -p data/external/miracl/archives`.
The automated run does not fall back to an unreviewed mirror archive, install a
credential, or accept a mirror's layout/license without inspection.

For a reproducible pilot, the subset is selected *before* model fitting:

```powershell
.\.venv-word-recognition\Scripts\python.exe scripts/multidataset.py ingest --dataset miracl --root data/external/miracl/original --pilot-repetitions 2 --output data/processed/multidataset/miracl_pilot_sources_v1.jsonl
.\.venv-word-recognition\Scripts\python.exe -m pipeline.multidataset.experiment --manifest data/processed/multidataset/miracl_pilot_sources_v1.jsonl --run runs/multidataset/my_miracl8_pilot --repetitions 2 --epochs 12 --minutes 8 --allow-training-exclusions
```

If these outputs already exist, use a **new** manifest/run path. An interrupted
experiment accepts `--resume` with the identical source contract and can increase
the maximum epochs. It restores optimizer/scheduler and Python/NumPy/Torch RNG
states, and verifies cached source/crop bytes. A completed held-out evaluation is
protected from further tuning. `--minutes` limits the training invocation at
epoch boundaries; preparation has a finite prespecified clip quota. One CPU
thread, batch size two and no data-loader subprocesses keep the pilot bounded.
The fixed confidence/margin rejection thresholds are uncalibrated, and cannot
guarantee rejection of arbitrary out-of-vocabulary speech.

The executed `miracl8_cpu_v1` preparation failed on four training clips and did
not train a model. `miracl8_cpu_v2` explicitly allowed those training-only
exclusions and reused the verified ROI cache with
`--cache-root runs/multidataset/miracl8_cpu_v1/clips`. It trained on 140 clips;
validation and test remained 48 each. Evaluation-clip tracking failures always
stop the run. Sources and the failed-run report remain preserved.

`run_multidataset.ps1` performs preflight/acquisition/ingestion; add `-Train` for
the bounded experiment or `-Train -Resume` to resume an incomplete experiment.
`run_multidataset.sh` forwards to that native Windows runner from Git Bash.
The existing `python -m pipeline.word_recognition.setup` provisions the separate
environment (including CPU torch/vision/audio, OpenCV, MediaPipe, NumPy, pandas,
scikit-learn, tqdm and bundled FFmpeg). No audio is read for inference. Avoid
reprovisioning an already working environment during another job.

## Dataset contracts

| Dataset | Ingestion and boundary policy |
| --- | --- |
| GRID | `ingest --dataset grid --root runs/word_recognition/grid_commands_cpu_20261029/words.jsonl --output data/processed/multidataset/grid_commands_v1.jsonl`. Reuses the real four-word baseline's annotation-bounded sources, never its letter-slot labels. Larger 26-lexical-word experiments remain separate under `pipeline/word_recognition/config.grid26.json`. |
| MIRACL | Native `speaker/{words,phrases}/class/repetition/color_NNN.jpg`. Depth is excluded. Ordered RGB frames have **unknown native FPS**; no timestamps are invented. Author table entries 01 `Hegin` and 10 `Well` remain unresolved and excluded, rather than being renamed `Begin`/`Web`. Phrase transcripts remain sequence supervision, never single-word labels. |
| TCD-TIMIT | Request noncommercial access from `sigmedia_database@tcd.ie`, with the official page's `[TCDTimit Question]` subject prefix. Place authorized originals under `data/external/tcd_timit`. Use a reviewed JSONL index with `id,video,transcript,speaker_id,split,view,provenance_reviewed:true`. Invoke `ingest --dataset tcd_timit --root data/external/tcd_timit --index data/external/tcd_timit/reviewed.jsonl --output data/processed/multidataset/tcd_v1.jsonl`. Unknown native annotation formats/units are not guessed. |
| MuAViC English | Its official `get_data.py` requires existing LRS3 `pretrain/trainval/test` directories; it does not automatically acquire English. Do not run its cleanup/CPU-wide worker code over this project. `ingest-muavic` accepts paired `--tsv` and `--transcripts` lists for all three native partitions plus a reviewed `--speakers` JSONL map. The exact example is below. Map rows need `id,speaker_id,split,view,provenance_reviewed:true`. Preserve publisher splits; unknown speaker IDs block a disjoint benchmark. Audio columns are discarded. |
| LRW | Use `scripts/acquire_lrw.py` and `scripts/lrw_words.py` only after BBC approval and an exact approved URL/local archive exist. The adapter consumes that tool's audited dataset directory. Synthetic local clips and vocabulary-only directories are rejected. Publisher splits are retained; undocumented speaker identities are not invented. |
| Roboflow | Optional **static phonetic images only**, version 1, current page license CC BY 4.0. No API key was available. After an authenticated reviewed classification export, use `ingest-roboflow --root data/external/roboflow_phonetic --metadata data/external/roboflow_phonetic/export.json --output data/processed/multidataset/phonetic_v1.jsonl`. Metadata must record exact source URL, version, license and `authenticated_export:true`; layout is `train|valid|test/phonetic_class/image`. Detection-format exports fail rather than becoming word classes. No speaker-disjoint claim is possible without identities. |

For TCD or MuAViC word crops, an optional reviewed index `alignment` file must
contain monotonic integer `start end token` rows, with an explicitly reviewed
`alignment_timebase_hz` and `alignment_method: native_reviewed` or
`forced_alignment_reviewed`. Tokens must reconstruct the transcript and map to
valid decoded video bounds. With sentence text alone the adapter emits sentences;
it never estimates word boundaries from a filename or model output. Audio-based
alignment is a separate, reviewed training-time preparation step, not an
implemented/verified forced aligner here.

After authorized English preparation, the native MuAViC invocation is:

```powershell
.\.venv-word-recognition\Scripts\python.exe scripts/multidataset.py ingest-muavic --root data/external/muavic/en --tsv data/external/muavic/en/train.tsv data/external/muavic/en/valid.tsv data/external/muavic/en/test.tsv --transcripts data/external/muavic/en/train.en data/external/muavic/en/valid.en data/external/muavic/en/test.en --speakers data/external/muavic/en/reviewed_speakers.jsonl --output data/processed/multidataset/muavic_v1.jsonl
```

## Pretrained word features and continuous decoding

The official AV-HuBERT Base checkpoint is already installed and verified as an
encoder, with no supervised word/sequence decoder. Its Fairseq environment is
`.venv-avhubert-cpu`. A full checkpoint load was deferred in this run because
available RAM was below the conservative 2 GiB extraction budget. The small
scratch word pilot is explicitly reported as such. To prepare a **separate GRID
whole-word** frozen-encoder experiment when resources permit:

```powershell
.\.venv-word-recognition\Scripts\python.exe scripts/multidataset.py ingest --dataset grid --root runs/word_recognition/grid_commands_cpu_20261029/words.jsonl --output data/processed/multidataset/grid_commands_v1.jsonl
.\.venv-word-recognition\Scripts\python.exe scripts/multidataset.py avhubert-manifest --manifest data/processed/multidataset/grid_commands_v1.jsonl --output data/processed/multidataset/grid_words_avhubert_v1.jsonl
.\.venv-avhubert-cpu\Scripts\python.exe scripts/avhubert_classifier.py extract --manifest data/processed/multidataset/grid_words_avhubert_v1.jsonl --output runs/features/grid_words_avhubert_v1 --frames 29
.\.venv-avhubert-cpu\Scripts\python.exe scripts/avhubert_classifier.py train --features runs/features/grid_words_avhubert_v1 --output runs/avhubert/grid_words_v1 --epochs 30 --batch-size 16 --lr 0.001 --warmup-epochs 5
```

No alphabet switch is appropriate for this four-word head. MIRACL's unknown FPS
does not silently enter a pretrained 25 FPS temporal contract. The current
AV-HuBERT preprocessing remains unchanged: gray `[0,1]`, 88px center crop,
mean 0.421/std 0.165, `[B,1,T,H,W]`, `audio=None`.

MuAViC's listed English checkpoint is **AVSR**; its demo explicitly loads
`['audio','video']`. This inspection did not establish visual-only inference for
that checkpoint. No decoder weight or tokenizer was downloaded. The AV-HuBERT
source has trained-decoder architectures, but no corresponding trained artifact
is available locally. Both facts block Track B, rather than justifying a fake
transcript or an audio-assisted silent-video claim.

`score-sequences --manifest ... --predictions ... --output ...` scores actual
held-out predictions with corpus-level WER/CER, per speaker/dataset and exact
sequence accuracy. Prediction JSONL requires the exact held-out `id`, real
`hypothesis`, `modality: video_only`, and `checkpoint_sha256`. This scorer records
provenance declarations; it cannot independently prove how predictions were
generated. Empty hypotheses are counted as errors. CER includes normalized
spaces/punctuation. Unit tests of CTC collapse/edit distance are not recognition.

`/api/recognition/capabilities` exposes offline experiment evidence and unavailable
Track B status without selecting or loading a model. The live page shows this
research status. No Word/Free-Speech mode pair is advertised while both tracks
lack verified operational models. Existing routes and backend selection remain.
