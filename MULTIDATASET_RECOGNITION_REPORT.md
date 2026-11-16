> Latest whole-word result, **2026-10-29**: a separate frozen AV-HuBERT eight-word head achieved **182/192 (94.79%)**, versus **24/192 (12.50%)** for the preserved scratch model on the same clips. A private research Word Mode runs separately on port 8001. A genuine publisher-trained 500-word LRW model also passed one research CLI smoke; its local accuracy has not been benchmarked. See [measured results, scope and commands](WORD_ACCURACY_IMPROVEMENT.md). Historical results below remain unchanged. This does not establish arbitrary-English, webcam or sentence accuracy, and the eight-word head is incompatible with the 26-class alphabet adapter.

# Multi-dataset silent-video recognition: executed results

Date: **2026-10-29, Asia/Calcutta**. The accessible MIRACL corpus was acquired,
validated and used for real word-model training/evaluation. The new model is
**offline research only**: 8/48 held-out clips correct, with weak confidence and
predictions heavily concentrated on one word. No reliable general-English word
recognizer or continuous video-to-text model has been established.

## Acquired data and access boundaries

| Dataset | Actual local state | Task and limitations |
| --- | --- | --- |
| GRID | Reused the existing 33-speaker corpus and 32,891 validated sentences; no redownload | Existing 25-letter-slot data stays separate. The previously prepared real four-word experiment has 160 timestamp-bounded clips. A new normalized manifest and AV-HuBERT word-annotation export were created from those sources. |
| MIRACL-VC1 | Downloaded the author's complete public archive, checked CRC and decoded/hashed all RGB sequences | 3,000 instances across 15 speakers: 1,200 usable word sequences in 8 unambiguous classes, 1,500 phrase sequences, 300 sequences excluded for unresolved word labels. RGB/source validation is distinct from mouth-tracking acceptance. |
| TCD-TIMIT | No authorized corpus present | Noncommercial access request required. Reviewed sentence/word-alignment adapter is implemented; actual corpus integration remains unverified without files. |
| MuAViC English | Official source inspected and cloned; no corpus/model downloaded | English preparation requires local LRS3 originals. No 436-hour acquisition or sequence training was launched. Native TSV/transcript adapter requires reviewed speaker identities. |
| Roboflow | No export downloaded; no API credential available | Optional version-1, 38-class static phonetic imagery, current page CC BY 4.0. Not word/temporal supervision. Local authenticated classification-export adapter is implemented but not validated on a real export. |
| LRW | No authorized complete dataset | BBC agreement/approved source still required. The 45 local videos are duplicate development fixtures. Three legacy 500-output files load as classifiers, but genuine LRW training and class mapping remain unverified. |

Official sources/conditions: [GRID](https://spandh.dcs.shef.ac.uk/gridcorpus/),
[MIRACL author](https://abenhamadou.github.io/miraclvc1/),
[TCD-TIMIT](https://sigmedia.tv/datasets/tcd_timit/),
[MuAViC](https://github.com/facebookresearch/muavic),
[Roboflow dataset](https://universe.roboflow.com/lipreading/lip-reading-h5j77),
[LRW access](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html).
MIRACL specifies research use. MuAViC specifies CC-BY-NC 4.0 plus upstream
source conditions. Access conditions were not bypassed and no external messages
were sent. Kaggle credentials were unnecessary because the author-published
public archive worked through its normal Google confirmation form.

MIRACL acquisition evidence:

- Archive: `data/external/miracl/archives/MIRACL-VC1_all_in_one.zip`.
- Bytes: **5,428,961,345**. SHA-256:
  `eb6dd7ed59333084394115df755060209f4649a1f001cb08910a2aa4da413824`.
- Publisher checksum unavailable; this is a locally calculated fingerprint.
- ZIP CRC passed. **69,525 files**, **6,171,785,373 expanded bytes**: 34,762 RGB
  JPEG frames, 34,762 depth PNG frames, and one calibration file. Depth never
  enters the RGB model. Archive and originals are retained.
- Actual public probe: HTTP 200, `application/octet-stream`, matching length,
  ZIP signature `504b0304`. Evidence: `.qa/miracl-public-confirmation-probe.json`.
  The acquisition manifest records completion, source, size, hash and extraction;
  its first-run transfer HTTP fields were not all retained. Range resumption is
  exercised by software tests, not claimed as a measured MIRACL network resume.
- Manifest: `data/external/miracl/acquisition.json`; full ingestion log:
  `.qa/miracl-ingestion.txt`.

## Labels, exact partitions and manifests

Supported word labels, in model index order: **choose, connection, hello,
navigation, next, previous, start, stop**. Author-table word IDs 01 `Hegin` and
10 `Well` conflict with commonly used mappings; they were not silently renamed
or trained. All 300 exclusions have an inspectable source-folder/instance index
in `data/processed/multidataset/miracl_unresolved_index.json`.

MIRACL speakers are nonconsecutive IDs, not invented F01–F10/M01–M05 ranges:

| Split | Speakers | Full words | Full phrases | Pilot words used |
| --- | --- | ---: | ---: | ---: |
| Train | F01, F02, F04, F05, F06, F07, M01, M02, M04 | 720 | 900 | 140 |
| Validation | F08, F09, M07 | 240 | 300 | 48 |
| Test | F10, F11, M08 | 240 | 300 | 48 |

Each full word class has 90/30/30 train/validation/test instances. The bounded
pilot prespecified repetitions 01–02: 144/48/48 clips before tracking. Its first
preparation failed safely: four training clips did not meet measured-landmark
coverage. The second run explicitly excluded those training clips, with no
replacements and no held-out exclusions: `F04/words/08/01`, `F04/words/08/02`,
`M01/words/04/02`, `M01/words/09/01`. They remain in the original data. Full
RGB validation does not imply that all 2,700 sequences pass mouth tracking.

Generated manifests remain separate from all publisher/GRID originals:

- `data/processed/multidataset/miracl_v1.jsonl`: 2,700 word/phrase rows;
  SHA-256 `4c6150dcd58736b9620807c9d00df2ff94b410dda3413ec84b614b8053152352`.
- `miracl_v1.inventory.json`: task/split/vocabulary/speaker counts and protocol.
- `miracl_pilot_sources_v1.jsonl`: 240 prespecified RGB inputs;
  SHA-256 `8a7d1cdc492acd7ab09cf2712781172355167ce97e3f8c93810e66ba0abc7be0`.
- `runs/multidataset/miracl8_cpu_v2/selected.jsonl`: 236 tracked inputs;
  SHA-256 `9452975343e198fee30eed835b4d796ae0830329caa45d3cfab4b4091b8f2346`.
- `data/processed/multidataset/grid_commands_v1.jsonl`: 160 real GRID words,
  96/32/32 clips, 23/5/5 speakers. Train: s1–s20, s22–s24; validation: s25–s29;
  test: s30–s34. Words: bin, lay, place, set.
- `grid_words_avhubert_v1.jsonl`: same 160 annotation-bounded words exported for
  the existing supervised frozen-encoder workflow. SHA-256:
  `5d099e1fc9650c020898664d6f7ce31515c93ceb5d752e0a7eee64ab32b70c5d`.

Normalized validation enforces task labels, vocabulary coverage, IDs, speaker and
source-content separation, frame bounds and authentic provenance. Speaker IDs
are namespaced by corpus; cross-corpus human identity is not assumed disjoint.
LRW retains publisher partitions and explicitly unknown speaker independence.
Phrase and sentence transcripts never become single word labels. Unreviewed
timing, missing identities and audio inputs fail rather than becoming training
examples. No compatible word vocabulary overlaps this MIRACL pilot and the
four-word GRID baseline, so a numerical cross-dataset comparison was not invented.

## Actual model capabilities

The primary audit snapshot enumerated **12 `.pt` checkpoints**, including the two newly
trained MIRACL files. Eleven small classifier artifacts strictly loaded into
their corresponding architecture. The large encoder was fingerprinted and its
previous real CPU extraction evidence retained; loading it was not repeated on
the low-memory machine. Full path/hash/state/output/vocabulary details and all
500 actual vocabulary entries are in `.qa/multidataset-final-audit.json`.

Late shared-workspace update: a separate process running
`scripts/_test_improved_architecture.py` added two baseline preservation copies,
`runs/word_recognition_v1/best_collapsed_baseline.pt` and
`runs/word_recognition_v1/checkpoints/best_word_model_collapsed_baseline.pt`.
The inventory consequently reached **14 checkpoints**. Their read-only
inspection is recorded separately in `.qa/multidataset-late-checkpoints.json`;
no accuracy is inferred from them. At the late check all twelve artifacts from
the primary audit still matched their fingerprints, and the evaluated MIRACL
model definition still matched its recorded source hash. This independent
experiment was neither launched nor interrupted by this task, and its eventual
results are outside this report.

| Mode | Architecture / output | Checkpoints | Endpoint / execution | Evidence and limits |
| --- | --- | --- | --- | --- |
| Existing alphabet | Per-frame Conv2D + BiLSTM + final clip classifier, 26 outputs; `[B,T,1,140,46]`, original T=30 | `models/alphabet_classifier.pt` | Existing `/api/predict` | A–Z isolated-letter classification. No CTC sentence decoder; no new alphabet accuracy result. Backend remains alphabet. |
| Legacy LRW-shaped | Same CNN-BiLSTM, 500 outputs; legacy T=29 | Three `.pt` files in `runs/finetune/lrw_500words` | Research code; no verified 500-word live deployment | Strict loading proves architecture compatibility, not authentic LRW training, mapping or accuracy. `data/vocab.txt` has 500 unique nonempty entries. |
| GRID word experiments | Conv3D + two-layer BiGRU, 4 outputs | `runs/word_recognition/grid_commands_cpu_20261029/{best,last}.pt`; three files under `runs/word_recognition_v1` | Existing experimental `/api/predict/word` uses the latter configured path | Original four-word baseline test: 8/32 (25%), single-class collapse. Kept separate from MIRACL and not used to claim broad vocabulary performance. |
| MIRACL word pilot | Conv3D(8,16,32) + BiGRU hidden48/layer2 + temporal mean + 8 outputs, **84,048 parameters**; `[B,1,24,64,64]` | `runs/multidataset/miracl8_cpu_v2/{best,last}.pt` | Offline `pipeline.multidataset.predict` | Genuine RGB training and held-out evaluation; weak results below. No live activation. |
| AV-HuBERT representation | Official Base encoder, 768-dimensional temporal features | `models/external/avhubert/base_vox_iter5.pt` | Existing `AVHubertLoader` / classifier CLI | Encoder available; no trained new word head or sentence decoder. Existing verified preprocessing remains unchanged. |
| Continuous video-to-text | CTC utilities / upstream trained-decoder implementations exist | No verified fine-tuned visual-only sentence checkpoint | Unavailable | MuAViC demo explicitly enables audio+video. Its AVSR checkpoint listing does not prove video-only support. No sequence recognition accuracy/WER/CER is claimed. |

A critical audit finding was corrected: the existing
`pipeline/sequence_recognition/ctc_decoder.py` evaluation used the constant
sentence `bin blue at f two now`, while `/api/models` reported that mode ready.
That path now raises an explicit unavailable error; readiness is false. Any
earlier report generated from that constant is **not recognition evidence**.
The new sequence scorer requires actual held-out decoder hypotheses and records
checkpoint provenance, WER/CER and speaker/dataset breakdowns. It is a metric
utility, not a trained recognizer, and does not independently authenticate how
prediction files were generated.

## Executed pilot and measured comparison

Initial hardware: Intel i3-1115G4, 2 physical/4 logical cores, 7.74 GiB RAM,
no CUDA, approximately 307.37 GiB free disk. Available RAM after importing Torch
was 0.55 GiB initially and 0.90 GiB at the later preflight. Consequently the
1.24 GB AV-HuBERT checkpoint was not loaded for a new experiment. No large
sentence model or full-corpus training was attempted. The small scratch model
used CPU PyTorch 2.2.2, one thread, batch two, no loader subprocesses, seed 42.

Measured eye-axis alignment creates 96px mouth crops, gray 64px inputs and
linear interpolation to 24 frames. MIRACL native FPS is unknown: ordered frame
indices are used, without invented timestamps. MediaPipe IMAGE mode avoids
fabricating frame times. Training-only coherent crop/brightness/contrast/flip
and frame-drop augmentations do not enter validation/test. Caches verify source
bytes, bounds, landmarker identity, preprocessing and pixel artifacts.

AdamW, smoothed cross entropy 0.1, gradient clipping, ReduceLROnPlateau, atomic
best/last checkpoints and early stopping are implemented. All parameter gradients
were finite in the real two-input smoke. The 12-epoch run took **58.44 seconds
of training-loop time**, after **230.61 seconds** of source/cache verification
and preparation in run v2. The first preparation took 99.98 seconds. These are
one-machine observations, not real-time guarantees. Best checkpoint: **epoch 11
by validation loss only**; the test partition was evaluated after selection.

| Same 48 held-out clips | Top-1 | Top-5 | Macro F1 | Single-word WER |
| --- | ---: | ---: | ---: | ---: |
| Training-majority baseline (`choose`, alphabetical tie break) | 6/48 = 12.5% | Not evaluated | Not evaluated | Not evaluated |
| Selected eight-word model | 8/48 = 16.67% | 30/48 = 62.5% | 0.08617 | 83.33% |

Top-5 equals the random expectation for eight balanced classes; two extra correct
top-1 predictions on this tiny sample do not establish a reliable improvement.
Six of eight classes have zero recall. Correct/total: choose 6/6, connection 0/6,
hello 0/6, navigation 0/6, next 0/6, previous 0/6, start 2/6, stop 0/6.
Held-out speakers: F10 2/16, F11 2/16, M08 4/16. Mean maximum softmax confidence
was 0.14230 (range 0.13130–0.14957). All 48 predictions were rejected by the
prespecified confidence>=0.70 and margin>=0.15 display policy. These thresholds
are uncalibrated, and reliable out-of-vocabulary detection remains unverified.

Fresh source-frame inference on `miracl:F10:words:03:01` (true `connection`)
tracked all 11 frames, strictly loaded the selected head and produced finite
eight-class probabilities. Its incorrect `choose` candidate had confidence
0.14893 and was rejected: `word:null, uncertain:true`. Exit code 0 means the
pipeline ran, not that it recognized the word. Proof:
`.qa/miracl-real-inference.json`. No audio is read by this path.

Selected checkpoint SHA-256:
`3c7c420d30ac111d4b752322318d92c3b9dfa2077ba2f3d04bc6a71b6aa937ed`.
Artifacts: `training.json`, `smoke.json`, `preparation.json`, `evaluation.json`,
`test_predictions.jsonl`, `confusion.csv`, and interactive offline
`confusion.html` inside `runs/multidataset/miracl8_cpu_v2`.

## Integration, repairs and validation

`GET /api/recognition/capabilities` and a small live-workspace research card expose
the actual offline vocabulary/evaluation and unavailable sequence status. No
new mode selector or model activation was added. The existing landing, Science
Lab, live camera pipeline and palette are preserved. Existing word health now
explicitly says research/uncalibrated, without claiming production readiness.
No application backend was restarted and no model-selection environment variable
was changed. Only a temporary preview server on port 4187 was started for
browser checks; it and the owned headless browser were stopped afterward.

Wider regression checks exposed five existing GRID-word workflow failures.
Repairs distinguish integer class rankings from float probabilities, preserve
completed runs, restore optimizer/scheduler/Python/NumPy/Torch/loader RNG state,
enforce timeouts before a batch and save checkpoints atomically. They add strict
dataset/preprocessing checks, finite-gradient clipping, a bounded 128-clip uint8
LRU cache, training-only augmentations, functional diagnostic CLI and `--resume`.
The active `pipeline/word_recognition/run_all.ps1` now forwards to the canonical
isolated-environment runner. These repairs ran against temporary test fixtures;
existing real checkpoints were not rewritten or reevaluated to replace history.

- Full final application regression: **467 passed, 4 skipped**. Three skips
  require optional synthetic fixtures; one skips the separate requests-based
  acquisition suite in the regular app environment.
- Isolated new acquisition/adapter/model suite: **27 passed**.
- Earlier targeted backend/data checks: **80 passed**; repaired GRID word suite:
  **22 passed**. These are subsets, not additional independent accuracy evidence.
- Frontend live tests: **10 passed**; math tests: **26 passed**.
- Production TypeScript/Vite build passed.
- Browser: **7 research-card checks** (actual capability JSON via interception,
  desktop/mobile, offline failure and unmount) and **4 interactive confusion-matrix
  checks** passed. No uncaught JavaScript errors. This is not a deployed backend
  or physical webcam recognition test.
- Twelve preservation fingerprints matched: original five checkpoints, original
  s1 manifest/three archives, both previous GRID word checkpoints, and the frozen
  combined GRID letter manifest. The primary twelve checkpoints and two late
  additions are enumerated in the primary and late audit JSON files above.

Logs: `.qa/multidataset-full-regression-final.txt`,
`.qa/multidataset-tests-final.txt`, `.qa/multidataset-frontend-build.txt`,
`.qa/multidataset-live-tests.txt`, `.qa/multidataset-math-tests.txt`,
`.qa/multidataset-live-browser-check.json`, `.qa/miracl-matrix-browser-check.json`.
Initial failed runs remain inspectable rather than being erased.

## Exact inspection and execution record

Key inspected files: `data/vocab.txt`, `models/alphabet_model.py`,
`app/main.py`, `app/services/browser_inference.py`, `word_inference.py`,
`inference_service.py`, `pipeline/avhubert/loader.py`, `classification.py`,
`pipeline/lrw/whole_word.py`, `scripts/train_lrw_500words.py`,
`scripts/preprocess_lrw_dataset.py`, `scripts/acquire_lrw.py`,
`scripts/lrw_words.py`, `scripts/prepare_grid_dataset.py`,
`tracking/mouth_canonicalization.py`, all modules under
`pipeline/word_recognition`, `pipeline/sequence_recognition/ctc_decoder.py`,
`pipeline/adapters/{base_adapter,grid_word_adapter,miracl_adapter,external_adapters}.py`,
`frontend/src/{lib/api.ts,hooks/useBackend.tsx,live-demo/useDemoPipeline.ts,workspace/pages/LiveDemo.tsx,workspace/api.ts,router.tsx}`,
`frontend/package.json`, `frontend/vite.config.ts`, `pyproject.toml`, `.gitignore`,
the GRID acquisition/inventory/video/word/letter manifests and existing audit
reports. Exact weight paths/metadata are in the final machine audit.

MuAViC reference: `external/muavic_reference` at commit
`12f7f09c8b28f1af4285c1e4f2c66f6c82c86af6`;
`README.md`, `LICENSE`, `requirements.txt`, `get_data.py`, `lrs3_utils.py`,
`utils.py`, `demo/demo_utils.py`, `demo/requirements.txt` inspected.
The upstream English manual-LRS3 requirement and demo audio+video configuration
block any claim of verified silent-video transcription.

Executed commands are listed in `MULTIDATASET_COMMANDS.md`. Future access/setup,
native adapter schemas and separate AV-HuBERT word commands are documented in
`pipeline/multidataset/README.md` and `DATASET_ACQUISITION_PLAN.md`.

Remaining blockers: ambiguous MIRACL labels/FPS; much stronger word learning and
calibrated webcam/open-set evaluation; RAM for larger pretrained experiments;
authorized LRW/TCD/LRS3 inputs and Roboflow credentials; a compatible trained
visual-only sentence decoder/tokenizer plus real held-out transcript evaluation.
No result here establishes arbitrary-English recognition or production readiness.
