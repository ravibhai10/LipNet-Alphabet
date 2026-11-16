# Reliability audit — 2026-11-09 (Asia/Calcutta)

This audit precedes changes for the attached accuracy/realtime request. Existing
working-tree edits belong to the user and are preserved. Evidence and rollback
files are in `.qa/reliability_20261109/`; final results are recorded in
`RELIABILITY_RESULTS.md`. Historical measurements are not new measurements.

## Architecture and active paths

The actual repository is the nested `LipNet-Alphabet-main` directory. React/Vite
uses `useWebcam` → MediaPipe `useLandmarker` → eye-axis-aligned 96px mouth JPEGs
→ bounded temporal capture → HTTP `/api/predict` or `/api/predict/word`.
The websocket is lifecycle-only and does not run classification. LiveDemo has
both automatic windows and a manual WordAssistant capture workflow. Browser
camera permission is required; no physical camera test has been performed.

The default alphabet adapter is `BrowserInference`: protected
`models/alphabet_classifier.pt`, frame CNN (16/32/64 channels), one 96-unit
bidirectional LSTM, 128-unit head, 26 logits. Input is 30×1×140×46, [0,1].
AdamW, smoothed cross entropy, coherent augmentation, warmup/cosine, and early
stopping are already implemented by `scripts/train_alphabet.py`. Taking the
last BiLSTM output underuses the backward sequence; changing this would change
checkpoint semantics and requires retraining. The unusually tall crop and
aggressive early pooling are further hypotheses, not proven diagnoses.

Default word service is the legacy `WordVideoNet` at
`runs/word_recognition_v1/checkpoints/best_word_model.pt`. It has a 3D CNN,
two-layer BiGRU, temporal mean pooling, and checkpoint vocabulary. The mutable
model definition now includes input instance normalization; the preserved
`legacy_word_model.py` does not. Identical state shapes cannot detect this
semantic incompatibility. Do not silently reinterpret historical weights.

`LIPNET_WORD_BACKEND=avhubert` selects the separate frozen visual AV-HuBERT
encoder and measured eight-word logistic head. Confirmed in code: layer 9,
29 frames, 768 dimensions per frame, mean+std = 1,536 pooled features,
training-only scaler, eight-way linear classifier. 96px crops become 88px,
mean .421/std .165. This is frozen transfer learning, not encoder fine-tuning.
`start_word_demo.ps1` selects it. Separate expanded 20-word and publisher
500-word DC-TCN routes must retain their own preprocessing and evidence.

The experimental three-stream Conformer/CTC pipeline in `live_pipeline.py`
is a separate path requiring compatible trained weights. It is not the model
behind the browser alphabet endpoint. Continuous sentence decoding has no
verified checkpoint. BiLSTM/AV-HuBERT windows require future context within a
completed input window and do not provide zero-lookahead streaming.

## Dataset inventory and evaluation

Directly counted manifests (not dashboard totals):

| Source | Samples | Speakers | Labels | Qualification |
|---|---:|---:|---:|---|
| MIRACL isolated words | 1,200 | 15 | 8 | 150/class; train/valid/test 720/240/240 |
| MIRACL phrases | 1,500 | 15 | 10 | Separate phrase diagnostic; never word training labels |
| GRID letter slots | 32,891 | 33 | 25 | W absent; coarticulated sentence letter slots |
| Processed LRW manifest | 45 | Unverified | Inspect provenance | Development fixtures, not eligible LRW benchmark |

Eight words: choose, connection, hello, navigation, next, previous, start, stop.
HELP, I, AM lack isolated-word examples in this eight-word corpus. Phrase
occurrences of I/AM do not constitute aligned isolated-word labels. Other
corpus vocabularies do not silently expand this head. Actual class/speaker
counts are saved in `manifest_inventory.json`. GRID labels range from
1,311 to 1,320 samples/class. Prior GRID inventory reports 109 invalid videos
among 33,000 sources; this is prior evidence, pending fresh decode checks.

Historical alphabet `training_summary.json` records .12 test accuracy on 100
examples, .20 best validation accuracy. Its original processed dataset and
speaker-independent provenance are not established; it is not live accuracy.
Existing eight-word evaluation records 182/192 correct, macro F1 .94668,
top-5 191/192. Those artifacts predate this work. Test speakers were exposed
in an earlier pilot, with repetitions 01–02 quarantined. Do not describe that
as an untouched new speaker population. Rejection leaves 137/192 known clips
accepted and falsely accepts 8/60 held-out phrases; unknown isolated-word
rejection is unvalidated. Fresh latency measurements are pending.

## Findings and priorities

1. Personal-training endpoint invents zero-weight classes for unrecorded words,
   overwrites `personal_words/head.pt` and reports an evaluation without actual
   test results. It attempts to mutate the live service after this overwrite.
   Startup also silently prefers that file over an explicitly selected head.
   Replace this with validated, immutable recording export and the existing
   separate training/validation/test workflow; preserve active weights.
2. Pretrained word service sets `word` to the candidate even when rejected.
   Keep candidate diagnostics but return null as the accepted prediction.
3. Python stabilizer accepts the first window immediately and returns the old
   label on switches. Browser alphabet stabilization lacks score rejection;
   its UI even displays an unconfirmed candidate as detected. Enforce actual
   consecutive evidence and clear rejection state.
4. Legacy word input normalization changed since its saved baseline. Freeze
   architecture semantics for original checkpoint loading; never deploy an
   experimental normalization change without comparative evidence.
5. JPEG/body bounds, output schemas, stage timings, vocabulary/model identity,
   and API errors need consistent validation. Models are lazy singletons;
   avoid repeated model construction and avoid unbounded request queues.
6. Complete A–Z accuracy optimization is blocked by missing W and missing
   audited isolated-alphabet recordings. Retain 26 outputs without claiming
   the missing class is trained. Do not merge MIRACL words into letters.
7. Compare a small temporal BiLSTM head against the frozen pooled baseline
   using training and validation caches only. Keep historical test artifacts
   sealed during selection; no promotion based on validation alone.

## Risk controls and baseline

Artifact hashes were recorded before changes; the protected alphabet SHA-256
is `7c00d5117b7fae9a76c05980ca70edd680e79be4c8a152ea2595c98ded90c4c3`.
Source and primary checkpoints are backed up. Large datasets and publisher
weights remain in place and are fingerprinted. Never restore the source
archive over unrelated user work without reviewing individual changes.

Baseline pytest in `.venv` fails collection: Windows Application Control
blocks scikit-learn `_radius_neighbors_classmode` DLL. The failure log/XML
are retained; the dedicated word environment is being tested separately.
Tests must not be deleted or skipped just to conceal this failure.

Plan order: record baseline → audit storage/quality/leakage → fix safety and
preprocessing contracts → run controlled cached temporal experiment → measure
real CPU/API inference → run Python/frontend checks → publish reproduction,
comparison, limitations, and artifact-integrity results. No accuracy or
latency improvement is claimed until measured.
