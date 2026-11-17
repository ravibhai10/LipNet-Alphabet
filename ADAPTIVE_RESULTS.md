# Adaptive feedback implementation results

2026-11-16, Asia/Calcutta. [Architecture/candidate audit](ADAPTIVE_ARCHITECTURE.md) ·
[setup, API and consent guide](ADAPTIVE_GUIDE.md) · [preserved reliability results](RELIABILITY_RESULTS.md).
Evidence directory: `.qa/adaptive_20261116/`.

The new `/demo/feedback` lab connects bounded video capture, independent tracking, real task-specific
inference, explicit human decisions, private review, immutable dataset versions and controlled batch
training. Existing alphabet and eight-word checkpoints remain unchanged. No newly collected human
feedback was available to train a real challenger, and no model was promoted. No recognition accuracy
improvement, complete A–Z reliability, OOV detector or physical webcam accuracy is claimed.

## Implemented behavior

- The live lab captures a complete bounded video-only WebM/MP4 recording alongside lossless full-frame
  PNG samples and actual sample timestamps. Native video and source frames can be replayed before
  prediction; saved sources have private authenticated playback and inspection. Every original video
  frame is decoded and checked for bounded length/geometry; independent MediaPipe tracking validates
  the sampled PNGs. Source bytes, decoded pixels, preprocessing, ROI and model identities are hashed.
- Prediction output is generated on the server and bound to its source. Confirmation requires an
  accepted prediction; correction requires an explicit task-valid label or UNKNOWN. The original
  prediction remains in history. Repeated requests/clicks are idempotent. Uncertainty is retained.
- Review decisions and reasons form an append-only event chain. Labels are not silently rewritten.
  Pending, approved, rejected, duplicate, quarantined, withdrawn and included states remain distinct.
  UNKNOWN clips can be approved for independent negative evaluation, never classifier training.
- Dataset Feedback Studio supplies owner/reviewer scopes, filters, source timelines, counters,
  distribution/growth summaries, hard-example suggestions and immutable manifest versions. Global
  speaker assignments prevent cross-session and cross-version partition leakage. Prior held-out
  speakers are quarantined. Final-test access seals both record IDs and speaker IDs across versions.
- Word retraining fits a new logistic head on frozen visual features, with training-only scaling.
  Alphabet retraining retains the CNN-BiLSTM architecture and requires all 26 classes. Both adapters
  use explicit train/validation partitions and new artifact directories, then paired validation.
  Existing approved training/validation records can contribute to growing dataset versions; exposed
  final-test speakers cannot become a fresh test population by changing recording sessions.
- Saved reports include top-1/top-5, macro precision/recall/F1, per-class metrics, confusion matrix,
  accepted accuracy, coverage/rejection, negative false acceptance where data exists, ECE/Brier and
  risk/coverage. The dashboard renders measured class tables, a confusion matrix and risk/coverage
  curve when available, with downloadable run/dataset-bound evidence. Missing results remain null.
- Training is a bounded background job. Final held-out evaluation is a separate one-time operation;
  it does not select the challenger. Promotion requires saved validation/resource/integration and
  regression evidence, complete negative evaluation and explicit reviewer approval. The adaptive
  champion pointer supports rollback. Historical inference routes retain their original selection.
- Consent, separate collector/reviewer keys, protected media routes, withdrawal, scheduled-by-operator
  retention maintenance and source/label audit history are implemented. Locking clears local access,
  media previews, consent and speaker/session context. Derived trained artifacts require separate
  review after withdrawal; model unlearning and backup erasure are not claimed.
- A step-by-step examiner workflow supports real camera and explicitly labelled recorded replay.
  No application prediction fixtures or simulated training progress were added. Existing live-demo
  edits were preserved; unsupported SOTA/OOV/instant-adaptation claims and assumed frame timing in
  the concurrent examiner/frame-inspection components were corrected.

## Actual model probes and interpretation

`real_adapter_probe.json` uses the genuine MIRACL validation recording `miracl:F08:words:09:02`,
publisher label HELLO, with ten original source frames. Native FPS remains unknown. The 100ms spacing
is an explicit replay schedule, not a claim about the source camera. Independent tracking completed
with no recorded lighting/motion warnings. No replay became a user-feedback dataset record.

| Probe | Actual result | Scope |
|---|---|---|
| Frozen eight-word adapter, three calls | HELLO candidate and accepted output in all three; mean reported latency 700.49ms | Repeated same historical clip, warm inference; not three independent samples |
| Alphabet adapter, three calls | Mean reported latency 74.45ms | Word ROI supplied to alphabet model; runtime/shape check only, no letter-accuracy interpretation |
| Word adapter load | 11,892.18ms | Single load measurement under the current process/CPU conditions |
| Real feedback API including native video | 14 checks passed; actual HELLO response, matching checkpoint hash, byte-identical video retrieval, idempotency and withdrawal | Disposable private database; automated QA decision, no retained or human-verified training samples |

The native-video ASGI probe's prediction field reports 24,393.25ms for registry/model readiness plus
inference while other CPU checks were running. Source validation precedes that timer, so it is not
the complete capture-to-display latency. The warm-adapter times exclude model loading and independent
source tracking (4,866.69ms in the first probe). These measurements have different scopes from the
earlier eight-clip ASGI benchmark; no speed improvement is inferred.

The preserved baseline's historical validation remains 228/240 and historical test 182/192. These
are prior results under the documented protocol, including prior test-speaker exposure. The existing
temporal candidates remain below that validation baseline. Fresh file hashes/sizes are recorded in
`final_audit/artifact_audit.json`; no new pretrained model was downloaded or merged. See the candidate
audit for input contracts, licensing evidence and the unavailable fair-comparison data.

## Verification evidence

The final Python regression command, including nested evaluation/GRID/AV-HuBERT tests, passed
**694 tests, 3 skipped** (`verified_tests.xml`, `verified_tests.txt`). The skips are the existing
optional synthetic-dataset fixture cases. Fifty new adaptive tests cover source/clip validation,
consent, confirmation/correction, task labels, duplicates, ownership, withdrawal, audit chaining,
immutable manifests, speaker leakage, final-test sealing, job failure/capacity, checkpoint mapping,
challenger rejection, rollback and a genuine logistic-fitting code path on explicit synthetic features.
Synthetic training fixtures were confined to temporary test directories and supply no speech metrics.

Frontend checks passed: **41 existing live tests, 26 math tests and 6 adaptive logic tests**.
TypeScript/Vite production build passed. Results are preserved as `verified_frontend_*.txt`.
The new lab passed **31 browser checks** (`browser_checks.json`). A separate retained live-demo
regression run passed **92 browser checks** (`legacy_browser/live-assistant-browser.json`).
These explicitly distinguish software API fixtures
from real MediaPipe and MediaRecorder on the genuine prerecorded AFTERNOON camera source. They cover
feedback, explicit correction, duplicate clicks, review, source preview/timeline, native clip capture,
chronological PNG/video submission, withdrawal, model views, walkthrough, mobile layout and access
locking, including clearing consent and identity before reconnecting with another key. The legacy run
also covers camera denial/unavailability/busy/disconnection, stale/failed/timed-out responses, retry,
navigation cleanup, the separate 500-word contract and twelve mobile routes. They are not physical
webcam recognition evidence.

The unchanged historical browser script initially failed because its former history-feedback control
was removed by concurrent UI edits. Its uncertain-confirmation assertion was also incompatible with
the required safety contract. The archived script/results were preserved. The task-local copy replaces
that assertion with checks that uncertain candidates cannot be confirmed and corrections require an
explicit label with UNKNOWN available; all remaining checks continue to run. The initial failure is
retained in `legacy_browser/live-assistant-browser-failure.json`.

An earlier capture under simultaneous regression/encoder CPU load failed the strict frame-gap gate.
The failure was retained as `browser_capture_gap_under_load.json`. Repeating the same checks after the
CPU jobs finished passed without weakening quality thresholds. The ordinary word environment's
MediaPipe import was blocked by Windows Application Control; tracking and real inference ran in
the existing encoder environment. Its optional HTTP test-client dependencies were unavailable, so
the real API probe drives standard ASGI directly without installing or replacing encoder packages.
The initial missing-client log is retained. No Windows security policy was changed.

Historical integrity checking verified **3,316 artifacts; 0 changed, 0 missing** (`final_integrity/artifact_audit.json`). Earlier reports,
evaluations, policies, datasets and model files were retained. The repository received concurrent
edits/commits during the work; no bulk reset or replacement of that work was performed.

## Limits and required next evidence

The implementation supplies controlled learning infrastructure, not a newly proven improved model.
Genuine consented feedback, stable reviewer-verified identities, complete A–Z data including W,
adequate class/speaker diversity, independent negatives and fresh held-out participants remain
necessary. Hard-example suggestions do not determine whether a label is correct. Mouth occlusion,
device identity and audio absence are not independently attested. Original imported file containers
are replayed into bounded video-only clips rather than uploaded in full. Legacy frame-only imports
remain supported and explicitly identify absent native clips.

This is a single-process local research deployment. Disk encryption, production accounts, distributed
workers, crash recovery, operator backup deletion and model unlearning need separate deployment work.
Retention runs when the operator invokes maintenance; it is not an automatic daemon. Continuous
sentences still have no validated decoder. New challenger accuracy, physical webcam acceptance and
population-level calibration must be measured with genuine reviewed data and hardware before any
stronger capability claim is made.
