# Adaptive lab setup, API and data policy

Run commands from the nested repository containing `app/main.py`.

```powershell
npm --prefix frontend run build
.\start_adaptive_demo.ps1 -Port 8002
```

Open `http://127.0.0.1:8002/demo/feedback`. The launcher selects the preserved alphabet and
eight-word backends and uses the existing `.venv-avhubert-cpu`. It prints separate random
collector/reviewer keys for the local session. Enter a key in the page. Keep the reviewer key
private. The launcher binds localhost, restores environment settings on exit, and does not stop
other servers. Use an unused port. `-BuildFrontend` can build before launch.

For persistent ownership across restarts, configure different random keys of at least 32 characters
as `LIPNET_FEEDBACK_TOKEN` and `LIPNET_REVIEW_TOKEN` in your private shell before launch.
`LIPNET_FEEDBACK_TOKENS` optionally accepts comma-separated collector keys for separate owners.
Use a distinct key per person; sharing a collector key shares its recording ownership. Losing
the key loses collector access; a reviewer can still handle withdrawal. Keys must not be committed
or pasted into issues. Keys are sent in Authorization headers and held in component memory;
the frontend does not put them into localStorage, URLs or sample metadata. Reviewer/collector
credentials identify audit actors through SHA-256 fingerprints, not claimed display names.

Use one backend process/worker for this local research app. Production authentication, account
recovery, encrypted storage, quotas and distributed job coordination are not supplied by this demo.
Do not expose the legacy research server or the dataset filesystem publicly. The private adaptive
API defaults to disabled until keys are configured; its media cannot be downloaded through the
legacy video route. Historical research/data routes retain their existing access model.

## Recording, verification and demonstration

1. Connect with a collector key. Start the camera after browser permission, or select a genuine
   recorded video. Replay remains labelled. The application performs real model inference.
2. Select alphabet or words, supply a stable consented pseudonymous speaker ID and a fresh session
   ID, and accept the recording/training consent statement. Never give the same person a new ID
   just to obtain another dataset partition.
3. Use a browser with MediaRecorder support, such as the installed Chrome. Start before lip movement and finish after the complete utterance. Capture is bounded to five
   seconds and 76 frames at the API; the browser stops at about 50 samples or 4.8 seconds.
   The source must be 96px–1280×720 at its native captured resolution; it is not silently resized.
4. Inspect the original bounded video and frame timeline/replay. Click **Predict reviewed recording** to retain the
   consented source and run server validation plus inference. The camera preview can continue.
5. Confirm an accepted result or choose the actual correction. Uncertain candidates cannot be
   confirmed. UNKNOWN / OTHER is explicit; HELP/I/AM may be collected but do not expand the head.
6. Saved feedback enters review, not training. Repeated submission is idempotent. Retry starts a
   new utterance. **Undo / withdraw recording** removes its source and derived ROI media.

A reviewer uses Dataset Feedback Studio to inspect prediction versus verified label, source frames,
timestamps, task, speaker/session, status and model version. Review needs a written reason.
Passing known-label records can be approved for training; reviewed UNKNOWN records are negative
evaluation only. Low light/contrast or low motion need a fresh recording rather than a quality
bypass. Duplicate content cannot be approved. Filters cover task, status and text searches across
labels, identities, dates and model versions. Hard examples remain review suggestions with no
automatic training weight.

The **Examiner walkthrough** tab explains capture, actual inference, feedback, queue changes,
review, dataset versioning, batch fitting, comparison and final approval. It does not simulate
training progress or prediction success. Camera denial/unavailability can be demonstrated directly;
for a repeatable model demonstration, use a genuine local video and label it replay.

## Dataset versions and training

In **Models & training**, reviewers provide explicit speaker assignments as JSON:

```json
{"speaker_train":"train", "speaker_validation":"valid", "speaker_heldout":"test"}
```

Every configured class needs at least two approved utterances in each partition; within-partition
class imbalance cannot exceed 5:1. These are structural research checks, not a claim that so little
data is sufficient. Existing historical validation/test speaker IDs from the MIRACL and GRID
manifests are prohibited. Aliased identities cannot be detected automatically. Keep genuinely new
evaluation speakers and freeze the protocol before selection. A speaker's assignment persists across
all versions and sessions. Negatives are never assigned to training.

Freeze the dataset, select its version and request a background job. Exact eight-word labels are
required for the word adapter; all 26 letters including W are required for alphabet fitting.
The server fits in a bounded background worker and writes a new `runs/adaptive/<job-id>/` directory.
It never overwrites the production checkpoint. Status and comparison reports contain actual results.
The fitted artifact, configuration, dataset receipt, histories, feature receipts and validation
comparison are preserved. One training attempt is allowed per version, including failed attempts.

Validation compares champion and challenger on identical examples and saves macro precision/recall/F1,
per-class metrics, confusion matrix, top-1/top-5, coverage, accepted accuracy, rejection, ECE, Brier,
risk/coverage and resource scopes. Reviewer-approved UNKNOWN validation clips measure false accepts.
No independent negative evidence means promotion is blocked. Low confidence alone is not OOV detection.

Only after selection is frozen should a reviewer request the one-time final evaluation. It saves
held-out results and runs the full regression command below in the verified word-test environment.
It is a background operation; refresh model evidence for its actual outcome. Failed final evaluations
retain their access receipt and cannot be silently retried after retuning. Comparison and test results
remain separate. Deployment requires the saved gates and a distinct explicit reviewer action.
Rollback changes only the adaptive champion pointer to an intact, eligible previous artifact.
Withdrawn lineage blocks new training/inference on that candidate and needs review of derived models.

## API

Generated request/response schemas are at `/docs` and `/openapi.json`. Every route below requires
`Authorization: Bearer <key>`. Management actions additionally require a reviewer key.

| Method / path under `/api/adaptive` | Purpose |
|---|---|
| GET `/config` | Task labels, correction labels, consent text, role |
| POST `/captures` | Retain bounded original PNG sequence, independently validate, run actual model |
| POST `/captures/{id}/predict` | Retry unavailable/busy inference on the same owned source |
| POST `/captures/{id}/feedback` | `{verdict: correct|incorrect, label: actual_label|UNKNOWN}` |
| POST `/captures/{id}/withdraw` | Withdraw consent and delete stored source/ROI BLOBs |
| GET `/stats`, `/records`, `/hard-examples` | Owner-scoped data; reviewers see the review queue |
| GET `/records/{id}` | Metadata and immutable audit-event history |
| GET `/records/{id}/frames/{index}` | Private original PNG, `Cache-Control: no-store` |
| GET `/records/{id}/source` | Private ordered lossless-source ZIP |
| GET `/records/{id}/original-video` | Private original bounded WebM/MP4 bytes, where present |
| POST `/records/{id}/review` | Decision and reason; reviewer only |
| POST `/datasets` | Task and speaker assignments → immutable manifest |
| GET `/datasets/{id}` | Manifest and hash, with withdrawal validation |
| POST `/training/jobs` | Dataset version → HTTP 202 background batch job |
| GET `/training/jobs/{id}` | Actual persisted job state and sanitized failure |
| GET `/models`, `/models/{id}/comparison` | Registry, champion pointers and comparison/evaluation evidence |
| POST `/models/{id}/final-evaluation` | Separate one-time held-out evaluation, HTTP 202 |
| POST `/models/{id}/deploy` | Explicit `{approved: true}` plus all saved gates |
| POST `/models/{task}/rollback` | Restore previous adaptive champion |
| POST `/retention/expire` | Remove uncommitted sources older than 30 days |

Capture body fields: `id`, `task`, `speaker_id`, `session_id`, timezone-aware `recorded_at`,
`capture_kind` (`physical_webcam` or `webcam_replay`), explicit `consent`, ordered original
`data:image/png;base64,...` frames, matching `timestamps_ms`, and `original_video` as a bounded
`data:video/webm;base64,...` or `data:video/mp4;base64,...` recording. The browser retains native
video plus lossless sampled frames; frame-only API imports remain supported and explicitly marked.
Native clips must fully decode at matching geometry within 5.8 seconds/180 frames and 18 MiB.
Client predictions are forbidden
extra fields: only actual server output is linked to the record. A recording ID is its idempotency
key; replaying the ID with different content is rejected. Feedback requires its actual accepted
prediction for confirmation or an independently supplied correction.

Status codes: 201 capture/version retained; 202 batch/final job accepted; 401 missing/invalid key;
403 wrong role; 404 inaccessible/removed record; 409 final evidence already exists; 413 oversized
body; 422 invalid source/label/protocol; 429 busy; 503 unavailable configuration/model; 504 bounded
capture processing deadline. A model failure after source retention returns the retained ID and
a sanitized retry notice. Native processing retains its lock after a timeout until it ends.

## Privacy and retention

`data/adaptive_private/feedback.sqlite3` stores private source ZIPs (original video, PNG frames and timestamps) and ROI arrays as BLOBs, separate
from label/audit history. This directory is Git-ignored. Source PNG bytes remain immutable while
retained; audit events form a SHA-256 chain. This detects accidental/history changes but is not
tamper-proof against someone who controls the database and filesystem.

Statuses are PENDING_REVIEW, APPROVED, REJECTED, DUPLICATE, QUARANTINED, WITHDRAWN and
TRAINING_INCLUDED. Confirmation never implies approval. Immutable manifests contain source hashes
and lineage, not copies of biometric media. Withdrawal removes the source/ROI BLOBs using SQLite
secure deletion and removes landmark metadata; it retains minimal labels, consent/audit receipts,
hashes and committed lineage. Copies in operator backups, derived features/checkpoints or browser
memory are not automatically erased. Previously fitted models require review/retraining; withdrawal
does not reverse learned weights. Data is not described as fully anonymized or encrypted at rest.

Run retention maintenance explicitly, or schedule an authenticated POST to `/retention/expire`.
Expiry is not a daemon: an installation that never runs maintenance will retain uncommitted
records longer. Included records remain until withdrawal/reviewer retention action. Protect the
workspace with OS access controls and disk encryption suitable to your deployment. No recording
is uploaded to an external service by this implementation.

## Reproducible checks

```powershell
.\.venv-word-recognition\Scripts\python.exe -m pytest tests pipeline/eval/tests pipeline/grid/tests pipeline/avhubert/tests
npm --prefix frontend run test:live
npm --prefix frontend run test:math
npm --prefix frontend run test:adaptive
npm --prefix frontend run build
.\.venv-avhubert-cpu\Scripts\python.exe scripts/probe_adaptive.py --out .qa/new-adapter-probe.json
.\.venv-avhubert-cpu\Scripts\python.exe scripts/probe_adaptive_api.py --out .qa/new-real-api-probe.json
.\.venv-word-recognition\Scripts\python.exe scripts/audit_adaptive.py --out .qa/new-artifact-audit
```

The adapter probe uses one real historical validation HELLO recording and three adapter invocations
per task. Its alphabet input is a word ROI, so that result is runtime evidence only. The temporary
API and browser tests use explicit software fixtures where stated; they do not count as verified
human feedback, a training dataset, or physical webcam accuracy. See [results](ADAPTIVE_RESULTS.md).

With the app serving a current production build on port 8018, run
`node scripts/check_adaptive_browser.mjs` for the new lab. The retained live-demo regression copy is
`.qa/adaptive_20261116/legacy_browser/browser_qa.mjs`. Those browser tests need the installed Chrome
and existing genuine `.qa/afternoon-paced-replay.y4m`; all prediction API fixtures in those tests
are explicitly software fixtures. They never submit their camera replay to a real dataset store.
