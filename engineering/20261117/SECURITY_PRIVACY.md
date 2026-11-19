# Security and privacy verification

This is an engineering verification report, not a compliance certification. All runtime tests use temporary software fixtures except the separately labeled reference-model smoke on a historical training sample.

## Implemented boundaries

Sensitive recording, studio export, adaptive, legacy training, generic video and compatibility personal routes authenticate before request parsing. Missing configuration fails closed. Legacy stores require reviewer credentials because they do not have reliable per-record ownership. Adaptive collectors can access their own sources and models; reviewers perform identity/permission review, dataset freezing and deployment. Stable principal IDs survive token rotation.

Request buffering has explicit byte ceilings and a total upload deadline. JPEG/PNG dimensions are checked before image decode, with aggregate decoded-pixel limits. Original video headers and decoded geometry are bounded and source samples are checked against video timing/content. Native video parsing is not an isolated security sandbox; malformed codec handling remains a residual risk requiring further process isolation for hostile/public deployment. The supported profile is localhost, authenticated collection and one process.

The global personal prototype service added concurrently was preserved in source but removed from supported serving. Collection no longer fits it automatically. It cannot claim ready based on a file or replace an owner-scoped approved model. The exact encoder hash is checked before legacy publisher pickle loading; ordinary heads use restricted `weights_only=True` loads. This does not certify every offline script as safe for arbitrary untrusted checkpoints.

## Lifecycle

Accepted adaptive sources have a persisted expiry; reads fail at expiry even before cleanup. Review quarantine preserves borderline recordings with consent. Rejected media is erased immediately, retaining a minimal decision receipt. A reviewer cannot clear quality warnings without a detailed rationale. Thresholds are versioned engineering heuristics and require independent calibration.

Withdrawal erases source/ROI BLOBs and image-derived metadata, invalidates downstream registry lineage, sets affected champion pointers to preserved reference IDs and prevents affected rollback. Private derivative deletion only targets that model's dedicated directory; public/reference files and sibling candidates are excluded. Failures remain retryable and operator-visible. SQLite uses secure_delete, but no forensic filesystem/SSD erasure or backup-unlearning claim is made.

Maintenance starts and stops with the application. Durable job states distinguish interrupted work from success. Final-test access is bound to a candidate; a second model cannot acquire another candidate's resumed claim. Native work cannot be safely interrupted: it retains resource capacity while draining. A stopped training executor must be reopened explicitly before new work.

Legacy collection bundles saved through the compatibility API have registered expiry. Managed eligible backups inherit the source capture's expiry and withdrawal and must live inside the designated private backup directory. The server cannot erase unregistered external backups or offline browser copies. Old unregistered studio, camera-export and collection data require an explicit ownership/consent migration; this mission does not destroy existing user data to simulate successful migration.

## Executed checks and practical limits

Tests cover stable ownership through credential rotation, invalid credentials, pre-parser rejection, source expiry before cleanup, renewal ownership/consent, withdrawal fallback, deletion refusal outside private candidate directories, job/attempt restart states, SQLite read/write concurrency, managed collection expiry, disabled unsupported serving, CPU cancellation capacity, candidate-bound final-test resumption, pre-decode JPEG rejection, codec correspondence and quality/deletion behavior. Exact suite counts and failures are retained in the verification record.

The baseline hash comparison checked 10,120 model/run/historical artifacts with zero changes or missing files. The existing protected alphabet and eight-word checkpoint semantics were retained. This check covers the recorded inventory, not every untracked dataset byte or later concurrent edit.

Unresolved engineering limitations remain explicitly registered: complete legacy-store migration; atomic withdrawal coordination during training/final-evaluation commits; quota enforcement during candidate artifact growth; full hostile-codec isolation; multiworker process coordination; complete typed responses; clean encoder dependency installation; physical camera validation; and coordinated browser retention, multi-tab, full-body deadline and service-worker tests. The browser build and Node tests do not establish these lifecycle guarantees.
