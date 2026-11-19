# Backend integration and migration contract

The generated [endpoint inventory](ENDPOINTS.md) and [OpenAPI snapshot](openapi.json) describe this checkout. OpenAPI `x-minimum-role` includes authorization performed before parsing the request. Some legacy success responses remain untyped; X14 remains partially fixed until these are typed and integration-tested. React, IndexedDB and service-worker changes remain owned by Antigravity.

## Authorization

Send `Authorization: Bearer <token>`. Never put credentials in URLs, source code, persisted exports or screenshots. A persistent access file maps a stable principal ID to a collector/reviewer role and one or more SHA-256 credential digests. Rotate digests while retaining that principal ID. Identical digests assigned more than once invalidate configuration. Collector access is limited to its own adaptive records and models; reviewer access permits inspection/review and deployment decisions. Legacy stores lack reliable per-record ownership and require reviewer access.

Missing configuration returns 503, missing/invalid credentials 401, insufficient role 403, concealed or absent records 404, immutable/conflicting operations 409, malformed input 422, bounded body overflow 413, upload deadline 408, busy CPU 429, inference response deadline 504. A 504 does not terminate a native kernel: capacity remains occupied until the kernel exits. Retry a fresh inference after capacity becomes available. Retry capture submission with the same ID and identical content to avoid another copy.

POST/PUT/PATCH uploads have a total 30-second body deadline. `/api/adaptive/captures...` has a 90 MiB encoded-body ceiling; other routes have 16 MiB. Adaptive quality processing currently has a separate 120-second worker deadline; ordinary, LRW and expanded inference use `LIPNET_INFERENCE_TIMEOUT_SECONDS` (default 30). Deployment proxy timeouts can end a request sooner and must be considered separately. Keep the browser abort timer active until the response body is fully consumed, including errors, clones, blobs and streams.

## Collection and training

1. `GET /api/adaptive/config` supplies supported task labels, consent text and limits. Extra correction/collection words are not evidence of a supported trained class.
2. `POST /api/adaptive/captures` submits an ID, task, stable participant/session IDs, timezone-aware recorded timestamp, physical-webcam or webcam-replay provenance, explicit consent, 8–76 original lossless PNG frames, strictly increasing timestamps and optional original bounded MP4/WebM. Source video is mandatory for later training inclusion. Do not relabel prerecorded material as physical capture. The server does not attest hardware identity.
3. Capture prediction may be absent when the model is unavailable. Preserve the returned recording state and allow retry. A candidate label is never an accepted prediction. Feedback requires actual inference; `correct` cannot confirm an uncertain output.
4. `POST /captures/{id}/feedback` records an explicit correct/incorrect decision and label. `UNKNOWN` is for negative evaluation and never becomes a training class.
5. Reviewers verify participant permission through `POST /participants/verify`, then review quality and label. A checkbox does not verify identity. Quarantined quality needs a detailed rationale; unusable sources must be rejected and erased.
6. `POST /datasets` takes `{task, owner, assignments}`. Assignments map **session IDs**, not participant IDs, to train/valid/test. A personal version contains one verified participant belonging to one owner and separate held-out sessions. There must be at least two reviewed examples per class per partition. This is an engineering minimum, not a statistical sufficiency claim. Historical exposed test participants and reused final-test sessions are ineligible.
7. Training, final evaluation, integration probe and deployment are separate reviewer operations. Training produces a challenger and does not activate it. A final evaluation attempt binds candidate bytes, manifest, preprocessing and policy; an interrupted attempt may resume only the same immutable candidate. Retuning after final-test access requires fresh evidence.

`POST /api/word/train-personal` is a compatibility **collection** endpoint requiring reviewer credentials. It stages validated JPEG bundles and returns HTTP 202 with `status=validation_required`, `success=false` and `model_activated=false`. It registers expiry but cannot establish full original-video/quality governance. Migrate new collection to adaptive captures. No automatic fitting occurs. The global personal-adapter source is preserved as research code; its prediction endpoint returns 503 and its health reports unavailable. Do not display “trained” or “ready” after a collection response.

## Explicit model selection

```javascript
const response = await fetch('/api/adaptive/predict', {
  method: 'POST',
  headers: {Authorization: `Bearer ${token}`, 'Content-Type': 'application/json'},
  body: JSON.stringify({
    task: 'isolated_word', capture_id: captureId, selection: 'reference'
  }),
  signal: controller.signal,
});
const result = await response.json(); // Keep the deadline active through this await.
if (!response.ok) throw new Error(result.detail);
// Render result.model_id and result.model_version verbatim.
// Render result.prediction only when result.status === 'PREDICTED'.
```

`selection` defaults to `reference`. Only explicit `adaptive` selection may use that authenticated owner's approved champion. Another user's model must never appear in a collector list. Show exact model ID and hash/version, vocabulary, uncertainty and evidence scope. The ordinary eight-word endpoint serves the selected reference backend; it must not silently switch after adaptation.

Alphabet and LRW500 remain experimental. Twenty-word inference is disabled by default and requires explicit research opt-in; sentence recognition has no verified trained decoder. File presence, model loading and recognition reliability are separate states. “Ready” requires a compatible loaded model for that exact task; it does not imply independent webcam accuracy.

## Deletion and retention

Withdrawal blocks source bytes, invalidates derived datasets/models, selects an eligible preserved reference fallback, and queues private derivative erasure. Invalidated models cannot be rolled back into service. This is invalidation and deletion, not machine unlearning. Missing reference prerequisites must produce unavailable, never a fabricated prediction.

`POST /captures/{id}/renew` requires `{consent:true}` before expiry and the same authenticated owner. The default retention is 90 days, configured centrally. `GET /retention/report` is reviewer-only and read-only; `POST /retention/expire` executes maintenance. The report distinguishes logical media bytes, SQLite file bytes, pending deletions and managed legacy collections.

Antigravity must implement expiry across IndexedDB media, previews, temporary frames, exports under application control, cached responses and derivatives. Use atomic IndexedDB transactions for quota/duplicate checks and insertion; protect multi-tab updates and broadcast withdrawal/expiry. Reject expired reads even before periodic cleanup. An offline browser or an operator's unmanaged backup cannot be erased or attested by this backend. Reviewer-only POST /retention/backups registers an existing eligible backup inside the private backups directory with its source recording and owner; it inherits source expiry and withdrawal. Unregistered copies require operator deletion.

## Required browser acceptance

- Compare actual responses to this OpenAPI snapshot and test 401/403/404/409/413/422/429/503/504 states.
- Confirm collection cannot trigger training or promotion; show accepted predictions separately from candidates.
- Use two tabs to race saves, quota checks, withdrawal and expiry. Refresh and restart the browser before checking retained state.
- Verify body-consumption deadlines with slow streamed responses and cancellation after headers.
- Verify a new service-worker build replaces the old build, missing JavaScript receives an error rather than HTML, API/private responses never enter caches, and offline media expires.
- Validate exact model selection, model/version display, withdrawal fallback and disabled research choices.

Existing Node tests and a successful production build do not constitute these browser acceptance tests.
