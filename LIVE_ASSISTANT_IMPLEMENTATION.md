# Live word assistant: audit and implementation plan

## Audit completed before implementation

The existing app is React 18/TypeScript/Vite with FastAPI. The shared camera/landmark/ROI loop is `frontend/src/live-demo/useDemoPipeline.ts`; the workspace screen is `frontend/src/workspace/pages/LiveDemo.tsx`. `useWebcam.ts` owns video-only streams and handles permission/playback/device errors. `useLandmarker.ts` loads the existing CPU MediaPipe asset. The same loop is also consumed by the landing/story sections and the testing workspace, so its current return contract must remain compatible.

The workspace currently starts inference from heuristic mouth movement, uses overlapping windows, and clears both results and history on stop/face loss/backend failure. It has no explicit next-word cycle. `WordConsensus.update()` currently confirms even uncertain first-window results; the existing tests require two accepted agreeing windows. The frontend request timeout currently ends when response headers arrive, so response-body parsing is not fully covered.

`app/main.py` serves `/api/predict` and `/api/predict/word`; `app/lrw500_routes.py` serves `/api/predict/word/lrw500`. Both word routes share a nonblocking CPU lock and return 429 when busy. The eight-word backend uses the preserved frozen AV-HuBERT encoder and supervised eight-class head, measured 96px ROIs resampled to 29 frames and center cropped to 88px. The separate publisher ResNet18/DC-TCN has 500 outputs and needs measured full-face frames, 68 landmark correspondences, timestamps, and `final=true` for its existing two-temporal-view check. It accepts 12–76 webcam samples spanning 0.32–3 seconds, at least 10Hz, no gap above 160ms. Existing alphabet windows resample to 30 frames at 140×46. None of these model/preprocessing contracts will be changed.

Confidence is an uncalibrated model-derived softmax score. The existing backend rejection decision remains authoritative. Missing confidence will be displayed as unavailable; an optional stricter frontend score filter must never turn a rejected response into acceptance. Session response counts and optional user feedback are not ground-truth accuracy.

The running research backend uses port 8001. It must not be stopped or have its selected models changed. CPU model loading, thread counts, precision, checkpoint files and research evaluation artifacts remain protected. Existing project changes are already present in the worktree and will be preserved.

## Planned changes

1. Add a pure typed state machine/controller and a small React subscription hook. It will own independent clip buffers, attempt IDs, cancellation, one-request gating, monotonic timers, result/history retention, local feedback and session statistics.
2. Integrate manual word capture into the existing pipeline through an option used by the workspace screen. Preserve alphabet streaming and existing landing/story consumers. A word finishes only when the user clicks Finish Word; model duration/frame limits reject oversized clips rather than guessing a word boundary.
3. Update the existing workspace screen and scoped CSS with camera guidance, capture/processing timers, explicit Next/Retry actions, uncertainty, history, feedback, summary and accessible controls. Preserve all model choices and the landing palette.
4. Harden prediction transport/response validation without changing endpoints or model output meanings. Keep existing backend rejection; return safe service errors and test route/resource handling.
5. Fix the existing word-consensus bug and exercise camera/device/unmount cleanup and stale-response protection.

Potential new source files: `wordAssistant.ts`, `useWordAssistant.ts`, and an inference transport/response boundary if necessary. Existing files likely affected: the shared live pipeline, webcam hook, API/backend context, workspace LiveDemo page, scoped live CSS and test script list. Tests and a final implementation report will be added. No new database, model, GPU dependency, external service or separate demo app is planned.

## Compatibility and regression checks

- Preserve `/api/predict/word` and `/api/predict/word/lrw500` request geometry and `final` semantics.
- Preserve the existing shared-hook fields used by the story, landing and testing routes.
- Use a manual controller only for workspace word modes; do not require unverified automatic word-boundary detection.
- One fresh buffer per attempt; no preroll from a previous word, no sliding-window truncation, one POST per completed manual clip.
- Stop, mode change, navigation and unmount cancel pending browser work and release tracks. Aborting HTTP cannot cancel an already executing CPU model; the backend CPU lock remains necessary.
- Retain session history/results when stopping; explicit reset clears them. Feedback stays in session memory.

## Test strategy

Run the current live tests as baseline, then add state/transition, duplicate click, next/retry, fresh-buffer, stale-response, timeout, feedback and summary tests using the existing Node test runner. Use backend pytest for request schema/invalid frames/exception/concurrency checks. Run the complete existing backend suite, both existing frontend test scripts and the production build after implementation. Use the installed Playwright/Chrome tooling with controlled camera/API fixtures to verify UI flows and lifecycle; label fixture checks as software tests. Perform a separate real labeled-ROI API smoke if the preserved inputs and running backend allow it. Record actual results and distinguish them from a physical-webcam accuracy benchmark.

Final measured outcomes, file changes, commands, limitations and manual verification instructions will be appended after implementation.

## Implemented outcome — 30 October 2026

The existing `/demo/live` workspace now provides manual, independent word capture for both **8 words** and **500 words**. Alphabet streaming, the original APIs, research evidence, camera preview, anatomical overlays, crop/signal previews, vocabulary search, and all existing workspace routes remain available. The implementation adds no separate application, model, database, dependency, training job or automatic word-boundary claim.

### Using the assistant

1. Choose **8 words** or **500 words**. Both choices remain present; actual backend health controls availability.
2. Click **Start camera**, allow camera access, face the camera, and wait for usable face/mouth tracking. Camera preview alone sends **zero word prediction requests**.
3. Click **Start Word**. The buffer begins empty. Say one supported word while keeping your mouth visible. The timer measures elapsed capture time, and the progress bar measures buffer usage, not linguistic completion.
4. Click **Finish Word**. The button becomes available after the model's minimum frame count. One completed clip produces one HTTP POST. The screen remains in **PROCESSING** until a response, cancellation or failure.
5. Read the last prediction, model score if supplied, measured server duration if supplied, and measured browser round trip. Backend-rejected output appears as **Uncertain**, with its candidate shown separately.
6. The camera remains active while capture and inference stay paused. Click **Next Word** for a fresh clip, or **Retry Word** to repeat the attempt without restarting the camera. Both clear the temporary clip and timer. History retains earlier responses.
7. **Stop camera** releases tracks and cancels browser work while preserving results/history/totals. **Clear History** removes entries, retaining cumulative totals. **Reset session** stops capture and explicitly clears the session.

History is newest first, capped at 100 responses in page memory. It contains candidate/word, status, model mode, timestamp, score, server inference duration, and retry linkage. Correct/Incorrect feedback is an explicit user annotation; it never changes the model or becomes an accuracy metric. Navigation ends this in-memory session; nothing is silently uploaded as feedback or persisted in a database.

### State machine and request ownership

`wordAssistant.ts` owns transitions, the current frame array, monotonic timing, unique attempt IDs, a cancellation generation, one AbortController, and one request deadline. `useWordAssistant.ts` subscribes through React's `useSyncExternalStore`; the camera loop samples only while the controller is capturing.

```text
IDLE → CAMERA_STARTING → READY → CAPTURING → PROCESSING
     → RESULT → WAITING_NEXT → READY → CAPTURING

RESULT / WAITING_NEXT → RETRY → CAPTURING
ERROR → READY or RETRY → CAPTURING
Any active phase → STOPPING → IDLE
Failed active operation → ERROR
```

The Next Word handler transitions through READY and begins capture in the same deliberate user action. RESULT and RETRY are explicit controller transitions even when React batches them into the following visible phase. Invalid transitions, rapid duplicate clicks, repeated Finish, and attempts to capture during processing are rejected. A response must belong to the current request generation and attempt. Stopped, timed-out, disconnected or unmounted attempts cannot publish late results. Frames and landmark arrays are not stored in history.

The workspace does not offer automatic word capture because the existing motion heuristic is not a verified linguistic boundary detector. Legacy alphabet/story/testing consumers retain their existing streaming path. The legacy word-consensus regression is repaired: uncertain windows reset consensus, and two accepted matching windows from the same utterance are required. The manual eight-word path makes one request per explicit completed clip; it does not send an identical clip twice to manufacture agreement. The manual 500-word path retains the backend's two distinct temporal views within its single `final=true` request.

### Model and API compatibility

| Mode | Existing POST | Preserved capture/preprocessing contract |
| --- | --- | --- |
| Alphabet | `/api/predict` | JPEG mouth crops; backend resamples to its checkpoint's 30 frames at 140×46 |
| Eight words | `/api/predict/word` | Measured 96×96 mouth JPEGs, 8–76 incoming frames; existing AV-HuBERT path interpolates to 29, crops to 88×88 and applies its recorded normalization |
| Publisher 500 words | `/api/predict/word/lrw500` | Full-face JPEGs plus measured 68 landmarks, monotonic timestamps, one-face counts, versioned convention and `final=true`; existing mean-face alignment and temporal-view check |

The manual eight-word capture has a five-second/76-frame resource safeguard; it does not assume that words last five seconds or use the timeout to split a word. The eight-word feature head has no native-FPS contract, so it is not subjected to the publisher model's 160 ms gap limit. The 500-word capture preserves 12–76 frames, 0.32–3 seconds, at least 10 Hz and maximum 160 ms gaps. Invalid or oversized clips fail instead of silently discarding the beginning of a word. No checkpoint, rejection threshold, target shape, normalization, architecture, thread count, precision, encoder-loading strategy or selected backend was changed.

The API fields used are `task`, `predicted_word`, `uncertain`, `confidence`, `top_predictions`, optional `all_probabilities`, optional `latency_ms`, optional `performance`, and the existing LRW `diagnostics.temporal_consensus.confirmed`. A missing/null score becomes **Confidence unavailable**; missing timing becomes **Unavailable**. Nonfinite or out-of-range scores, invalid JSON, wrong task/word structure, malformed candidate vectors, negative timings or malformed supplied performance/confirmation fields fail at the frontend boundary. A valid backend rejection always remains rejection. The optional minimum-score filter can only reject additional outputs. Scores are explicitly described as uncalibrated.

No inference response schema was changed. Backend additions are safe generic HTTP 500 handling and sanitized loader errors in readiness responses. Value errors remain 422, unavailable models 503 and CPU contention 429; locks release in `finally`. JPEG count, encoded-size (200,000 characters/frame), base64, JPEG signature and decoded-dimension (maximum side 1024) checks remain in the existing decoder. Model-specific spatial and landmark checks remain in their existing services. Full-frame 500-word capture no longer performs an unused second mouth-JPEG encoding; the model receives the same full-face contract.

### Error handling and resource lifecycle

- Permission denied, no device, device busy, unsupported media APIs and playback failure have distinct camera recovery messages.
- Disconnect releases every track and ends the active attempt. Face/crop loss during capture invalidates the incomplete clip. Existing completed results/history survive transient face loss.
- Stop, mode switch and navigation abort browser requests and discard stale callbacks. The webcam hook also stops streams that arrive after a cancelled permission request; cancellation no longer blocks a new camera attempt behind the old promise.
- A 30-second prediction deadline covers transport **and JSON/error-body parsing**, including noncooperative transports. Capture deadlines are separate. No failure leaves the controller in PROCESSING indefinitely.
- The backend's shared nonblocking CPU lock still prevents concurrent word-model execution. Aborting a browser fetch does not cancel a model already executing on the server; a subsequent attempt can legitimately receive 429 and be retried.
- Processing/recording states have visible indicators and measured timers. Keyboard focus, disabled controls, announced phase/result changes and reduced-motion spinner handling use the current design system. Fast timer ticks are kept outside the phase live announcement.

### Exact source files changed for this task

| File | Change |
| --- | --- |
| `frontend/src/live-demo/wordAssistant.ts` (new) | Pure state machine, independent clips, deadlines, cancellation, results/history/feedback/statistics |
| `frontend/src/live-demo/useWordAssistant.ts` (new) | React subscription and timer cleanup |
| `frontend/src/workspace/components/WordAssistant.tsx` (new) | Capture/status/result actions, score interpretation, session summary, history and feedback |
| `frontend/src/live-demo/useDemoPipeline.ts` | Opt-in manual word integration; preserved shared return interface and legacy streaming path |
| `frontend/src/live-demo/useWebcam.ts` | Safe cancellation/restart while a permission request is still pending |
| `frontend/src/live-demo/wordConsensus.ts` | Repair uncertain/first-window acceptance regression |
| `frontend/src/workspace/pages/LiveDemo.tsx` | Integrate manual flow into the existing workspace, retain all three recognition choices and real diagnostics |
| `frontend/src/workspace/pages/Testing.tsx` | Render absent confidence safely for the shared history type |
| `frontend/src/lib/api.ts` | Full-response prediction timeout, cancellation and typed response validation |
| `frontend/src/hooks/useBackend.tsx` | Optional error propagation to the manual controller; existing callers retain nullable behavior |
| `frontend/src/styles/live-workspace.css` | Scoped responsive assistant styles using existing mint/charcoal/ivory colors |
| `frontend/package.json` | Include added tests in the existing live test command |
| `app/main.py`, `app/lrw500_routes.py` | Sanitize unexpected word-inference exceptions while releasing the existing shared CPU lock |
| `app/services/pretrained_word_inference.py`, `app/services/lrw500_inference.py`, `app/services/word_inference.py` | Keep loader details in local logs; return safe readiness errors |
| `frontend/tests/word-assistant.test.mjs`, `frontend/tests/prediction-api.test.mjs` (new) | State, request, independent clips, timing, uncertainty and failure regression coverage |
| `tests/test_live_assistant_api.py` (new) | Both routes, schema/decoder validation, absent confidence, sanitized errors and lock release |
| `scripts/check_live_assistant.mjs` (new) | Existing-app Chrome software checks with explicit API fixtures and genuine-video camera replay |
| `scripts/check_live_assistant_inference.py` (new) | Genuine saved-input HTTP execution and checkpoint preservation receipts |
| `LIVE_ASSISTANT_IMPLEMENTATION.md` | Preimplementation audit/plan and this final implementation report |

Pre-existing unrelated changes in the working tree, the research paper, other pages, models, data and evaluation outputs were preserved.

### Tests actually executed

| Command | Actual outcome | Receipt |
| --- | --- | --- |
| `npm run test:live` before repair | 14 passed, 4 failed; all four failures were the existing word-consensus regression | Initial command output; also documented in the preceding project audit |
| `.\.venv\Scripts\python.exe -m pytest -q` | **526 passed, 4 skipped**, 64.76 s; no ignored failures | `.qa/live-assistant-python-tests.txt` |
| `npm run test:live` after implementation | **33 passed, 0 failed** | `.qa/live-assistant-live-tests.txt` |
| `npm run test:math` | **26 passed, 0 failed** | `.qa/live-assistant-math-tests.txt` |
| `npm run build` | TypeScript and Vite production build passed | `.qa/live-assistant-build.txt` |
| `node scripts/check_live_assistant.mjs` | **90 browser assertions passed**, no uncaught exceptions | `.qa/live-assistant-browser.json`, `.qa/live-assistant-browser-log.txt` |
| `.\.venv\Scripts\python.exe scripts/check_live_assistant_inference.py --stage before` | Genuine saved 29-frame connection input, HTTP 200, correct candidate, not rejected | `.qa/live-assistant-real-before.json` |
| Same command with `--stage after` | Same source, identical full class probability vector; five checkpoint SHA-256 values unchanged | `.qa/live-assistant-real-after.json` |
| `.\.venv\Scripts\python.exe -m pipeline.lrw_tcn.inference --lrw-centered-video .qa/lrw-public-example/AFTERNOON.mp4` | Genuine publisher checkpoint loaded and returned 500 finite probabilities; candidate AFTERNOON, score 0.958904, not rejected | `.qa/live-assistant-lrw500-smoke.json` |

The full Python run includes the existing real alphabet-checkpoint load/probability test. The new backend test file initially exposed a Windows pytest temporary-path problem caused by an oversized parameter ID; short explicit test IDs fixed the test harness. A browser run initially rejected eight-word capture under the unnecessarily inherited 160 ms LRW rule; that implementation error was corrected before the successful run. Existing tests were not removed or weakened.

The fresh 500-word CLI execution used one genuine public Oxford AFTERNOON clip and the existing fixed LRW-centered crop. Its recorded model forward duration was 0.6972 s, excluding load/preprocessing. That crop does not reproduce full publisher mean-face preprocessing. This single execution is checkpoint/inference evidence, not a 500-class or physical-webcam accuracy estimate.

Browser coverage includes real MediaPipe tracking on a **genuine public AFTERNOON video replay**, manual camera start, first/second independent capture, one pending POST, no automatic inference during waiting, Next/Retry, uncertain results, unavailable confidence, history and feedback, Stop during processing, stale-response rejection, permission/no-device/busy errors, invalid HTTP/schema responses, disconnect, cancellation with late camera permission, explicit reset, separate LRW metadata, navigation/unmount cleanup, and mobile layout. Twelve routes rendered without horizontal overflow: `/demo/live`, `/demo/overview`, `/demo/dataset`, `/demo/training`, `/demo/testing`, `/demo/models`, `/demo/experiments`, `/demo/pipeline`, `/demo/docs`, `/science`, `/research`, and `/`. API results in this browser suite are explicitly controlled **software fixtures**, not model recognition measurements. Desktop/mobile screenshots are `.qa/live-assistant-{desktop,mobile}.png`.

### Measured CPU execution and before/after scope

Two separate genuine-input HTTP requests used the **same existing running backend**, unchanged models and the same source `miracl:F10:words:03:03`. Both predicted `connection`, score 0.9990013242, with identical eight-class probabilities. Five fingerprints match exactly: alphabet checkpoint, frozen AV-HuBERT encoder, eight-word head, publisher LRW checkpoint, and preserved scratch baseline.

| Measurement | First smoke, before browser validation | Repeat smoke, after browser validation |
| --- | ---: | ---: |
| Server-reported model duration | 5862.53 ms | 2339.36 ms |
| Measured HTTP round trip | 5983.92 ms | 2379.97 ms |
| Process CPU percent of one core | 26.00% | 51.51% |
| Resident process memory after request | 369.84 MiB | 374.63 MiB |

These are **two uncontrolled execution samples, not a performance improvement claim**. Cache/page residency, warmup and other machine activity differ. The receipt labels `before`/`after` mean before/after browser validation, not a controlled prechange/postchange frontend benchmark. The backend process was never replaced or reconfigured. No physical-camera startup or capture baseline was recorded before source changes; those comparisons are unavailable. Browser receipts record actual fake-camera acquisition durations and captured counts, but their fixture response timings are not CPU model timings. No reduction in model latency, memory consumption or accuracy improvement is established.

| Behavior | Audited previous workspace | Implemented manual word workspace |
| --- | --- | --- |
| When capture/inference starts | Mouth-motion heuristic after camera start | Explicit Start Word/Next Word/Retry, then Finish Word |
| Requests per utterance | Potential overlapping sliding windows | One POST per completed manual clip; zero while waiting |
| Word boundaries | Heuristic, not validated | User-controlled; no automatic boundary claim |
| New word | No explicit independent cycle | Empty buffer, reset timer, unique attempt ID |
| Stop/face-loss history | Cleared by the shared loop | Preserved until explicit clear/reset or navigation |
| Confidence | Assumed supplied numeric score | Supplied score or unavailable; rejection remains authoritative |
| Feedback/summary | No manual word session summary | Session-only feedback and counts explicitly separated from accuracy |
| Transport deadline | Ended after response headers | Covers complete prediction response and body parsing |
| Model/precision/CPU selection | Existing research models | Same research models and configuration |

### Run and manually verify

The existing research server on **port 8001 was left running**, as was the pre-existing server on port 8002. The production frontend was rebuilt; refresh **http://127.0.0.1:8001/demo/live** to load the assistant. No camera starts on page load. Backend source hardening takes effect on the next normal server restart; its new behavior was tested through isolated FastAPI test clients without interrupting either existing server.

For a future start, from the repository directory containing `app`, `frontend` and the existing environments, use a **free port**:

```powershell
# Build the existing frontend.
Push-Location frontend
npm run build
Pop-Location

# If 8001 is already in use, keep it running or choose an unused port.
# This existing launcher selects the separately configured eight-word backend.
.\start_word_demo.ps1 -Port 8001
```

For frontend development against that existing backend in another terminal:

```powershell
Push-Location frontend
$env:VITE_API_BASE_URL = 'http://127.0.0.1:8001'
npm run dev -- --host 127.0.0.1
Pop-Location
```

Repeatable checks:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
Push-Location frontend
npm run test:live
npm run test:math
npm run build
Pop-Location

# Requires installed Chrome, the existing real replay fixture, and the running app.
node scripts/check_live_assistant.mjs
```

The real-inference receipt script refuses to overwrite completed receipts. Its stage commands above are the exact executed commands, not instructions to repeatedly retune against the saved test clip. Browser checks replace their software-test receipts on rerun.

Manual verification: select 8 words, start the camera, wait for face/mouth readiness, capture **hello**, finish, inspect the real response, wait without clicking and confirm no new request in the browser Network panel. Click Next Word, capture **stop**, and inspect a second independent request and history entry. Retry once, annotate feedback, stop the camera and confirm history remains. Reset explicitly. Repeat capture/Finish with a word from the 500-word vocabulary while meeting its stricter timing requirements. Navigate away during a capture and confirm the browser camera indicator turns off. Physical recognition results from this procedure must be recorded before being described as an accuracy benchmark.

### Remaining limitations

- Physical-webcam recognition accuracy, unseen isolated-word rejection, automatic word boundaries and a controlled performance comparison remain unmeasured.
- The eight-word mode still recognizes only its eight supported words; the 500-word publisher model retains its separate, uncalibrated research limitations. Neither is free-speech transcription.
- Confidence percentages are uncalibrated model scores. User feedback and response counts are not ground truth evaluation.
- CPU inference can remain slow under memory pressure. Processing expires after 30 seconds; stopping a browser request cannot terminate an already running server model.
- LRW capture legitimately fails when the camera/CPU cannot meet its existing timing/landmark requirements. It is not silently weakened to make a demo pass.
- History lives in current page memory, with at most 100 entries. Navigating away clears it; there is no cross-session feedback storage.
- No checkpoint was retrained, no backend was activated/replaced and no new recognition improvement is claimed.
