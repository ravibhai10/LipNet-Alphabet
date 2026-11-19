# Optional contextual interpretation — 2026-11-18

The application now has a separate, optional text-only interpretation panel in the word assistant. Visual predictions, uncertainty, scores, history, classifier weights, preprocessing and existing prediction APIs remain unchanged. The feature is off by default. An LLM suggestion never replaces a visual result, including an abstention. A person may accept it into a separate interpretation field.

## Selection and evidence

The selected hosted model is **GPT-6 Luna**, with reasoning disabled and strict JSON schema output through the Responses API. Official documentation describes it as an efficient focused-task model and supports structured outputs: [model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna), [structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs). Model availability, account access, provider quality and live latency have not been verified with an actual key. This is a practical provisional choice, not a comparative LLM accuracy winner.

| Option | Suitability on this machine | Decision |
|---|---|---|
| Existing visual model alone | No new cost, no text disclosure, known offline evidence | Default |
| Hosted small model | No additional model weights or GPU/RAM allocation locally; network/cost/privacy tradeoff | Optional GPT-6 Luna adapter |
| Local quantized language model | Competes with AV-HuBERT on a 7.74 GiB CPU-only machine; only about 596 MiB RAM free during checks | Not installed or benchmarked |

No dependencies were installed. The deployment environment lacks the development HTTP client; the adapter therefore uses Python's standard library in a dedicated two-thread executor. Two outstanding socket operations maximum, no queued backlog, socket timeout 1.2 seconds, service deadline 1.5 seconds and browser deadline 2.2 seconds. Running socket operations remain counted after caller cancellation. DNS/OS socket teardown can outlast the caller deadline; those operations cannot occupy the visual executor or create an unbounded queue.

## Pipeline and decision rules

1. Existing capture, preprocessing and model inference complete normally.
2. Only a completed word result may trigger the optional interpreter. Stop, next capture, mode/context/credential changes, disabling and consent revocation invalidate pending results.
3. The new authenticated `/api/context/interpret` endpoint accepts at most five unique descending-probability candidates, one vocabulary word per candidate, 240 context characters, and six vocabulary history words. Total request body is limited to 8 KiB before parsing. Existing private collector/reviewer access is reused.
4. Alphabet and 500-word requests bypass the LLM because reliable confidence gates are unavailable. Personal mode is not integrated. Twenty-word interpretation does not activate the separately gated twenty-word visual backend or grant any dataset permission.
5. Eight-word confidence/margin .95/.20 and twenty-word .50/.30 are copied from existing validation-selected rejection policies; accepted high-confidence results bypass the provider. These are not calibrated correctness probabilities. Eight-word historical validation ECE was 0.04038, Brier 0.09299; this is not conversational calibration. Original model and evaluation hashes remain unchanged.
6. No genuine explicit context, no consent, disabled provider, missing key or insufficient visual evidence means deterministic unchanged output. History alone is insufficient. Top-score .25 and suggested-score .10 are conservative request/suggestion floors, **not validated accuracy thresholds**; they never authorize automatic correction.
7. Eligible ambiguous events send only candidates/scores, explicit context and bounded history to the fixed HTTPS OpenAI endpoint. No image, frame, video, audio, source filename, user ID or access credential enters the prompt. `store=false`; this does not promise zero provider retention under its account policies. Proxies and redirects are disabled. Context/history are explicitly marked untrusted data.
8. JSON schema restricts the returned word to this event's Top-K or null. The application additionally validates schema, membership, reason, response size, refusal and incomplete status. Unsupported words are rejected even if schema adherence fails.
9. Failures, timeout, overload or malformed output retain the original visual prediction. The UI validates event identity, original prediction, unchanged scores and allowed suggestions again. There is no retry loop.
10. Per-principal cooldown is two seconds. Two provider operations maximum. Cache is in-memory, 128 entries, 60-second TTL, scoped by authenticated actor and exact model/candidates/context/history hash. Bounded context text remains only in the current request; cache keys are hashes and values are validated suggestions. UI text/private access credentials are memory-only.

The UI displays original visual output, candidate, visual score, LLM suggestion, its **visual** score, latency/cache status and separate user-confirmed interpretation. It does not display invented LLM confidence. Confirmation/rejection goes to authenticated `/api/context/feedback` as audit-only human-reported feedback. It is not an evaluation label, training example, visual prediction or correction policy update.

Service INFO logs contain event IDs, visual scores/results, suggestion, applied=false, decision and latency. Feedback INFO logs contain human-reported choice and local actor. They exclude prompt context, transcripts, keys and media. Configure the service logger at INFO and use the existing deployment's log retention if durable audit logs are required; no new private log database is created.

## Setup

Run from the inner repository directory. Use the existing deployment runtime and its existing private access configuration. No real credential was requested, generated, printed or changed by this implementation; no live provider request was made.

```powershell
# Set OPENAI_API_KEY privately in this server process using your secret manager.
# Never put it in VITE_* variables, frontend fields, source files or committed .env files.
$env:LIPNET_LLM_ENABLED = '1'
# OPENAI_API_KEY must already be set privately before launching.
# Existing deployment normally selects .access/access.json; configure its path if needed:
$env:LIPNET_ACCESS_FILE = "$PWD\.access\access.json"
.\start_server.bat
```

The existing AV-HuBERT dependency incompatibilities and model startup requirements are unchanged; this command does not repair or downgrade them. The new module imports successfully in that environment without package changes. Actual full visual model startup/inference was not rerun for this integration.

### Dedicated local context runtime-test identity

The optional `runtime-test` principal uses the `context_tester` role and the same Bearer-token and SHA-256-digest authentication mechanism as the existing local principals. Its exact protected route allowlist is `POST /api/context/interpret`, `POST /api/context/sessions`, `POST /api/context/sessions/consent`, `POST /api/context/sessions/consent/revoke`, and `POST /api/context/sessions/terminate`. `GET /api/context/health` remains public. The test role is denied context feedback, adaptive/capture/recording operations, participant administration, reviewer operations, and model administration. It is a local test actor, not a verified webcam participant.

Provisioning is explicit and separate from the existing operator/collector credential set. Create a new ACL-restricted directory, then run `python scripts/configure_local_access.py --context-test --directory <private-directory>` and set `LIPNET_ACCESS_FILE` to its `access.json` before `start_server.bat`. On Windows, the provisioning utility verifies that the current account, SYSTEM and Administrators are the only principals with allow access on the directory and credential files; it refuses unverified ACLs, non-ignored in-repository paths, existing credential files and partial provisioning. The access file stores only the token digest; `tokens.json` holds the raw token privately and is never printed. See `engineering/20261117/DEPLOYMENT.md` for the operator ACL and revocation procedure. This identity does not enable LLM interpretation by default and automated tests use only temporary files and databases with provider calls disabled.

For a separately served development UI:

```powershell
cd frontend
npm run dev
```

In the word assistant enable **optional LLM interpretation**, enter application context and the **existing local private access credential** (not the OpenAI key), and explicitly consent to text sharing. Finish a word capture; use Interpret this result to request/retry manually. The camera must still be active and the capture complete. Missing credentials/service/key produce a visible fallback message. `/api/context/health` reports configuration without exposing secrets. Disabling the feature or setting `LIPNET_LLM_ENABLED=0` stops new provider use immediately.

## Verification and replay

```powershell
.\.venv-word-recognition\Scripts\python.exe -m pytest tests/test_context_interpretation.py tests/test_engineering_resolution.py tests/test_backend_contract.py tests/test_recognition_capabilities.py tests/test_expanded_words.py tests/test_reliability.py tests/test_live_assistant_api.py -q
.\.venv-word-recognition\Scripts\python.exe scripts/benchmark_context_interpretation.py --output .qa/llm_context_replay_NEW
cd frontend
node --experimental-strip-types --test tests/context-interpretation.test.mjs
npm run test:live
npm run test:adaptive
npm run test:math
npm run build
```

Replay refuses to overwrite reports. It reads frozen historical validation/test predictions, never trains, never creates context from labels and never calls a provider. Provider/transport/schema/auth/timeout/concurrency/cache/stale behavior is verified with fixtures. Network fixtures are functional tests, not labeled LLM accuracy evidence. The report distinguishes top-1 accuracy from the acceptance/abstention policy. See `.qa/llm_context_20261118/final/benchmark.json`, `protected_comparison.json` and the accompanying report.

Isolated-word data cannot establish conversational correction benefits. A consented conversational validation collection with unseen speakers, explicit real context and unchanged vocabularies is required before testing any automatic policy. Tune thresholds on validation only; freeze model/prompt/gates/cache/context protocol; evaluate the protected contextual test once. Measure base/enhanced accuracy, macro-F1, gained/lost corrections, abstentions, actual provider and end-to-end p50/p95 latency, cost and webcam throughput. Reject automatic promotion if introduced errors, protected high-confidence overrides, speaker leakage or latency regression occur. No automatic correction is currently implemented.

## Implementation and rollback

New backend: `app/services/context_interpretation.py` and `app/context_routes.py`. Small insertions: router registration in `app/main.py`; collector authorization/body cap in `app/security.py`. New UI: `frontend/src/live-demo/contextInterpretation.ts` and `frontend/src/workspace/components/ContextInterpretation.tsx`; one import and one panel insertion in existing `WordAssistant.tsx`. New verification: `tests/test_context_interpretation.py`, `frontend/tests/context-interpretation.test.mjs`, `scripts/benchmark_context_interpretation.py` and this document. Generated frontend `dist` was rebuilt. No package manifests, lockfiles, classifier code, checkpoint, label order, preprocessing adapter or dataset were changed by this integration.

Operational rollback: set `LIPNET_LLM_ENABLED=0`, restart backend, disable the UI toggle. For code rollback, remove only this feature's imports/router/panel and its new files, preserving other dirty edits in the shared files. Do not use a whole-file Git reset: this worktree already contains unrelated user changes. No model restoration or retraining is needed. Historical and current visual predictions remain the authoritative recognition outputs.
