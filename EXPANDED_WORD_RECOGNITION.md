# Selectable eight-word and 500-word research recognition

Date: **2026-10-29, Asia/Calcutta**. The existing application now has independent **8 words** and **500 words** modes. The eight-word AV-HuBERT model remains the default fallback. The publisher-trained 500-word model is connected to a separate real FastAPI endpoint, with explicit research status and no inherited accuracy claim.

Open **http://127.0.0.1:8001/demo/live** on this machine. The combined launcher's default is now port 8001, matching the user's existing browser URL. The separately started port-8002 instance remains available. Start from the repository root with:

```powershell
.\start_research_demo.ps1
# If this port is already occupied, leave that process alone and choose another:
.\start_research_demo.ps1 -Port 8003
```

The launcher uses the verified `.venv-avhubert-cpu` environment, restricts CPU threads, selects the existing eight-word fallback only within its child process, and restores the shell environment afterwards. The alphabet backend setting is not changed. The application serves the built frontend from `frontend/dist`.

Port-8001 correction: the user screenshot exposed a later server started with the regular `.venv` and legacy four-word backend, which correctly failed the eight-word readiness gate. That inspected process was replaced with the CPU environment and original eight-word head at the user's request. The standard BAT launcher now selects the same runtime when the head is present; its existing port remains 8000. The eight-word default is selected independently of 500-word health, and the mode buttons now use a real two-column grid. All 18 live logic tests, the production build and 12 actual-port browser checks passed, including switching 8 → 500 → 8 and keeping eight-word mode enabled during a deliberately failed 500-mode health call. No recognition responses were mocked, camera access was not started, and checkpoints/evaluations were not changed. Evidence: `.qa/eight-word-8001-before.json`, `.qa/restored-eight-word-8001.json`, `.qa/eight-word-8001-frontend-exits.json` and `.qa/restored-eight-word-browser.json`.

## What is implemented

| Mode | Model and vocabulary | Endpoint | Evidence and limitation |
| --- | --- | --- | --- |
| Alphabet | Existing 26-class CNN–BiLSTM | `/api/predict` | Preserved executable baseline; not a CTC or sentence model. Webcam accuracy unvalidated. |
| 8 words | Frozen AV-HuBERT layer 9, mean/std pooling, supervised linear classifier | `/api/predict/word` | Original **182/192, 94.79%** MIRACL result remains frozen. A real connection clip still passes the actual HTTP API. |
| 500 words | Author's video-only ResNet18 + DC-TCN, exact ordered 500-class mapping | `/api/predict/word/lrw500` | Real AFTERNOON clip passes two API input paths. Broader cross-dataset results are weak; no complete local LRW benchmark or validated webcam accuracy. |
| Sentences | CTC utilities and encoder representations only | No working prediction endpoint | No verified compatible supervised visual sentence decoder/tokenizer bundle. No sentences are manufactured from word guesses. |

Eight-word vocabulary: **choose, connection, hello, navigation, next, previous, start, stop**.

The 500-class ordered vocabulary is [pipeline/lrw_tcn/vocabulary.txt](pipeline/lrw_tcn/vocabulary.txt), matching all 500 unique nonempty entries of `data/vocab.txt`, SHA-256 `c882e2ffcbf301d4145a33b753d3c70af36a1d9e08a0a1cd1f93c01145c421fb`. Its index order is preserved through loading, inference, top-5 and evaluation. The live page includes a searchable list rather than an abbreviated invented vocabulary.

The existing three legacy 500-output CNN–BiLSTM files are preserved and remain distinct from this genuine publisher-trained model. The full authorized LRW training/test corpus is still absent; duplicate generated LRW development fixtures are not used as recognition evidence.

## Alignment and capture contracts

The [publisher preprocessing source](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/blob/0cc99fab046bd959578f2baac52e654746da4825/preprocessing/dataloader.py) was inspected directly. The new backend uses the pinned author's 68-point mean-face asset, stable indices `(28,33,36,39,42,45,48,54)`, centered landmark smoothing up to six neighboring frames, preservation of the current frame's translation, OpenCV LMEDS partial-affine estimation, a 256 × 256 grayscale face, and a 96 × 96 crop centered on transformed lip landmarks 48–67. The publisher asset SHA-256 is `dbf68b2044171e1160716df7c53e8bbfaa0ee8c61fb41171d04cb6092bb81422` and is verified before use.

**The webcam uses anatomical MediaPipe-to-68 correspondences, not the publisher's landmark detector.** That approximation and its unverified recognition effects are exposed in the API and UI. Author geometry is implemented; full benchmark-equivalent preprocessing is not claimed. A geometry fixture verifies identity alignment against the actual reference asset, not recognition accuracy.

After alignment, the model applies center 88 × 88 crop, `[0,1]` grayscale scaling and mean 0.421/std 0.165. Webcam timestamps are sampled onto a 25-Hz grid without stretching every word to 29 frames. The received source sampling rate and duration are reported. Interpolation cannot recover missing visual information. Inputs are bounded to 12–76 real capture frames, 0.32–3.0 seconds, no inter-frame gap over 160 ms and at least 10 capture samples/second. These are heuristic operational limits, not calibrated recognition thresholds.

The browser requests at most two measured faces so it can reject multi-person frames. Zero/multiple faces, off-frame landmarks, poor brightness/contrast, static mouth and incomplete/overlong sequences prevent usable requests or yield explicit HTTP 422 diagnostics. CPU backpressure returns 429 and retains only the latest pending window. Landmark processing and frame capture remain on-device; the private backend transiently receives frames for inference, without saving camera recordings.

The eight-word model retains its original eye-axis crop and trained 29-frame preprocessing. It is not silently switched to the LRW transform. Alphabet preprocessing also remains separate.

## Reliability and interface

- Separate Alphabet, 8 words and 500 words controls; free speech remains unavailable.
- Actual model loading/ready/processing/unavailable status and capture-quality messages.
- Live preview and optional measured mouth overlay; start, stop, reset and clear-history controls.
- Current confirmed result, confidence/uncertainty, five real word candidates, and session history.
- Eight-word results require two accepted agreeing windows. The 500-word browser submits completed utterances only, with about 240 ms of timestamp-based preroll; the backend checks the full sequence and a second real view omitting its first/last frame. Both views must independently pass rejection and agree. This avoids spending CPU on partial-word requests. One history entry is permitted per utterance even if later candidates change. No accuracy improvement is inferred from this consistency rule.
- Face loss, mode changes and stop abort stale work and reset confirmation. Stop releases camera tracks. Reset stops capture and clears the session.
- Server processing time, round-trip time, process RAM, CPU time/use and request-rate information. These are measured values, not simulated telemetry.
- Responsive desktop/mobile layout; original charcoal/ivory/mint colors preserved. Science Lab status and workflow copy distinguish the measured eight-word model, publisher 500-word research path and unavailable sentence decoder.

Unknown-word rejection is configurable using local environment variables:

```powershell
$env:LIPNET_LRW_CONFIDENCE_THRESHOLD = '0.7'
$env:LIPNET_LRW_MARGIN_THRESHOLD = '0.15'
.\start_research_demo.ps1
```

These defaults were not tuned on the new diagnostic. Rejected responses keep the model candidate for inspection but return `word: null` and `display_word: Uncertain`. Confidence and margin do not establish vocabulary membership. The eight-word model keeps its separate, previously frozen rejection policy.

## Genuine execution and diagnostic results

Actual HTTP requests used the real private server and real checkpoints, without mocked predictions:

| Input | Actual outcome |
| --- | --- |
| Genuine Oxford AFTERNOON video; explicit caller-aligned fixed mouth crop | 500 probabilities and five candidates; top-1 AFTERNOON |
| Same genuine video; measured MediaPipe points, full-face JPEGs, mean-face alignment and capture timestamps | HTTP 200; top-1 AFTERNOON |
| Capture request declaring two faces | HTTP 422; no recognition claimed |
| Genuine historical MIRACL connection clip, re-tracked under the original eight-word preprocessing | HTTP 200; accepted connection, five candidates and eight probabilities |

Evidence: `.qa/expanded-word-real-api.json`, native exit proof `.qa/expanded-word-real-api-v3-exit.json`. Earlier QA attempts exposed an incorrect optional import and an attempt to find a quarantined utterance in a feature cache; these were corrected without replacing input data or model weights. Their failure logs remain retained. Repeated smoke executions are not additional independent accuracy samples.

The final completed-utterance implementation was exercised again against the actual HTTP server. The genuine AFTERNOON input was accepted by **both 29-frame and 27-frame temporal views**, at scores **0.92698** and **0.93379**, with **949.28 ms** server processing. The original eight-word connection regression passed again. These scores are confidence values for one public clip, not measured recognition accuracy. Proof: `.qa/expanded-word-completed-api.json` and `.qa/expanded-word-completed-api-exit.json`.

The subsequent Chrome camera replay passed **14 real-backend UI/lifecycle checks** and submitted one completed utterance: 20 measured source frames over 1,183 ms, maximum gap 83.4 ms. The two model views returned UNTIL (0.09684) and AFTERNOON (0.08762); they disagreed and were low-confidence. The result was correctly **Uncertain**, with no history entry. This validates capture, actual inference, rejection and lifecycle behavior, **not successful live recognition**. The replay used a genuine public clip with paced quiet intervals, not a physical webcam. Proof: `.qa/expanded-word-browser-v4.json`.

A **separate seven-source diagnostic** was executed with frozen publisher weights and thresholds:

| Sources | Actual result |
| --- | --- |
| One previously observed public AFTERNOON clip | Correct |
| Three MIRACL START clips from F10, F11 and M08 | Zero correct: F10 had fewer than eight native frames and failed explicitly; F11 was incorrectly accepted as OFTEN; M08 predicted ENOUGH and was rejected |
| Three MIRACL HELLO clips from those speakers; HELLO is absent from LRW vocabulary | All three rejected; this does not validate general unknown-word rejection |

Only **two of 500 known classes** and **four known samples** are represented. The diagnostic has **1/4 known clips correct**, top-5 **1/4**, one failed input retained in the denominator, and macro F1 **0.50 over supported classes**. These figures are available-source diagnostics, **not overall LRW accuracy**. They demonstrate that the model is not a reliable arbitrary-webcam recognizer. The live 500-word page shows this limitation; the eight-word fallback remains the default.

This frozen offline diagnostic uses the single-view classifier. It does not evaluate the later live two-view consistency rule; no improvement from that rule is claimed.

Full outputs, including precision/recall/F1 and correct/total counts for every class, predictions, 500 × 500 confusion CSV, sources and protocol:

- `runs/lrw500/available_samples_v1/{manifest.jsonl,PROTOCOL.md}`
- `runs/lrw500/available_samples_v1/evaluation/{evaluation.json,predictions.jsonl,confusion.csv}`

The Oxford example was already used for smoke testing; it is not an untouched test. Its speaker identity/separation from the publisher's training data cannot be verified locally. The MIRACL sources have authentic RGB images but unknown native FPS; their source order is preserved without inventing a frame rate. The three HELLO negatives do not cover arbitrary unseen words. No model fitting, threshold search or selection used these diagnostic rows.

## CPU measurements

Environment: Intel i3-1115G4, 7.74 GiB RAM, CPU-only PyTorch 2.2.2, one compute thread. Free RAM snapshots were below 1 GiB; paging/system contention materially affected timings. Model outputs and spatial resolution were not reduced to improve these measurements.

| Real API request | Server wall time | Process CPU time | Process RSS after request |
| --- | ---: | ---: | ---: |
| 500-word fixed-crop smoke after idle/paging | 9,233 ms | 1,125 ms | 224.7 MiB |
| 500-word mean-face capture path | 1,548 ms | 906 ms | 278.2 MiB |
| Eight-word fallback regression | 1,655 ms | 688 ms | 476.3 MiB |

Both loaded word models produced a measured process peak RSS of **573.5 MiB** in this run. The 500-word load plus discarded warmup measured **5.86 seconds**. Warmup is an initialization operation, not speech evidence.

Those table values describe the initial single-view measurements. The final direct two-view request measured **949 ms wall / 875 ms process CPU**, **655 MiB RSS afterwards**, and a **789 MiB process peak**. The browser replay under simultaneous landmark tracking measured **2,499 ms server / 2,684 ms round trip**, about **605 MiB RSS**, and still produced an uncertain result. These different scopes must not be combined into a real-time accuracy or throughput guarantee.

Three subsequent sequential requests repeating the same genuine AFTERNOON clip took **6.65 seconds total**, giving measured throughput **0.451 requests/second** and median round-trip **1,026 ms**. The first request took about 4.70 seconds; later requests were faster. Evidence: `.qa/expanded-word-throughput.json`. Repetition increases performance measurements, not recognition sample coverage. This hardware cannot guarantee continuous real-time inference, and these are not physical-webcam throughput measurements.

## Evaluation commands and safeguards

The executed seven-source diagnostic command was:

```powershell
.\.venv-avhubert-cpu\Scripts\python.exe -m pipeline.lrw_tcn.evaluate --manifest runs/lrw500/available_samples_v1/manifest.jsonl --output runs/lrw500/available_samples_v1/evaluation
```

That output is protected against overwrite. For an authorized new dataset, provide a new JSONL manifest and unused output path. Rows require unique IDs, `split: test`, an explicit boolean `unknown`, authentic-source provenance and the actual source SHA-256. Known labels must exactly match the ordered uppercase vocabulary; a supported label cannot be declared unknown. Repeated source content is rejected. Missing/corrupt/untrackable inputs remain visible as failures rather than disappearing from the denominator.

Supported offline input kinds are `aligned_mouth_npy`, standard `lrw_centered_video`, `raw_video` and authentic ordered `image_sequence`. The fixed-crop shortcut is explicitly a preprocessing approximation. Exact publisher-compatible landmarks/ROIs and an authorized complete LRW test set remain necessary before claiming a publisher-comparable benchmark. Historical publisher training-byte/speaker separation cannot be reconstructed from a new local manifest; it is reported unverified rather than inferred.

The full LRW corpus still requires the applicable approved BBC agreement/source. No protected download was bypassed and no URL or benchmark was invented. The author [LICENSE](pipeline/lrw_tcn/LICENSE) is retained; this integration is a separate private comparative research prototype, not commercial/operational deployment.

## Tests, files and remaining limits

The full Python suite passed **499 tests, 4 skipped**. After adding the completed-utterance rule, **27 relevant Python checks passed**, including two new agreement/disagreement cases; that is a targeted follow-up, not an additional full-suite count. The executed command was `.\.venv\Scripts\python.exe -m pytest -q tests/test_lrw500_live.py tests/test_lrw500_evaluation.py tests/test_lrw_tcn_inference.py tests/test_pretrained_word_service.py`. The latest production TypeScript/Vite build, **18 live logic tests** and **26 mathematics tests** passed. Native commands/exits and logs are in `.qa/expanded-word-validation-commands.json` and `.qa/completed-utterance-frontend-exits.json`. New checks cover publisher geometry, timestamp sampling/discontinuities, preroll duration, invalid face/lighting/static-mouth inputs, rejection, backpressure, HTTP validation, failed-input denominators, content duplication and history deduplication.

The initial actual browser run passed **14 UI/lifecycle checks** and reached the endpoint, but replay requests encountered backpressure/invalid capture windows. That prompted buffer resets at timestamp gaps and waiting for stale CPU requests instead of assuming fetch cancellation stops the model. A subsequent replay (`.qa/expanded-word-browser-v3.json`) received a real HTTP 200 partial-window response, but its candidate LATER at confidence 0.041 was correctly rejected. The final completed-utterance replay passed **14 checks**, received actual final-window inference and safely rejected the uncertain result, as detailed above. The interim v2 check loaded the legacy page during a build transition and is retained as failed QA evidence. A Chrome test-camera replay is **not physical webcam validation**. Physical-camera accuracy, natural utterance-boundary performance and general unknown-word rejection remain unmeasured.

Created or updated implementation files:

- `pipeline/lrw_tcn/{inference,preprocessing,diagnostics,evaluate}.py`, pinned `20words_mean_face.npy`, `SOURCE.json` and README.
- `app/lrw500_routes.py`, `app/services/{lrw500_inference,word_compute,pretrained_word_inference,recognition_capabilities}.py`, `app/main.py`.
- `frontend/src/lib/api.ts`, `hooks/useBackend.tsx`, `live-demo/{useLandmarker,useDemoPipeline,wordCapture}.ts*`.
- Workspace LiveDemo, RecognitionEvidence, scoped live CSS, Science Lab capabilities/workflow/legacy architecture copy and frontend test command.
- `tests/test_lrw500_live.py`, `tests/test_lrw500_evaluation.py`, `frontend/tests/word-capture.test.mjs`, `start_research_demo.ps1`.
- `pipeline/sequence_recognition/CONTINUOUS_REQUIREMENTS.md` and this report. Reproducibility drivers and raw proof remain in `.qa`.

Preservation verification compares 24 protected files, including original checkpoints, new pretrained weights, eight-word head, feature manifest, protocol, selection, rejection policy, evaluation and predictions. Evidence: `.qa/expanded-word-preservation.json`. No recognition checkpoint or frozen evaluation output was overwritten. Existing independent workspace work was retained.

Continuous recognition still needs a compatible supervised visual decoder, tokenizer/dictionary, exact utterance preprocessing, legitimately acquired sentence supervision, held-out word-error evaluation and adequate compute. See [continuous requirements](pipeline/sequence_recognition/CONTINUOUS_REQUIREMENTS.md). The current application does not fabricate sentences or convert isolated-word predictions into a claimed transcript.
