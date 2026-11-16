# Eight-word webcam collection and measured personal adaptation

Date: 30 October 2026. The existing eight-word model remains available. A new collection page and a complete import/feature/head/evaluation pipeline are implemented. **No new recognition accuracy is claimed: genuine user recordings have not yet been supplied for training.** The preserved MIRACL result remains 182/192 (94.79%) on its original stated protocol, not on the user's webcam.

## What the linked project actually does

Inspected repository: [allenye66/Computer-Vision-Lip-Reading-2.0](https://github.com/allenye66/Computer-Vision-Lip-Reading-2.0), pinned commit **e88f8a3e2c14cd25b44a0c180e1ebf3510790822**. The complete, nontruncated Git tree, source files and notebook code are saved in `.qa/allenye66-audit/`.

- Its collection script asks for a word label, detects mouth opening with dlib landmarks, retains a short lead-in, waits for mouth closure, and stores 22 frames at 80×112 RGB. It applies contrast/filter processing. This is a heuristic collection aid, not proof of reliable word boundaries. [Collection source](https://github.com/allenye66/Computer-Vision-Lip-Reading-2.0/blob/e88f8a3e2c14cd25b44a0c180e1ebf3510790822/data_collection/collect.py), [constants](https://github.com/allenye66/Computer-Vision-Lip-Reading-2.0/blob/e88f8a3e2c14cd25b44a0c180e1ebf3510790822/data_collection/constants.py).
- The training notebook has **13 classes**: a, bye, can, cat, demo, dog, hello, here, is, lips, my, read, you. Only **hello** overlaps our eight labels. The network is a Keras Conv3D classifier taking `[22,80,112,3]`, with 13 softmax outputs. Its head/weights cannot replace our 8-output AV-HuBERT classifier. [Notebook](https://github.com/allenye66/Computer-Vision-Lip-Reading-2.0/blob/e88f8a3e2c14cd25b44a0c180e1ebf3510790822/training/3DCNN.ipynb).
- The stored notebook uses random `train_test_split(..., test_size=0.2, random_state=42)`, producing **544 training and 137 other clips**. Those same 137 clips are supplied as validation during fitting and then used for the final classification report. No speaker/session grouping is present in that split. The final saved epoch shows 90.44% training and 97.08% validation accuracy; a later saved balanced-accuracy output is 0.97389277. These are not a reproduced independent test or unseen-speaker result.
- Repository text reports several different numbers: its About text says 99.2% testing, while its README describes 98.5% validation and 95.7% training. The README also says the larger showcased model's weights were not uploaded. These claims should not be substituted for the inspected notebook outputs. [README](https://github.com/allenye66/Computer-Vision-Lip-Reading-2.0/blob/e88f8a3e2c14cd25b44a0c180e1ebf3510790822/README.md).

The useful transferable idea is **collect repeated, correctly labelled words using the same camera and participant who will use the demo**. This project implements that idea with its existing MediaPipe alignment, explicit manual boundaries, local data storage and stricter session separation. It does not copy the upstream source, introduce TensorFlow/dlib, adopt its frame count, or claim to reproduce its reported accuracy.

## Upstream data acquired and its limits

`scripts/import_allenye66_reference.py` downloaded the **48 available matching hello MP4 files**: **413,636 bytes** total. Every file matched its pinned Git blob SHA-1 and size; local SHA-256 values were recorded, and all decoded as 22×80×112×3. The tree contains 49 hello `data.txt` entries, but only 48 hello MP4 files; those counts are not interchangeable.

Files are under `data/external/allenye66_lipreading2_reference/`, with `inventory.jsonl` and `acquisition.json`. Downloads are cached and retried; reruns verify existing bytes. They are recorded as **reference/quarantine**, with `training_clips_added: 0`. No speaker/session IDs, source full-face video, compatible eye-axis crop provenance or repository reuse license were found in the inspected tree. Their prefiltered mouth crops cannot reconstruct our measured 96px alignment. They are not renamed into our other seven classes, silently stretched into the training manifest, redistributed with source, or presented as a complete new eight-word dataset.

## Record your actual eight words

Open **http://127.0.0.1:8001/demo/collect** and refresh once if the route was already open during a build. It is part of the existing app and is linked from the sidebar and the live eight-word panel.

Words: **choose, connection, hello, navigation, next, previous, start, stop**.

1. Keep the same anonymized Participant ID for yourself. The default is `participant01`.
2. In the Training partition, use a training Session ID, give recording consent, and start the camera. No microphone is requested.
3. Pick a prompted word, click **Record Word**, say the complete word naturally, and click **Finish Word**. Review the actual captured mouth frames. Confirm that you spoke the displayed word, then click **Save labelled clip**. Discard a wrong, interrupted or incomplete attempt.
4. Record **15 saved training examples per word**. Vary word order, normal speaking speed, modest head/camera position and lighting. Do not record the same video repeatedly as separate examples.
5. Stop the camera. Take a break or change the recording setup, select Validation, and keep its newly generated separate Session ID. Record **5 examples per word**.
6. In another separate session, select Test and record **13 examples per word**. Keep these clips out of all training/model-selection decisions.
7. Click **Export training bundle** after each session and keep the downloaded JSON files. The final export contains all saved clips from this browser, so it is sufficient by itself.

This gives **120 training + 40 validation + 104 test = 264 user-confirmed recordings**. The test quota also satisfies the existing research service's minimum 100 evaluated clips; it does not guarantee that a candidate will pass its quality checks. Multiple participants are supported only when each has complete quotas and different sessions in all three partitions.

Recordings use the current eye-axis/mouth-centered 96×96 convention and actual capture timestamps. The word buffer starts empty, accepts 8–76 frames, and rejects a five-second overrun rather than inventing a word boundary. A saved clip carries the participant, session, partition, prompt label, explicit confirmation/consent, actual recording date, frame fingerprint and actual landmark-model SHA-256. There are no model-generated labels.

Data are kept in **IndexedDB for this browser/origin**, not a new backend database. Save persists across refresh/navigation; browser storage can still be cleared or evicted, so export is essential. Changing ports/browsers does not transfer IndexedDB data. No collection clip is uploaded and no prediction request is made from this page. The active camera is released on Stop and navigation. Dataset settings are locked during capture/review and while the camera is active.

## Train, compare and preserve the baseline

From the repository root, after the final JSON is downloaded:

```powershell
$bundle = Get-ChildItem -LiteralPath (Join-Path $env:USERPROFILE 'Downloads') -Filter 'lipnet-8words-*.json' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $bundle) { throw 'Export your actual eight-word recordings first.' }

.\.venv-word-recognition\Scripts\python.exe scripts/train_personal_words.py --bundles $bundle.FullName --run runs/word_accuracy/personal8_v1 --through evaluate
```

Choose a new run name for a changed experiment; completed model/evaluation files are protected. The wrapper uses the existing word environment for import/head fitting/evaluation and the existing `.venv-avhubert-cpu` environment for frozen features. It records exact commands, native exit codes and durations under `runs/word_accuracy/personal_logs/`. No package install or GPU is needed.

The stages are:

1. **Import:** verify real export format, consent/label confirmation, timestamps, content fingerprints, JPEG dimensions and local landmark fingerprint. Coalesce identical repeated exports by recording ID; reject conflicting copies, duplicate decoded clips, missing class quotas and sessions crossing partitions. Require at least 2 GiB free disk before preparation. Do not accept the upstream reference geometry or marked QA fixture exports.
2. **Extract:** use the unchanged official frozen AV-HuBERT encoder at one CPU thread with the existing low-memory loader. Interpolate the complete clip to 29 frames, center crop to 88×88, normalize using mean 0.421/std 0.165, and cache actual layer-9 `[29,768]` features. Verify cache/source/encoder fingerprints on reuse. The default extraction budget is 30 minutes per invocation; completed feature receipts survive a timeout and the same command resumes them. Actual completion time depends on the laptop and other running programs.
3. **Train:** keep layer 9 and mean/std pooling fixed. Fit a lightweight eight-way logistic head using only personal training clips plus the preserved genuine MIRACL **training** features. Training-only scaling, class balancing, C∈{0.1,1,10} and personal sample weights∈{1,4,10} produce nine candidates. Include the preserved model itself as a tenth candidate. Select using personal validation accuracy, then validation log loss. Historical MIRACL test clips and personal test features never select a candidate.
4. **Evaluate once:** freeze the selected head/selection first, then compare selected versus preserved heads on exactly the same personal test clips. Save raw predictions, top-1/top-5, macro F1, per-word correct/total/precision/recall/F1, single-token substitution error, and paired confusion matrices in CSV/PNG/SVG. Evaluation can resume only with the same frozen selection/features after an interrupted output stage; a completed evaluation cannot be overwritten or used for another selection pass.

Main outputs are `personal_protocol.json`, `recordings.jsonl`, `features.jsonl`, `head.pt`, `selection.json`, `rejection_policy.json`, `evaluation.json`, `baseline_evaluation.json`, `test_predictions.jsonl`, `confusion.csv`, `baseline_confusion.csv` and `paired_confusion.{png,svg}`. Rejection thresholds are inherited without retuning on test data and remain uncalibrated; no new OOV-rejection claim is made.

The encoder is **frozen**. This is personal downstream-head adaptation, not end-to-end encoder fine-tuning. A validation improvement alone is not a test result. A new model can perform worse on test; the report preserves that outcome instead of replacing it with a better-looking number.

## Selecting a completed personal head

No current server, alphabet selection, eight-word baseline, 500-word model, checkpoint or historical evaluation is automatically changed. The personal comparison must set `eligible_for_manual_research_activation: true`: at least 100 test examples, at least 70% top-1, every word recall at least 20%, and no lower correct count than the preserved baseline on those same clips. These are existing research readiness checks plus the paired nonregression check, **not a guarantee of 95% accuracy**.

After reviewing that actual result, use the existing launcher on a **free** port:

```powershell
.\start_word_demo.ps1 -Port 8003 -Head runs/word_accuracy/personal8_v1/head.pt
```

This serves the same application with the explicitly selected evaluated head; it is not a separate frontend. The original server on 8001 remains available. Omitting `-Head` still uses the preserved original model. A custom head without a passing paired report/hash is rejected. Health now identifies personal cross-session evaluation and the live notice shows that head's actual test correct/total rather than attributing the old MIRACL score to it. Existing running servers see backend-source additions on their next normal restart.

## Scientific scope and next step

This experiment measures recognition on **new recording sessions of the same participant(s)**, after training on their own recordings. It is deliberately personalized. It does not establish unseen-speaker accuracy, general English, automatic word boundaries, continuous speech, calibrated confidence, or reliable rejection of unknown words. No 95% result can be promised before the locked evaluation. The linked repository's random-clip numbers and our prior speaker-held-out MIRACL result measure different protocols.

The remaining data-dependent work is to receive the actual exported JSON, extract its real features, fit candidates and run the locked comparison. No genuine personal head or new accuracy result exists yet.

## Implementation and reproducibility index

- Collection: `frontend/src/workspace/pages/WordCollection.tsx`, `frontend/src/live-demo/wordCollection.ts`, `frontend/src/styles/word-collection.css`.
- Existing-app integration: router, workspace title/sidebar, live-page link and active-head scope notice; existing alphabet, eight-word and 500-word choices remain.
- Data/learning: `pipeline/word_recognition/personal_data.py`, `pipeline/word_recognition/personalized.py`, `scripts/train_personal_words.py`.
- Reference acquisition: `scripts/import_allenye66_reference.py`; raw acquisition receipt `.qa/allenye66-reference-acquisition.txt`.
- Optional evaluated-head selection: `start_word_demo.ps1`, added optional health metadata in `app/services/pretrained_word_inference.py` and the typed frontend health contract.
- Tests: `tests/test_personal_words.py`, `frontend/tests/word-collection.test.mjs`, `scripts/check_word_collection.mjs`. Browser fixture exports are explicitly marked `qa_fixture: true` and are rejected for training.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
Push-Location frontend
npm run test:live
npm run test:math
npm run build
Pop-Location
node scripts/check_word_collection.mjs
node scripts/check_live_assistant.mjs
```

## Executed validation

| Check | Actual result | Receipt |
| --- | --- | --- |
| Full Python suite, `.\.venv\Scripts\python.exe -m pytest -q` | **546 passed, 4 skipped**, 80.93 s | `.qa/personal-word-python-full.txt` |
| New pipeline cases in the actual `.venv-word-recognition` training environment | **20 passed**, 30.05 s | `.qa/personal-word-training-env-tests.txt` |
| Frontend `npm run test:live` | **37 passed** | `.qa/personal-word-live-tests.txt` |
| Frontend `npm run test:math` | **26 passed** | `.qa/personal-word-math-tests.txt` |
| `npm run build` | TypeScript and production Vite build passed; final Vite stage 10.12 s | `.qa/personal-word-build.txt` |
| `node scripts/check_word_collection.mjs` | **25 browser checks passed** | `.qa/word-collection-browser.json` |
| Existing `node scripts/check_live_assistant.mjs` | **90 browser checks passed** | `.qa/personal-word-live-browser-log.txt`, `.qa/live-assistant-browser.json` |
| Genuine saved-input eight-word HTTP request | `connection`, HTTP success, score 0.9990013242, not rejected; original evaluation protocol identified | `.qa/personal-word-runtime-smoke.json` |
| Protected-model preservation | All five checkpoint SHA-256 values match the pre-existing receipt | Same runtime receipt |
| Pinned upstream acquisition | 48 matching MP4s, correct Git blob size/hash, local SHA-256 and 22×80×112 decode | `data/external/allenye66_lipreading2_reference/acquisition.json` and `inventory.jsonl` |

The new Python cases cover actual bundle ingestion, repeated-export coalescing, label/consent/geometry/hash failures, duplicate decoded content, session leakage, quotas, resumable verified feature caching, training-only feature access, real logistic fitting/artifact plumbing on explicit synthetic fixtures, paired metrics/matrices, and refusal to retune a completed evaluation. Synthetic fixture model scores are **not recognition results**.

Collection browser checks use a genuine public video replay in an isolated Chrome profile, real MediaPipe tracking, and software label fixtures. They verify explicit consent, minimum capture frames, review/confirmation, duplicate Save protection, IndexedDB persistence, the actual downloaded JSON, partition/session separation, no prediction/upload POSTs, mobile layout, and track release during navigation. Those exports are marked `qa_fixture: true` and rejected for training. Screenshots: `.qa/word-collection-{desktop,mobile}.png`. No physical camera was activated by these checks.

One early browser-check failure was a quoting error in its navigation expression; the harness was corrected. Port 8001 was subsequently found unavailable; the baseline AV-HuBERT eight-word configuration was restored on the free port, and actual readiness/inference was rechecked. The pre-existing 8002 process was left running. One full live-browser attempt correctly rejected LRW capture at roughly 9 Hz under load, below that model's existing timing requirements; the unchanged assertions passed on the subsequent run. The 160 ms/minimum-rate requirements were preserved, not relaxed to conceal that resource limitation.

No personal recordings were found in Downloads during this implementation. The real user-data training/evaluation stages remain pending; no new head was activated and no higher recognition accuracy has been established.
