> Latest integration, 2026-10-29: separate selectable eight-word and publisher 500-word research modes are connected to the private demo on port 8002. The eight-word 94.79% result remains frozen. The 500-word available-source diagnostic and camera replay expose substantial limitations; no LRW-wide or webcam accuracy is established. See [current implementation, measured results and commands](EXPANDED_WORD_RECOGNITION.md). Historical findings below are retained.

> Latest whole-word result, **2026-10-29**: a separate frozen AV-HuBERT eight-word head achieved **182/192 (94.79%)**, versus **24/192 (12.50%)** for the preserved scratch model on the same clips. A private research Word Mode runs separately on port 8001. A genuine publisher-trained 500-word LRW model also passed one research CLI smoke; its local accuracy has not been benchmarked. See [measured results, scope and commands](WORD_ACCURACY_IMPROVEMENT.md). Historical results below remain unchanged. This does not establish arbitrary-English, webcam or sentence accuracy, and the eight-word head is incompatible with the 26-class alphabet adapter.

# LipNet-AI model capabilities audit

Current **2026-10-29** update: MIRACL acquisition and a separate eight-word pilot
are complete, with **8/48 (16.67%)** held-out top-1 accuracy. It remains offline
research; continuous visual transcription remains unavailable. An existing
experimental four-word endpoint is now present, without establishing LRW-500
recognition. See [current multi-dataset results](MULTIDATASET_RECOGNITION_REPORT.md)
for the checkpoint audit (12 primary artifacts plus two late baseline copies),
mappings, exact splits, real inference and blockers.
The dated audit below is historical.

Subsequent **2026-10-29** experiment: a separate 3D-CNN/BiGRU was trained on four complete GRID command words and evaluated at **8/32 (25%)**, with single-class collapse. See [AUTOMATED_WORD_PIPELINE_REPORT.md](AUTOMATED_WORD_PIPELINE_REPORT.md). This does not validate LRW-500, sentence recognition, or a deployed word endpoint; the dated audit below remains a historical snapshot.

Audit date: **2026-10-28, Asia/Calcutta**. Scope: the current local repository, saved artifacts, class mappings, inference routes, and Science Lab claims. No training was started, checkpoints overwritten, backend selection changed, or existing server stopped.

**Working dual-mode letter/500-word recognition is not established.** The alphabet CNN–BiLSTM is executable. Three experimental 500-output CNN–BiLSTM artifacts are also loadable, but genuine LRW training and a connected word endpoint are unverified/absent. GRID sentence data and AV-HuBERT encoder weights do not supply functioning sentence or whole-word recognition.

Port 8000 was not listening during this audit: the initial network request failed with `WinError 10061`. The prediction route was subsequently exercised with the real FastAPI application through **in-process ASGI TestClient**, without starting or replacing the deployed backend. An ASGI 200 response is not a claim that a deployed network server is currently online.

## Capability table

| Mode | Architecture | Vocabulary size | Checkpoint | Endpoint | Verified capabilities | Evaluation evidence | Limitations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Isolated letters | Per-frame Conv2D 16/32/64 + BN/pooling → 1-layer BiLSTM, 96 units/direction → Linear 192→128 → ReLU/dropout → Linear 128→26 | 26 outputs; index 0=A through 25=Z; **no blank** | `models/alphabet_classifier.pt` | `POST /api/predict` | Strict CPU load; 913,274 parameters; input `[1,30,1,140,46]`; output `[1,26]`; finite scores; real-input ASGI call returned 200 and 26 probabilities | One alignment-derived GRID **F** segment returned **Y**, not F. Archived summary records 12% on 100 test samples, but was not reproduced | Experimental, uncalibrated clip classification. No verified accuracy for A–Z, unseen speakers, or physical webcams. Original checkpoint contains no dataset/speaker/split provenance or class-count evidence |
| LRW whole-word experiment | The same per-frame **2D CNN–BiLSTM**, with a 500-output final linear layer; **not AV-HuBERT, ResNet, or a 3D CNN** | 500 output indices; current external file-order mapping | Three files under `runs/finetune/lrw_500words/`, listed below | **No connected word prediction endpoint** | Each file strictly loads as `AlphabetCNNBiLSTM(num_classes=500)`; 974,420 parameters; `[1,29,1,140,46] → [1,500]`; finite forward pass on real video | Stored one-epoch summary reports validation accuracy 0.0666667; not an authenticated LRW evaluation. No genuine 500-class test run was performed | Local 45 videos are byte-identical generated fixtures with 15 labels and cross-split duplication. Training provenance, vocabulary fingerprint, genuine LRW coverage, and recognition quality are not established |
| AV-HuBERT pretrained representation | Official `AVHubertModel`, video frontend + Transformer; frozen CPU video-only extraction | **768 features per frame**, not 768 or 500 recognition classes | `models/external/avhubert/base_vox_iter5.pt` | Feature extraction Python API/CLI; **no ready supervised head** | Official loader succeeded; 103,330,280 parameters; real GRID clip produced finite `[29,768]` features | Feature extraction only; no class prediction or accuracy measured | Pretraining cluster embeddings/projection are not a supervised letter/word/sentence decoder. `models/avhubert_head.pt` and trained head runs are absent |
| Letter-sequence CTC research | Separate `MultiTaskAlphabetModel`: Conformer over fused 192-D features + temporal letter head | 27: blank=0, A=1 … Z=26 | **No compatible `best.ckpt`/`latest.ckpt` found** | Research service/local pipeline source exists; browser HTTP endpoint does not use it | Head, CTC loss, greedy repeat-collapse/blank removal, and tests exist | Unit/fixture tests pass; no trained-artifact recognition test | Source-code availability is not trained inference. No sentence-space/digit/word vocabulary is supplied by this letter head |
| GRID sentences | Dataset ingestion and word alignment code; no identified trained sentence recognizer | 51 observed lexical tokens in six grammar positions; **not a model output vocabulary** | **No sentence checkpoint found** | **No sentence prediction endpoint** | 1,000 video/audio/word-alignment triples from one speaker; manifest and grammar inspected; real video decoded/tracked | Corpus inventory and encoder smoke only; no sentence error rate/word error rate measured | One speaker cannot establish speaker-disjoint evaluation. The observed letter slot omits W. Dataset availability does not implement sentence decoding |
| Face/mouth tracking (auxiliary) | MediaPipe FaceLandmarker bundle; eye-axis alignment and mouth crop | Facial landmark coordinates, **not speech classes** | Two `.task` copies listed below | Browser tracking / Python preprocessing; no speech classification | Real GRID clip yielded tracked crops for all 75 frames | Tracking smoke only; not a recognition score | Tracking weights cannot replace any letter, word, or sentence classifier |

## Checkpoint inventory and actual model definitions

The repository walk excluded virtual environments, dependency folders, Git metadata, caches, and `.qa` test outputs. It found **five recognition/representation checkpoint files**, **two auxiliary `.task` bundles**, and **90 dataset tensor files** under `data/processed/`. The dataset tensors are input/cache artifacts, not trained checkpoints. No other `.pth`, `.ckpt`, `.safetensors`, `.onnx`, `.pkl`, `.pickle`, `.joblib`, `.bin`, `.h5`, `.hdf5`, `.keras`, `.tflite`, or `.pb` model artifacts were found in project-owned files.

| File | Bytes | Class head / artifact metadata | SHA-256 |
| --- | ---: | --- | --- |
| `models/alphabet_classifier.pt` | 3,664,980 | `fc2.weight=[26,128]`; config `{num_classes:26,t:30,h:140,w:46}`; BN training-batch counter 2,300 | `7c00d5117b7fae9a76c05980ca70edd680e79be4c8a152ea2595c98ded90c4c3` |
| `runs/finetune/lrw_500words/alphabet_classifier.pt` | 3,909,167 | `fc2.weight=[500,128]`; config `{num_classes:500,t:29,h:140,w:46}`; epoch 1; BN counter 8 | `9cef5deae4ed099d4b368a27e9c66b577df5e4d54f797e51e0e4d6340639b198` |
| `runs/finetune/lrw_500words/best_model.pt` | 3,908,834 | `fc2.weight=[500,128]`; epoch 1; val loss 5.975058841705322; stored val accuracy 0.06666666666666667; BN counter 2,315 | `c44440ad00bcc3be76ceb2f89f948f35aa99fe3191ab9fe85bfea35ef912fd2e` |
| `runs/finetune/lrw_500words/checkpoint_epoch_1.pt` | 11,724,879 | Same model weights as `best_model.pt`, plus optimizer state; epoch 1; train loss 6.139615376790364 | `5afe9d65f6bf54b2728037bf4f3227fff62e4b0a24e7e889b4aab5a563919d8a` |
| `models/external/avhubert/base_vox_iter5.pt` | 1,240,185,329 | 768-D encoder; pretraining `label_embs_concat=[2004,256]`, `final_proj.weight=[256,768]`; these are **not** LRW class mappings | `dee8b0651dbea3a21cda0f2c52b956e3a878d5f6ee503d427d6a4044bb95c9cc` |

The word file named `alphabet_classifier.pt` has different learned tensors from the other two word files. They must not be presented as three independently validated recognizers, or interchangeable exports of one proven model. A batch counter and an epoch field are training-state evidence, not proof of dataset authenticity or class coverage.

`models/alphabet_arch.txt` is stale: it describes 2,754,906 parameters, Conv 32/64/128 and a two-layer 128-unit BiLSTM. The current code and strict checkpoint load instead establish **913,274 parameters, Conv 16/32/64, one 96-unit bidirectional LSTM**. The stale text file was preserved; it was not used as architecture evidence.

### Auxiliary tracking bundles

- `models/face_landmarker.task`: 3,758,596 bytes; SHA-256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
- `frontend/public/models/face_landmarker.task`: 3,758,596 bytes; SHA-256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.

The Python smoke used the frontend/public landmark bundle. The matching hashes identify whether the other copy contains the same bytes; neither bundle predicts speech labels.

## Alphabet inference and decoding

`BrowserInference.load()` selects the `alphabet` backend by default and strictly instantiates `models.alphabet_model.AlphabetCNNBiLSTM(num_classes=26)`. The audit process had no `LIPNET_MODEL_BACKEND` override. An explicit `avhubert` selection would require a separately trained compatible A–Z head, which is absent; no selection was changed here.

The actual path is JPEG crops → grayscale → resize to 46×140 → linear temporal interpolation to 30 frames → pixel scaling by 255 → per-frame Conv2D → BiLSTM → **last time-step features** → dense head → **one `[B,26]` logit vector per clip** → softmax/top-three A–Z mapping. A recurrent temporal encoder does not make this a temporal label-sequence predictor. There is no CTC blank, repeat collapsing, beam search, word composition, or sentence decoding in `/api/predict`.

Frontend EMA/consensus stabilizes successive clip-level distributions. It does not convert the classifier into CTC or prove improved accuracy. `/ws/inference` is a lifecycle-only transport and explicitly reports classification unavailable. The separate CTC implementation uses the different 27-class mapping in `alphabet/vocab.py`; it is not the deployed alphabet checkpoint's mapping.

## Vocabulary and LRW evidence

`data/vocab.txt` contains **500 non-empty entries and 500 unique entries**, with no duplicates. `data/schema.py` uppercases entries and preserves file order to produce `WORD_TO_IDX` / `IDX_TO_WORD`. Its ordered mapping matches the current file. The vocabulary SHA-256 is `c882e2ffcbf301d4145a33b753d3c70af36a1d9e08a0a1cd1f93c01145c421fb`.

The file proves the configured label list only. None of the three word checkpoints embeds the ordered vocabulary or a vocabulary hash, speaker IDs, source-video fingerprints, or an authenticated evaluation split. Their historical mapping cannot be independently tied to the present file merely because the output dimension matches.

Current LRW data: 15 train, 15 valid, 15 test videos, labeled ABOUT, ABSOLUTELY, ABUSE, ACCESS, ACCORDING, ACCUSED, ACROSS, ACTION, ACTUALLY, AFFAIRS, AFFECTED, AFRICA, AFTER, AFTERNOON, AGAIN. **All 45 have SHA-256 `33be06600bab2ee0d28aeca31c467074278e83ad31311e1526e2687e2412db8c`.** `scripts/setup_lrw_dataset.py:create_mock_video()` generates these moving-ellipse development inputs. This is conflicting-label/cross-split content duplication, not a 15-class or 500-class real-speech benchmark. The remaining 485 vocabulary classes have no local LRW video examples. No LRW recognition accuracy was evaluated in this audit.

`scripts/train_lrw_500words.py` currently requires an audited real-data manifest for normal training; synthetic training requires an explicit test option. The old one-epoch summary lacks that newer provenance field, so the precise historical training inputs cannot be certified from it. No training was rerun to fill that gap.

### Complete current vocabulary, in zero-based file order

Each range labels the following 20 successive indices. This is the actual local list, not a statement that every token is recognized.

```text
000-019: ABOUT, ABSOLUTELY, ABUSE, ACCESS, ACCORDING, ACCUSED, ACROSS, ACTION, ACTUALLY, AFFAIRS, AFFECTED, AFRICA, AFTER, AFTERNOON, AGAIN, AGAINST, AGREE, AGREEMENT, AHEAD, ALLEGATIONS
020-039: ALLOW, ALLOWED, ALMOST, ALREADY, ALWAYS, AMERICA, AMERICAN, AMONG, AMOUNT, ANNOUNCED, ANOTHER, ANSWER, ANYTHING, AREAS, AROUND, ARRESTED, ASKED, ASKING, ATTACK, ATTACKS
040-059: AUTHORITIES, BANKS, BECAUSE, BECOME, BEFORE, BEHIND, BEING, BELIEVE, BENEFIT, BENEFITS, BETTER, BETWEEN, BIGGEST, BILLION, BLACK, BORDER, BRING, BRITAIN, BRITISH, BROUGHT
060-079: BUDGET, BUILD, BUILDING, BUSINESS, BUSINESSES, CALLED, CAMERON, CAMPAIGN, CANCER, CANNOT, CAPITAL, CASES, CENTRAL, CERTAINLY, CHALLENGE, CHANCE, CHANGE, CHANGES, CHARGE, CHARGES
080-099: CHIEF, CHILD, CHILDREN, CHINA, CLAIMS, CLEAR, CLOSE, CLOUD, COMES, COMING, COMMUNITY, COMPANIES, COMPANY, CONCERNS, CONFERENCE, CONFLICT, CONSERVATIVE, CONTINUE, CONTROL, COULD
100-119: COUNCIL, COUNTRIES, COUNTRY, COUPLE, COURSE, COURT, CRIME, CRISIS, CURRENT, CUSTOMERS, DAVID, DEATH, DEBATE, DECIDED, DECISION, DEFICIT, DEGREES, DESCRIBED, DESPITE, DETAILS
120-139: DIFFERENCE, DIFFERENT, DIFFICULT, DOING, DURING, EARLY, EASTERN, ECONOMIC, ECONOMY, EDITOR, EDUCATION, ELECTION, EMERGENCY, ENERGY, ENGLAND, ENOUGH, EUROPE, EUROPEAN, EVENING, EVENTS
140-159: EVERY, EVERYBODY, EVERYONE, EVERYTHING, EVIDENCE, EXACTLY, EXAMPLE, EXPECT, EXPECTED, EXTRA, FACING, FAMILIES, FAMILY, FIGHT, FIGHTING, FIGURES, FINAL, FINANCIAL, FIRST, FOCUS
160-179: FOLLOWING, FOOTBALL, FORCE, FORCES, FOREIGN, FORMER, FORWARD, FOUND, FRANCE, FRENCH, FRIDAY, FRONT, FURTHER, FUTURE, GAMES, GENERAL, GEORGE, GERMANY, GETTING, GIVEN
180-199: GIVING, GLOBAL, GOING, GOVERNMENT, GREAT, GREECE, GROUND, GROUP, GROWING, GROWTH, GUILTY, HAPPEN, HAPPENED, HAPPENING, HAVING, HEALTH, HEARD, HEART, HEAVY, HIGHER
200-219: HISTORY, HOMES, HOSPITAL, HOURS, HOUSE, HOUSING, HUMAN, HUNDREDS, IMMIGRATION, IMPACT, IMPORTANT, INCREASE, INDEPENDENT, INDUSTRY, INFLATION, INFORMATION, INQUIRY, INSIDE, INTEREST, INVESTMENT
220-239: INVOLVED, IRELAND, ISLAMIC, ISSUE, ISSUES, ITSELF, JAMES, JUDGE, JUSTICE, KILLED, KNOWN, LABOUR, LARGE, LATER, LATEST, LEADER, LEADERS, LEADERSHIP, LEAST, LEAVE
240-259: LEGAL, LEVEL, LEVELS, LIKELY, LITTLE, LIVES, LIVING, LOCAL, LONDON, LONGER, LOOKING, MAJOR, MAJORITY, MAKES, MAKING, MANCHESTER, MARKET, MASSIVE, MATTER, MAYBE
260-279: MEANS, MEASURES, MEDIA, MEDICAL, MEETING, MEMBER, MEMBERS, MESSAGE, MIDDLE, MIGHT, MIGRANTS, MILITARY, MILLION, MILLIONS, MINISTER, MINISTERS, MINUTES, MISSING, MOMENT, MONEY
280-299: MONTH, MONTHS, MORNING, MOVING, MURDER, NATIONAL, NEEDS, NEVER, NIGHT, NORTH, NORTHERN, NOTHING, NUMBER, NUMBERS, OBAMA, OFFICE, OFFICERS, OFFICIALS, OFTEN, OPERATION
300-319: OPPOSITION, ORDER, OTHER, OTHERS, OUTSIDE, PARENTS, PARLIAMENT, PARTIES, PARTS, PARTY, PATIENTS, PAYING, PEOPLE, PERHAPS, PERIOD, PERSON, PERSONAL, PHONE, PLACE, PLACES
320-339: PLANS, POINT, POLICE, POLICY, POLITICAL, POLITICIANS, POLITICS, POSITION, POSSIBLE, POTENTIAL, POWER, POWERS, PRESIDENT, PRESS, PRESSURE, PRETTY, PRICE, PRICES, PRIME, PRISON
340-359: PRIVATE, PROBABLY, PROBLEM, PROBLEMS, PROCESS, PROTECT, PROVIDE, PUBLIC, QUESTION, QUESTIONS, QUITE, RATES, RATHER, REALLY, REASON, RECENT, RECORD, REFERENDUM, REMEMBER, REPORT
360-379: REPORTS, RESPONSE, RESULT, RETURN, RIGHT, RIGHTS, RULES, RUNNING, RUSSIA, RUSSIAN, SAYING, SCHOOL, SCHOOLS, SCOTLAND, SCOTTISH, SECOND, SECRETARY, SECTOR, SECURITY, SEEMS
380-399: SENIOR, SENSE, SERIES, SERIOUS, SERVICE, SERVICES, SEVEN, SEVERAL, SHORT, SHOULD, SIDES, SIGNIFICANT, SIMPLY, SINCE, SINGLE, SITUATION, SMALL, SOCIAL, SOCIETY, SOMEONE
400-419: SOMETHING, SOUTH, SOUTHERN, SPEAKING, SPECIAL, SPEECH, SPEND, SPENDING, SPENT, STAFF, STAGE, STAND, START, STARTED, STATE, STATEMENT, STATES, STILL, STORY, STREET
420-439: STRONG, SUNDAY, SUNSHINE, SUPPORT, SYRIA, SYRIAN, SYSTEM, TAKEN, TAKING, TALKING, TALKS, TEMPERATURES, TERMS, THEIR, THEMSELVES, THERE, THESE, THING, THINGS, THINK
440-459: THIRD, THOSE, THOUGHT, THOUSANDS, THREAT, THREE, THROUGH, TIMES, TODAY, TOGETHER, TOMORROW, TONIGHT, TOWARDS, TRADE, TRIAL, TRUST, TRYING, UNDER, UNDERSTAND, UNION
460-479: UNITED, UNTIL, USING, VICTIMS, VIOLENCE, VOTERS, WAITING, WALES, WANTED, WANTS, WARNING, WATCHING, WATER, WEAPONS, WEATHER, WEEKEND, WEEKS, WELCOME, WELFARE, WESTERN
480-499: WESTMINSTER, WHERE, WHETHER, WHICH, WHILE, WHOLE, WINDS, WITHIN, WITHOUT, WOMEN, WORDS, WORKERS, WORKING, WORLD, WORST, WOULD, WRONG, YEARS, YESTERDAY, YOUNG
```

## GRID inventory and grammar

The inspected manifest is `data/external/grid/s1/grid_manifest.jsonl`, resolving media relative to `data/external/grid/s1/corpus`. There are 1,000 manifest rows, 1,000 MPG videos, 1,000 WAV files, and 1,000 `.align` files. All referenced files exist. The single manifest speaker ID is `grid_T01` (directory s1). Stored validation statuses are all `ok`; that is not an independent full-media decoding or recognition evaluation.

After removing silence/pause tokens `sil` and `sp`, all 1,000 transcripts have this observed six-token grammar:

| Position | Observed tokens |
| --- | --- |
| Command | bin, lay, place, set |
| Color | blue, green, red, white |
| Preposition | at, by, in, with |
| Letter | a–v, x, y, z: 25 letters, **no w** |
| Number | zero, one, two, three, four, five, six, seven, eight, nine |
| Adverb | again, now, please, soon |

This gives 51 observed lexical tokens, not 500 LRW outputs and not 26 isolated alphabet test classes. Twenty-four stored manifest transcripts retain an `sp` short-pause token because the builder filters `sil` only. The audit normalized pause tokens in memory; it did not rewrite corpus annotations. Original alignment offsets use **25,000 units/second**, not milliseconds. `scripts/prepare_alphabet.py` documents the correct conversion; `pipeline/grid/manifest.py` misleadingly calls its raw fields `start_ms/end_ms`. Any future supervised segmentation must explicitly handle these units and short pauses.

No sentence checkpoint, trained sentence label mapping, grammar/CTC/seq2seq sentence decoder, compatible sentence artifact, or sentence endpoint was found. Source ingestion, grammar-conforming annotations, and a feature encoder do not complete that path. One speaker and missing W coverage also prevent a speaker-disjoint full A–Z evaluation from this subset alone.

## Real inference smoke and accuracy limits

Input: `data/external/grid/s1/corpus/T01/video/s1/bbaf2n.mpg`, SHA-256 `e468120039e208b5ff9b9e269f8dc96bbddc45701f8dddcf2c0ae21abc0df546`. Transcript: **“bin blue at f two now.”** The video has 75 frames at 25 FPS; all 75 yielded real tracked/aligned mouth crops.

The word alignment identifies F at units 35,500–41,000, corresponding to 1.42–1.64 seconds and source frame indices `[35,41)`. Six real crops were interpolated to eight JPEGs to satisfy the endpoint minimum; the backend then interpolated the sequence to its required 30 frames. The audit rejects dropped tracking frames before applying these source indices.

ASGI status **200**; predicted **Y** for annotation **F**; uncalibrated softmax score **0.1161996275**. Top three: Y=0.1161996275, O=0.1132275313, I=0.0717593431. The response contained 26 probabilities (sum 0.9999999264), uncertainty state `UNCALIBRATED`, and `46.86` ms of backend time in this run. This latency excludes the preceding tracking/cropping and is not a throughput guarantee.

This is one coarticulated, alignment-derived GRID letter segment, not a newly recorded isolated-letter trial. Training-speaker overlap, exact historical crop agreement, and benchmark independence are unknown. The result is **incorrect recognition**, even though loading, tracking, transport, and forward execution worked. There was no physical webcam test, new held-out accuracy, accuracy improvement, or statistically meaningful per-class evaluation.

Each of the three word artifacts was also forwarded on the real full GRID sentence with 29 frames. All produced finite `[1,500]` outputs; the input is outside their intended whole-word task, so the resulting word indices were recorded only as execution diagnostics, not scored as recognition. The official AV-HuBERT loader separately produced finite `[29,768]` features from the real clip on CPU in **13.807 seconds including loading, tracking, and encoding**. This is one measurement, not a real-time guarantee or a classifier test.

### Historical accuracy evidence, not reproduced

`models/training_summary.json` records 30 epochs, 800/100/100 train/validation/test samples, best validation accuracy 0.20, test accuracy 0.12, and test loss 2.9523. `models/training_curve.json` contains 30 epoch entries and repeats the 0.12 test value. These files contain no evaluated clip IDs, per-class sample counts, speaker assignments, per-sample predictions, original split manifest, or checkpoint fingerprint. The original processed `.npy` clips were not found. The stored A–Z per-class values therefore cannot be reconstructed into a verified confusion matrix or independent benchmark. A reported zero for W is not proof that W had any test examples.

The separate claimed **32/40 (80%) across A–D** could not be reproduced: no matching original 40 labeled inputs, protocol, split, or prediction report was found. It must not be substituted for the 12% archived JSON value, a current benchmark, or a live accuracy estimate.

## Exact verification commands and outcomes

Commands below run from the repository containing `app`, `models`, and `frontend`:

```powershell
.\.venv\Scripts\python.exe scripts/audit_model_capabilities.py --in-process
.\.venv-avhubert-cpu\Scripts\python.exe scripts/audit_model_capabilities.py --encoder-only --output .qa/model-capabilities-encoder.json
.\.venv\Scripts\python.exe -m pytest tests pipeline/avhubert/tests pipeline/grid/tests pipeline/eval/tests -q | Tee-Object -FilePath .qa/model-audit-tests.txt
```

Frontend commands from `frontend`:

```powershell
npm run build
npm run test:math
npm run test:live
npm run dev -- --host 127.0.0.1
```

Targeted browser verification from the repository root:

```powershell
$env:MATH_ONLY = '1'
$env:QA_SKIP_PRODUCTION = '1'
node scripts/frontend-qa.mjs
```

| Check | Outcome |
| --- | --- |
| Python regression suite | **424 passed, 3 skipped**, 57.35 seconds; skips require the absent optional `synthetic_dataset` fixture |
| Artifact/ASGI audit | Five model files inventoried/hashed; four CNN–BiLSTM strict loads and finite forwards; real ASGI prediction 200; hashes and audit-process backend selection unchanged |
| Official encoder CPU smoke | Passed; finite `[29,768]`; pretrained cluster heads inspected; no recognition performed |
| TypeScript/Vite build | Passed |
| Equation-rendering tests | **26 passed**; strict TeX rendering and accessible MathML |
| Live stabilization/socket tests | **10 passed** |
| Science browser checks | **44 passed** at 1366, 768, and 390 CSS pixels, including all 11 selectable architecture stages, mathematical controls, honest capability/teaching labels, and mobile header; no uncaught exceptions or failed resource responses |
| Production backend / physical camera | **Not tested online in this audit**; production browser pass explicitly skipped because port 8000 was offline; no camera was opened |

Raw evidence: `.qa/model-capabilities-audit.json`, `.qa/model-capabilities-encoder.json`, `.qa/model-audit-tests.txt`, `.qa/math-browser-report.json`, and `.qa/science-*.png`. `.qa` is ignored by Git. Tests using synthetic tensors check software contracts, not learned recognition.

The first audit attempt without `--in-process` failed because port 8000 was offline. A combined CPU-environment/TestClient attempt exposed that environment's missing `httpx2` test dependency. The working commands above deliberately use the regular environment for ASGI tests and the existing CPU environment for official Fairseq encoding. No packages, backend environment variables, or server processes were changed to conceal those failures. The ASGI OpenAPI generation emitted an existing duplicate SPA-fallback operation-ID warning; MediaPipe/PyTorch emitted deprecation/runtime notices. These did not prevent inference or the successful test suite.

Inspection used `Get-Content -Encoding utf8`, `rg --files`, `rg -n`, `Get-ChildItem`, `Get-NetTCPConnection -State Listen -LocalPort 5173,8000`, `Get-ChildItem Env:LIPNET*`, and read-only Python `torch.load(..., map_location='cpu', weights_only=True)`, strict state-dictionary loads, source-video decoding, and SHA-256 comparisons. No training, live/start, model-switch, checkpoint-write, or server-stop endpoint was called.

## Files inspected and Science Lab corrections

| Area | Exact inspected files / artifacts |
| --- | --- |
| Model artifacts | All five checkpoint paths and both `.task` paths above; `models/alphabet_model.py`, `models/feature_classifier.py`, `models/alphabet_arch.txt`, `models/training_summary.json`, `models/training_curve.json`, `runs/finetune/lrw_500words/training_summary.json` |
| Class mappings/configuration | `data/vocab.txt`, `data/schema.py`, `alphabet/vocab.py`, `config.yaml`, `configs/alphabet_dataset.yaml` |
| Inference/decoding | `app/main.py`, `app/services/browser_inference.py`, `app/services/inference_service.py`, `app/services/live_pipeline.py`, `app/services/project_evidence.py`, `alphabet/heads.py`, `alphabet/decoding.py`, `alphabet/losses.py`, `frontend/src/lib/api.ts`, `frontend/src/live-demo/useDemoPipeline.ts` |
| Data/preprocessing/training | `data/lrw_dataset.py`, `pipeline/lrw/loader.py`, `scripts/setup_lrw_dataset.py`, `scripts/preprocess_alphabet_clips.py`, `scripts/prepare_alphabet.py`, `scripts/train_alphabet.py`, `scripts/train_lrw_500words.py`, `tracking/mouth_canonicalization.py`, `src/video_training.py`; all 45 LRW source files hashed; 90 processed tensor paths inventoried |
| GRID | `pipeline/grid/schema.py`, `pipeline/grid/manifest.py`, `pipeline/grid/builder.py`, `pipeline/grid/README.md`, `data/external/grid/s1/grid_manifest.jsonl`; all 1,000 manifest media references checked; `data/external/grid/s1/corpus/T01/alignment/align/bbaf2n.align` and `bbiz2p.align`; real `bbaf2n.mpg` decoded/tracked |
| AV-HuBERT/evaluation | `pipeline/avhubert/loader.py`, `pipeline/avhubert/classification.py`, `pipeline/eval/classification.py`; official vendored AV-HuBERT/Fairseq loading and video-only extraction executed |
| Existing audit context | `MODEL_AVAILABILITY.md`, `ACCURACY_PIPELINE.md`, `scripts/verify_local.py`, `scripts/_audit_models.py`, `scripts/_smoke_test_models.py`, `tests/test_backend_contract.py`, `tests/test_lrw_vocab.py`, `tests/test_inference_extended.py`, `tests/test_data_source_gate.py`, `tests/test_services.py`; full executed test-file list is in `.qa/model-audit-tests.txt` |
| Science Lab | `frontend/src/science/SciencePage.tsx`, all files under `frontend/src/science/components/`, `frontend/src/science/science.css`; equation component/catalog/styles and `scripts/frontend-qa.mjs` |

Science changes preserve the existing layout, palette, interaction, animations, and mathematical typesetting. They add a dated capability inventory; remove the false AV-HuBERT/3D-CNN stage from the LRW checkpoint diagram; distinguish 26-class clip classification from 27-class CTC research; mark all probability examples, confusion counts, and synthetic loss curves as invented teaching material; remove claims that those examples are calibrated/GRID measurements; correct corpus counts, normalization, data-split status, and unavailable-mode descriptions; and disclose the failed real F recognition instead of presenting execution as accuracy.

## Remaining blockers

- Authentic labeled LRW clips and an audited speaker/content-disjoint evaluation spanning the claimed word classes; embedded/fingerprinted class mapping; a deliberately implemented word prediction contract before any word-mode claim.
- Authentic isolated A–Z recordings, including W, across independent speakers; recoverable split IDs and per-sample predictions. Historical summaries alone cannot supply these.
- A supervised AV-HuBERT head trained and evaluated on eligible data; encoder features alone cannot be activated as class predictions.
- A trained compatible sequence/sentence model and complete decoder/vocabulary/endpoint integration before claiming continuous CTC or GRID sentence recognition.
- Physical webcam trials with ground truth, timing, face-loss recovery, multiple utterances, poses, speeds, lighting, and speakers. No fabricated trial or gain replaces this missing evidence.

No new recognition accuracy or improvement has been established. The verified outcome is an artifact-based capability boundary, reproducible execution checks, and corrected Science Lab claims.
