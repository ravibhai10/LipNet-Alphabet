# Executed commands and reproducible next steps

Working directory for all relative commands:
`C:\Users\Dell\Downloads\LipNet-Alphabet-main\LipNet-Alphabet-main`.
Date: 2026-10-29. Commands below were executed; log files retain errors as well
as successful results. No full LRW/TCD/MuAViC acquisition or pretrained word-head
training command was run.

## Audit, source inspection and acquisition

```powershell
git status --short
git clone --depth 1 https://github.com/facebookresearch/muavic.git external/muavic_reference
.\.venv-avhubert-cpu\Scripts\python.exe scripts/multidataset.py preflight --output .qa/multidataset-preflight.json
.\.venv-word-recognition\Scripts\python.exe scripts/multidataset.py preflight --output .qa/multidataset-before-pilot.json
.\.venv-word-recognition\Scripts\python.exe -u scripts/multidataset.py acquire-miracl --minutes 10
.\.venv-word-recognition\Scripts\python.exe -u scripts/multidataset.py ingest --dataset miracl --root data/external/miracl/original --output data/processed/multidataset/miracl_v1.jsonl
.\.venv-word-recognition\Scripts\python.exe -u scripts/multidataset.py ingest --dataset miracl --root data/external/miracl/original --pilot-repetitions 2 --output data/processed/multidataset/miracl_pilot_sources_v1.jsonl
.\.venv-word-recognition\Scripts\python.exe scripts/multidataset.py ingest --dataset grid --root runs/word_recognition/grid_commands_cpu_20261029/words.jsonl --output data/processed/multidataset/grid_commands_v1.jsonl
.\.venv-word-recognition\Scripts\python.exe scripts/multidataset.py avhubert-manifest --manifest data/processed/multidataset/grid_commands_v1.jsonl --output data/processed/multidataset/grid_words_avhubert_v1.jsonl
```

Official web pages listed in the report and the actual MuAViC source files were
inspected. PowerShell `Get-Content`, `Get-ChildItem`, `Get-CimInstance`, `rg`, and
`rg --files` inspected the exact repository paths enumerated in the report.
Small Python probes followed only the author-published public Drive confirmation,
read HTTP headers/signature, and inspected the archive member names/counts.
Network-probe and preflight JSON evidence is under `.qa`.

## Real preparation, training and prediction

```powershell
# First preparation stopped with four tracking failures; it trained nothing.
.\.venv-word-recognition\Scripts\python.exe -u -m pipeline.multidataset.experiment --manifest data/processed/multidataset/miracl_pilot_sources_v1.jsonl --run runs/multidataset/miracl8_cpu_v1 --repetitions 2 --epochs 12 --minutes 8

# New run: explicit training-only exclusions; same prespecified held-out inputs.
.\.venv-word-recognition\Scripts\python.exe -u -m pipeline.multidataset.experiment --manifest data/processed/multidataset/miracl_pilot_sources_v1.jsonl --run runs/multidataset/miracl8_cpu_v2 --repetitions 2 --epochs 12 --minutes 8 --allow-training-exclusions --cache-root runs/multidataset/miracl8_cpu_v1/clips

# Fresh real RGB-frame tracking + strict model loading + inference.
.\.venv-word-recognition\Scripts\python.exe -m pipeline.multidataset.predict --run runs/multidataset/miracl8_cpu_v2 --manifest data/processed/multidataset/miracl_pilot_sources_v1.jsonl --id miracl:F10:words:03:01
```

The prediction command was executed through a Python `subprocess.run` driver
capturing stdout/stderr and the native return code: **0**. Evidence:
`.qa/miracl-real-inference.json` and `.qa/miracl-real-inference-stderr.txt`.
Its candidate was wrong and rejected; see the report.

PowerShell's direct `*>` redirection formatted MediaPipe stderr warnings as
`NativeCommandError` in the preparation/training logs. The second training log
reaches epoch 12, writes the complete training/evaluation artifacts and prints
the final metrics; its shell wrapper reported exit 1. That shell result is not
presented as a captured successful native Python return code. The fresh
subprocess-based inference and subsequent tests independently loaded the result.
Future canonical orchestration captures subprocess streams explicitly to avoid
mistaking diagnostic stderr for recognition or training failure.

Do not rerun completed experiments over these paths. Choose a new run for a
different experiment; use `--resume` only for an interrupted matching contract.
Completed held-out evaluations are protected. Next bounded run, if desired:

```powershell
.\run_multidataset.ps1 -Train -Run runs/multidataset/next_prespecified_pilot
```

This reuses/reverifies existing acquisition and normalized manifests. It does
not add missing labels, activate a model, or establish accuracy by itself.
The README provides exact reviewed-index schemas and native MuAViC commands for
when authorized data exists, plus separate AV-HuBERT word commands to run when
RAM permits. Those future commands are not reported as already executed.

## Tests and build

```powershell
.\.venv-word-recognition\Scripts\python.exe -m pytest tests/test_multidataset.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_recognition_capabilities.py tests/test_backend_contract.py tests/test_browser_inference.py tests/test_accuracy_pipeline.py tests/test_lrw_whole_word.py tests/test_grid_expansion.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_word_recognition.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

The first attempt to include the acquisition suite in the regular `.venv`
failed collection because that environment lacks `requests`. The optional suite
now skips there and runs in `.venv-word-recognition`, which already has the
pinned dependency. No app-environment installation was needed. The initial full
run found five legacy workflow failures; fixes were verified and the final
full suite passed. Earlier logs are retained under `.qa`.

From `frontend`:

```powershell
npm run build
npm run test:live
npm run test:math
node node_modules/vite/bin/vite.js preview --host 127.0.0.1 --port 4187 --strictPort
```

Temporary preview server was stopped after the browser checks. From project root:

```powershell
node .qa/check-miracl-matrix.mjs
node .qa/check-multidataset-live.mjs
```

Browser scripts used an owned headless Chrome session, preserved the user's
existing servers, and closed their own processes. The research-card check
fulfilled the real capability JSON through CDP and simulated offline APIs; it
was not a claim that a backend had been restarted/deployed.

Additional inline Python audit commands recalculated all 12 preservation
fingerprints, enumerated all 12 checkpoint files, strictly loaded all 11 small
classifier artifacts, and wrote `.qa/multidataset-final-audit.json`. They did
not load the large AV-HuBERT encoder anew or change weights. AST/PowerShell
parser checks and CLI `--help` checks validate the final entry points.
