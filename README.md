# LipNet-Alphabet: Visual Speech Recognition (A–Z)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![License: Research](https://img.shields.io/badge/license-Research-green.svg)](DESIGN.md)

LipNet AI is a visual speech research presentation with an experimental A–Z camera demo and a separate technical workspace. The active browser model is the saved **CNN-BiLSTM alphabet classifier on Windows CPU**, not the research Conformer below. Confidence scores are uncalibrated; no independent webcam accuracy or new accuracy gain has been established.

The official AV-HuBERT encoder is present and has encoded a real video on CPU. Its supervised A–Z head is not trained. The current corpus has 1,000 GRID sentences from one speaker and 45 generated LRW fixtures, with no registered authentic alphabet recordings. Those inputs cannot establish a speaker-independent A–Z benchmark.

Start with [Project status](PROJECT_STATUS.md), [Model availability and API contract](MODEL_AVAILABILITY.md), [Validation report](VALIDATION_REPORT.md), and the [data/training/evaluation guide](ACCURACY_PIPELINE.md).

## Story and technical experiences

The default `/` route tells six scenes: the problem, movement over time, the actual processing steps, a simple live demo, evidence and limitations, and the research journey. The original animated face, landmark particles, and GSAP scroll motion are reused in the hero. Motion can be paused and respects reduced-motion settings. Diagnostics stay collapsed in the story demo. The header links Live Demo and Science Lab directly; its All pages menu links every project page. `/demo` retains dataset, training, model, and evaluation tools, with Science and research links in its sidebar. `/research` preserves the original detailed exhibits, and `/science` opens the interactive Science Lab.

## Windows CPU setup

Run commands individually from `C:\Users\Dell\Downloads\LipNet-Alphabet-main\LipNet-Alphabet-main`. Use the existing environments and dependencies; GPU, WSL, Linux, and Docker are unnecessary.

Start the backend in one PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Start the frontend in another terminal, with its working directory set to `frontend`:

```powershell
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Open [the story](http://127.0.0.1:5173/), [the live story demo](http://127.0.0.1:5173/#demo), or [the technical workspace](http://127.0.0.1:5173/demo/overview). Backend health is at [port 8000](http://127.0.0.1:8000/health), with API documentation at [/docs](http://127.0.0.1:8000/docs).

If dependencies are missing, install `requirements.txt` into `.venv`, and run `npm ci` in `frontend`. Do not replace the existing AV-HuBERT CPU environment with an unpatched upstream Fairseq installation; its vendored Windows/OmegaConf compatibility changes are documented in [ACCURACY_PIPELINE.md](ACCURACY_PIPELINE.md).

For another backend address, copy `frontend/.env.example` to `frontend/.env.local`, set `VITE_API_BASE_URL`, then restart Vite. Set `LIPNET_CORS_ORIGINS` on the backend to a comma-separated allowlist if using another frontend origin. An occupied port requires checking the existing server before starting another instance. The backend serves `frontend/dist` at port 8000 after a successful `npm run build` in `frontend`.

Run the regression suite from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest tests pipeline/avhubert/tests pipeline/grid/tests pipeline/eval/tests -q
```

Run `npm run test:live` and `npm run build` individually in `frontend`. For actual local HTTP/WebSocket smoke checks:

```powershell
.\.venv\Scripts\python.exe scripts/verify_local.py --http --output .qa/local-http.json
```

For a bounded real-video encoder check in the existing CPU environment:

```powershell
.\.venv-avhubert-cpu\Scripts\python.exe scripts/verify_local.py --encoder-smoke --output .qa/local-encoder.json
```

These smoke tests verify execution and transport, not recognition accuracy. To add real data, validate annotations, cache features, train a new head, evaluate per-class results, or activate the trained model, follow [ACCURACY_PIPELINE.md](ACCURACY_PIPELINE.md).

## Research architecture reference

The multistream/Conformer/CTC code below remains available for research. It is separate from the active browser CNN-BiLSTM path and does not establish calibrated recognition or sentence availability.

```
       [Video / Webcam Frame Stream]
                     ↓
  MediaPipe FaceLandmarker (Canonical 20-pt Mouth)
                     ↓
┌────────────────────┬────────────────────┬────────────────────┐
│ Appearance Stream  │  Geometry Stream   │   Motion Stream    │
│  (ResNet Stem 96d) │    (MLP 20→64d)    │ (Savitzky-Golay 64d│
└────────────────────┴────────────────────┴────────────────────┘
                     ↓
          Learned Gated Fusion (192d)
                     ↓
     4× Conformer Temporal Encoder Blocks
    (Macaron FFN + Multi-Head Self-Attention + Depthwise Conv)
                     ↓
┌────────────────────┬────────────────────┬────────────────────┐
│  Letter CTC Head   │  Phonetic CTC Head │  Viseme Aux Head   │
│   (27 classes)     │  (ARPABET 27 cls)  │ (Bear-Harvey 15cls)│
└────────────────────┴────────────────────┴────────────────────┘
                     ↓
  Ambiguity Engine & Temperature Calibration
       [Letter | AMBIGUOUS | UNKNOWN]
```

---

## 🌟 Key Features

1. **Recording-Centric Dataset Format**:
   - `alphabet_dataset/recordings/<LETTER>/<speaker>/<session>/<recording>/video.mp4`
   - Complete `metadata.json` tracking face confidence, mouth width/height, Laplacian blur, and brightness.
2. **Three-Stream Representation**:
   - **Appearance**: 64×64 normalized RGB mouth crop passed through a ResNet stem → 96-dim embedding.
   - **Geometry**: 20 canonical scalars normalized by mouth-width at rest → MLP → 64-dim embedding.
   - **Motion**: First- and second-order temporal derivatives of articulators → 64-dim embedding.
   - **Learned Gated Fusion**: Dynamic per-timestep gating yielding a rich 192-dim representation.
3. **Temporal Conformer Backbone**:
   - Depthwise separable convolutions + relative sinusoidal positional encoding + multi-head self-attention.
4. **Phonetic & Viseme Multi-Task CTC**:
   - Supervised by ARPABET phone strings (e.g. `B` → `/B IY/`, `F` → `/EH F/`) and Bear-Harvey viseme clusters.
5. **Calibrated Ambiguity & Abstention**:
   - Temperature scaling fit on validation data with Expected Calibration Error (ECE) metrics.
   - Emits `AMBIGUOUS` (top candidates) or `UNKNOWN` (non-speech) rather than hallucinating confident errors.
6. **Interactive Explorer & Live Pipeline**:
   - Modern FastAPI web server with camera capture, live temporal ring buffer, and web exploration UI.

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/your-repo/LipNet-Alphabet.git
cd LipNet-Alphabet-main

# Set up virtual environment
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Alternative single-origin built frontend

Run the startup script:
```bash
start_server.bat
```
Or start via uvicorn directly:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- **Web App**: Open [http://localhost:8000](http://localhost:8000) after building `frontend` with `npm run build`. Use the Windows CPU setup above for development.
- **Interactive OpenAPI/Swagger Docs**: Open [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🛠️ CLI Operations

```bash
# Validate dataset integrity
python -m alphabet validate-dataset ./alphabet_dataset

# Inspect dataset statistics
python -m alphabet inspect-dataset ./alphabet_dataset

# Preprocess recordings into representation caches
python -m alphabet preprocess ./alphabet_dataset

# Train the Conformer CTC decoder
python -m alphabet train-decoder --config configs/alphabet_dataset.yaml
```

---

## 🧪 Testing

Run the full suite of unit and integration tests:
```bash
pytest tests/ -v
# or using the test runner script:
python test_runner.py
```

---

## 📂 Project Structure

```
LipNet-Alphabet-main/
├── alphabet/          # Core model architectures, losses, vocab, CLI, and trainer
├── app/               # FastAPI backend, WebSocket streaming, and Camera services
│   └── frontend/      # Interactive browser UI for recording and dataset exploration
├── config/            # YAML configuration loader and schema validation
├── configs/           # Configuration files (alphabet_dataset.yaml)
├── data/              # Video ingestion, dataset validation, manifest generator
├── models/            # Pre-trained models and MediaPipe task bundles
├── repr/              # Appearance, Geometry, Motion extractors and GatedFusion
├── temporal/          # 4-block Conformer temporal encoder
├── tracking/          # MediaPipe FaceLandmarker tracking and mouth aligner
└── tests/             # Comprehensive unit and integration test suite
```

---

## 📜 Documentation

- [`DESIGN.md`](DESIGN.md): Detailed scientific research and architectural specification.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): Setup, coding standards, and contribution guide.
