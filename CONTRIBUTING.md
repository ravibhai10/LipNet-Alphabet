# Contributing to LipNet-Alphabet

Welcome to the **LipNet-Alphabet** project! This guide explains how to set up your environment, follow code conventions, and contribute improvements.

## 🚀 Quick Setup

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd LipNet-Alphabet-main
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Verify installation**:
   ```bash
   pytest tests/ -q
   ```

## 🧠 Architecture Overview

- **`tracking/`**: MediaPipe FaceLandmarker detection and canonical mouth alignment (20 landmarks).
- **`repr/`**: 3-stream representation extractors (Appearance ResNet stem, Geometry MLP, Motion Savitzky-Golay) + learned GatedFusion.
- **`temporal/`**: Conformer temporal encoder (4 blocks, depthwise separable convolutions, multi-head self-attention).
- **`alphabet/`**: Multi-task model with Letter CTC decoding, ARPABET phonetic & Bear-Harvey viseme supervision, and uncertainty calibration.
- **`app/`**: FastAPI backend with live WebSocket streaming, dataset exploration UI, and camera service.

## 🧪 Testing Guidelines

- Always run `pytest tests/ -v` before pushing.
- Unit tests should be isolated and fast.
- Never use synthetic landmarks in production code. Real MediaPipe landmarks are required.

## 💻 Running the Server

Start the interactive UI with:
```bash
# Windows
start_server.bat
# or
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Navigate to `http://localhost:8000` to view the explorer.
