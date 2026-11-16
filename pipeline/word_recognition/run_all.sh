#!/usr/bin/env bash
# End-to-End Automated Lip-Reading Pipeline Orchestrator
# Strict error handling: halts immediately if any stage fails.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG="${1:-pipeline/word_recognition/config.cpu.json}"
RUN_DIR="${2:-runs/word_recognition_v1}"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -f .venv/Scripts/python.exe ]]; then
    PYTHON=.venv/Scripts/python.exe
  elif [[ -f .venv/bin/python ]]; then
    PYTHON=.venv/bin/python
  else
    PYTHON=python3
  fi
fi

echo "================================================================================"
echo "STARTING END-TO-END AUTOMATED LIP-READING PIPELINE"
echo "Root: $ROOT"
echo "Config: $CONFIG"
echo "Run Directory: $RUN_DIR"
echo "Python Executable: $PYTHON"
echo "================================================================================"

# Stage 1: Dependency & Environment Verification
echo ""
echo ">>> [STAGE 1/6] Environment & Dependency Check..."
"$PYTHON" -c "import torch, torchvision, cv2, mediapipe, sklearn, pandas, numpy, matplotlib; print('[PASS] All Core AVSR libraries confirmed loadable.')"

# Stage 2: Dataset Acquisition & Verification
echo ""
echo ">>> [STAGE 2/6] Automated Dataset Ingestion & Checksum Verification..."
"$PYTHON" -m pipeline.word_recognition.download --config "$CONFIG" --run "$RUN_DIR"

# Stage 3: Preprocessing, Mouth ROI Extraction & Augmentation Staging
echo ""
echo ">>> [STAGE 3/6] Preprocessing, Viseme Landmark Extraction & Canonical Normalization..."
"$PYTHON" -m pipeline.word_recognition.preprocess --config "$CONFIG" --run "$RUN_DIR"

# Stage 4: Smoke Testing & Early Validation Suite
echo ""
echo ">>> [STAGE 4/6] Smoke Testing & Layer Geometry / Gradient Propagation Verification..."
"$PYTHON" -m pipeline.word_recognition.smoke_test --config "$CONFIG" --run "$RUN_DIR"

# Stage 5: Model Architecture & End-to-End Training
echo ""
echo ">>> [STAGE 5/6] Spatio-Temporal Model Training (Conv3D + BiGRU)..."
"$PYTHON" -m pipeline.word_recognition.train --config "$CONFIG" --run "$RUN_DIR"

# Stage 6: Comprehensive Testing, Diagnostics & Confusion Matrix Export
echo ""
echo ">>> [STAGE 6/6] Isolated Test Evaluation, Classification Report & Confusion Matrix Export..."
"$PYTHON" -m pipeline.word_recognition.test --config "$CONFIG" --run "$RUN_DIR"

echo ""
echo "================================================================================"
echo "[SUCCESS] PIPELINE EXECUTION COMPLETED WITH 0 ERRORS."
echo "Artifacts, metrics, and diagnostics saved in: $RUN_DIR"
echo "================================================================================"
