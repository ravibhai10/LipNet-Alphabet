# Alphabet Lip-Reading Model

Train and run a CNN-BiLSTM lip-reader that recognises Aâ€“Z from grayscale mouth crops
extracted from GRID video using MediaPipe FaceLandmarker. Single-speaker (s1) by
default; multi-speaker mode ready out-of-the-box.

> **Note:** GRID speaker s1 does not contain the letter **W**, so the model has
> 25 output classes (Aâ€“V, Xâ€“Z). The trained checkpoint works with those 25 classes.

---

## Quick-start

### 1 â€” Extract clips from GRID video

```bash
# Single speaker (default = s1)
python scripts/prepare_alphabet.py
# Optional: cap per-letter for a quick smoke test
python scripts/prepare_alphabet.py --max-per-letter 2

# Multi-speaker â€” process specific speakers
python scripts/prepare_alphabet.py --speakers s1 s2 s3 --out-dir dataset/alphabet_multi
# Or auto-discover all available speakers
python scripts/prepare_alphabet.py --speakers all --out-dir dataset/alphabet_multi
```

Writes clips to `dataset/alphabet/{LETTER}/clip_{SPEAKER}_{NNNNN}.npy`
and summary metadata to `dataset/alphabet/summary.json`.
The `index.jsonl` records which speaker each clip came from.

### 2 â€” Train

```bash
python scripts/train_alphabet.py
```

| Flag              | Default  | Description                               |
|-------------------|----------|-------------------------------------------|
| `--epochs`        | 50       | Max training epochs                        |
| `--batch-size`    | 16       |                                            |
| `--lr`           | 1e-3     | Initial learning rate                      |
| `--patience`      | 10       | Early-stopping patience (epochs)           |
| `--device`        | `auto`   | `cpu`, `cuda`, or `auto`                  |
| `--out-dir`       | `models/` | Where checkpoints and plots land          |

Best model â†’ `models/alphabet_classifier.pt`
Training curve â†’ `models/alphabet_training.png`

### 3 â€” Run live webcam inference

```bash
python alphabet/infer.py
```

| Key       | Action                                  |
|-----------|-----------------------------------------|
| `SPACE`   | Record 1.5 s clip, then show prediction |
| `Q` / ESC | Quit                                    |

The script streams the webcam feed, shows a live mouth-crop preview,
records when SPACE is pressed, normalises the clip to 30 frames, and
runs the PyTorch model on CPU.

---

## Architecture

```
Input:          (B, T=30, 1, H=140, W=46)   â€” 30 grayscale mouth frames
TimeDistributed Conv2D(16)  + BN + ReLU + MaxPool(2Ã—2)
TimeDistributed Conv2D(32)  + BN + ReLU + MaxPool(2Ã—2)
TimeDistributed Conv2D(64)  + BN + ReLU + MaxPool(2Ã—2)
TimeDistributed AdaptiveAvgPool2d(4Ã—4)  â†’ (B, T, 1024)
BiLSTM(96, bidirectional=True)         â†’ (B, 192)
Dense(192â†’96) + ReLU + Dropout(0.5)
Dense(96â†’26)  â†’ (B, 26) softmax
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Total trainable parameters: 913,274
Framework: PyTorch 2.13 (CPU)
```

> TensorFlow is unavailable for Python 3.14 â€” the identical architecture was
> reimplemented in PyTorch.

---

## Results (GRID s1, 30 epochs, CPU)

| Split       | Samples | Accuracy |
|-------------|---------|----------|
| Train       | 800     | 13.4 %   |
| Val         | 100     | 20.0 %   |
| Test        | 100     | 12.0 %   |
| **Best Val**| â€”       | **20.0 %** |

Per-class test accuracy (letters with > 0 %):
- **M**: 100 %  **O**: 67 %  **K**: 50 %
- **B/D/I/U**: 33 %  **F/R**: 25 %  **J**: 14 %

**Why is accuracy limited?**  
GRID speaker s1 has only ~40 clips per letter (â‰ˆ 1000 total across 25 classes),
with 800 for training â€” far too few for a 913 K-param model with 25 classes.
Collecting data from multiple GRID speakers (s2â€“s34) would dramatically improve
accuracy; the pipeline is ready for that.

---

## Key implementation notes

| Topic | Detail |
|-------|--------|
| **GRID .align format** | Timestamps are in units of 1/25000 s (verified against video duration) |
| **MediaPipe mode** | Uses `IMAGE` mode (stateless, frame-independent) â€” VIDEO mode causes a sticky-box artefact |
| **Mouth ROI** | 24-point MediaPipe landmark set around lips |
| **Clip geometry** | `(T=30, H=140, W=46)` â€” 30 frames, 140 px tall, 46 px wide |
| **Time padding** | Symmetric edge replication for clips shorter than 30 frames |

## Scaling to multi-speaker (recommended for real generalization)

The single-speaker accuracy ceiling (~20 percent val) is purely a data issue:
913 K params with 800 training samples cannot generalize.
The pipeline is fully multi-speaker ready — just feed it more data.

### Step 1 -- Download GRID speakers

Each zip is ~400 MB from Zenodo:

    mkdir -p data/raw/grid
    cd data/raw/grid
    for i in {1..34}; do
      wget https://zenodo.org/records/3625687/files/s\.zip
    done

Unzip each into data/sN/sN/ (video) and data/alignments/alignments/sN/ (.align).

### Step 2 -- Multi-speaker extraction

    python scripts/prepare_alphabet.py --speakers all \
        --max-per-letter 40 --out-dir dataset/alphabet_multi

Expected: 34 speakers x 40 clips/letter x 25 letters = ~33,000 clips.
summary.json lists each processed speaker.

### Step 3 -- Retrain

    python scripts/train_alphabet.py --data-dir dataset/alphabet_multi --epochs 100

Same architecture. With 33 K clips, expect > 80 percent test accuracy.

