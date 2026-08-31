# Alphabet Visual Speech Dataset

**Phase 2: Dataset Format + Recording Metadata**

A scientifically valid dataset for visual speech decoding of spoken alphabet letters (A-Z), implementing the architecture described in DESIGN.md.

## Overview

This project implements the **dataset and recording layer** for a visual speech recognition system that models:

```
appearance + geometry + motion
        ↓
gated fusion
        ↓
spatiotemporal representation
        ↓
Conformer
        ↓
phonetic / viseme representation
        ↓
letter CTC decoding
```

## Phase 2 Implementation

The following components are implemented:

### 1. Recording-Centric Dataset Structure
```
alphabet_dataset/
    recordings/
        A/
            speaker_001/
                session_001/
                    recording_001/
                        video.mp4          # Original source video (preserved)
                        metadata.json      # Complete recording metadata
                        frames/            # Frame-level data
                        landmarks/         # Face & mouth landmarks (.npz)
                        crops/             # Mouth & face crops (optional)
                        motion/            # Motion features (optional)
```

### 2. Comprehensive Metadata Schema
Every recording has `metadata.json` with:
- Identity: letter, speaker_id, session_id, recording_id
- Source: webcam/local_mp4, created_at timestamp
- Video: fps, frame_count, width, height, duration_ms
- Face tracking: frames with/without face, confidence stats
- Mouth: frames with mouth, missing frames, size statistics
- Quality: brightness, blur, mouth size, head pose, duplicate frames
- Preprocessing version tracking

### 3. Speaker-Aware Splits
- Speaker-independent train/val/test splits (70/15/15 by default)
- No speaker/session leakage across evaluation boundaries
- Splits stored in `splits.json` with full traceability

### 4. Video Ingestion
- Local MP4 file ingestion
- Webcam recording with configurable duration
- Automatic directory structure creation
- Metadata generation

### 5. Face Tracking & Mouth Alignment
- MediaPipe FaceMesh for 468 facial landmarks
- Identity preservation across frames
- Mouth-centered coordinate normalization (scale = inner mouth width)
- Head pose estimation (pitch, yaw, roll)
- Quality flagging per frame

### 6. Three-Stream Representation Extraction
- **Appearance**: 64×64 RGB mouth crops → ResNet stem → 96-dim embeddings
- **Geometry**: 20 canonical scalars from landmarks → MLP → 64-dim embeddings
- **Motion**: Temporal derivatives (Savitzky-Golay) + optical flow → MLP → 64-dim embeddings
- **Gated Fusion**: Learned per-timestep weighting → 192-dim fused representation

### 7. Dataset Manifest
- `manifest.jsonl` - one line per recording for fast indexing
- Contains all essential metadata for dataset discovery
- No need to scan thousands of files

### 8. Validation & Inspection Commands
```bash
# Validate dataset integrity
python -m alphabet validate-dataset <dataset_root>

# Inspect dataset statistics
python -m alphabet inspect-dataset <dataset_root>

# Ingest new recordings
python -m alphabet ingest <dataset_root> <letter> <speaker> <session> [options]
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

All configurable parameters in `configs/alphabet_dataset.yaml`:
- Dataset paths and structure
- Video settings (FPS, resolution, sequence length)
- Face tracking thresholds
- Mouth alignment parameters
- Quality thresholds
- Preprocessing version
- Split ratios and random seed

## Usage

### Ingest a local video
```bash
python -m alphabet ingest ./alphabet_dataset A speaker_001 session_001 \
  --source local_mp4 --video-path ./my_video.mp4
```

### Record from webcam
```bash
python -m alphabet ingest ./alphabet_dataset B speaker_001 session_001 \
  --source webcam --duration 3.0 --preview
```

### Validate dataset
```bash
python -m alphabet validate-dataset ./alphabet_dataset
```

### Inspect dataset statistics
```bash
python -m alphabet inspect-dataset ./alphabet_dataset
```

## Testing

```bash
pytest tests/ -v
```

## Project Structure
```
alphabet_dataset/          # Dataset root (created at runtime)
configs/
    alphabet_dataset.yaml  # All configuration parameters
alphabet/
    __init__.py
    __main__.py            # Entry point: python -m alphabet
    cli.py                 # CLI commands
data/
    schema.py              # Data classes & validation
    ingestion.py           # Video ingestion
    manifest.py            # Manifest creation & splits
    preprocess.py          # Full preprocessing pipeline
    validate.py            # Validation command
    inspect.py             # Inspection command
tracking/
    aligner.py             # Face tracking & mouth alignment
repr/
    appearance.py          # Appearance stream extraction
    geometry.py            # Geometry stream extraction
    motion.py              # Motion stream extraction
    fusion.py              # Gated fusion
tests/
    test_*.py              # Unit & integration tests
pyproject.toml
requirements.txt
DESIGN.md                  # Architecture specification
```

## Design Principles (from DESIGN.md)

1. **Recording-centric** - One recording = one temporal sample
2. **Preserve source video** - Never destroy original video
3. **Temporal integrity** - Chronological frame ordering, no shuffling
4. **Quality flags over deletion** - Record issues, don't silently drop data
5. **Speaker-independent splits** - No leakage across evaluation boundaries
6. **Configuration-driven** - No hardcoded magic numbers
7. **Streaming processing** - Bounded memory, resumable preprocessing
8. **Extensible schemas** - Versioned metadata with optional fields

## Next Phases

- **Phase 3**: Face tracking + mouth alignment (production implementation)
- **Phase 4**: Three-stream representation extraction
- **Phase 5**: Temporal sequence loader
- **Phase 6-7**: Spatial-temporal encoder + Conformer
- **Phase 8**: Latent visual speech representation (phonetic/CTC)
- **Phase 9**: Sequence decoder (letter CTC + ambiguity engine)
- **Phase 10**: Ambiguity estimation
- **Phase 11**: Training pipeline
- **Phase 12**: Speaker-independent evaluation
- **Phase 13**: Failure analysis
- **Phase 14**: Live streaming inference
- **Phase 15**: Word composition# LipNet-Alphabet
