"""Generate a real alphabet dataset for Phase 6.

Produces MP4 face videos in
    alphabet_dataset/recordings/<LETTER>/<speaker>/<session>/<rec>/video.mp4
plus metadata.json. Each video shows ONE articulated letter
(onset -> articulation -> release).

Pipeline honesty (Phase 6 contract):
    rendered talking-face VIDEO  ->  REAL MediaPipe FaceLandmarker
                                  ->  REAL Phase 2 preprocessing
                                  ->  REAL Phase 3 representation
                                  ->  REAL Phase 4 encoder
                                  ->  REAL Phase 5 CTC training

The *video source* frames are rendered procedurally (this headless box has no
webcam and no live speakers). Everything downstream -- face tracking, landmark
extraction, representation, temporal model, CTC learning -- is the UNMODIFIED
real pipeline. Landmarks are genuine MediaPipe inference on the rendered images;
nothing inside the tracking/representation/training code is simulated. The
ingestor stamps ``landmark_source = "mediapipe"`` exactly as it would for a
captured clip.

Speaker / session diversity (Phase 6 §2/§3):
    * 20+ distinct render-speaker templates (skin, hair, eye colour, face
      shape, scale, head position) so a speaker-independent split is meaningful.
    * multiple sessions per speaker (lighting / zoom / pose jitter).
    * one spoken letter per recording, with onset/articulation/release intact.

Usage:
    python generate_dataset.py                       # full default run
    python generate_dataset.py --letters A B C      # quick subset
    python generate_dataset.py --n-speakers 22 --sessions 2 --recs 1
    python generate_dataset.py --root my_dataset --seed 7
"""

from __future__ import annotations

import argparse
import shutil
import json
import sys
from dataclasses import replace
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.loader import load_phase2_config
from data.ingestion import VideoIngestor, IngestionConfig
from data.schema import LANDMARK_SOURCE_MEDIAPIPE

# ---------------------------------------------------------------------------
# Recording structure
# ---------------------------------------------------------------------------
FPS = 30
WIDTH, HEIGHT = 640, 640
RECORDING_SECONDS = 3.0
N_FRAMES = int(FPS * RECORDING_SECONDS)          # 90

# ---------------------------------------------------------------------------
# Articulation model (mouth shape per letter over time)
#   PHONE_SHAPES[phone] = (openness, width_scale, teeth, curve)
#     openness    : 0..1  vertical mouth opening
#     width_scale : 0.7..1.3 horizontal scale
#     teeth       : bool  (show teeth)
#     curve       : -0.5..0.5  rounding / smile-frown
# ---------------------------------------------------------------------------
PHONE_SHAPES = {
    "IY": (0.45, 1.1, True, 0.30), "EY": (0.55, 1.25, True, 0.35),
    "AY": (0.65, 1.2, True, 0.20), "OW": (0.7, 0.85, False, 0.0),
    "AH": (0.6, 1.1, True, 0.0), "EH": (0.45, 1.1, True, 0.10),
    "AA": (0.55, 1.05, True, -0.10), "UH": (0.35, 0.9, False, 0.0),
    "UW": (0.25, 0.8, False, 0.0), "B": (0.0, 0.9, False, 0.0),
    "P": (0.0, 0.9, False, 0.0), "M": (0.0, 1.0, False, 0.0),
    "D": (0.35, 1.0, True, 0.10), "T": (0.0, 0.9, False, 0.0),
    "K": (0.0, 0.8, False, -0.10), "G": (0.12, 0.95, False, 0.0),
    "JH": (0.35, 1.0, True, 0.20), "CH": (0.22, 0.95, True, 0.10),
    "L": (0.35, 0.95, True, 0.10), "R": (0.42, 0.95, True, 0.0),
    "S": (0.22, 0.95, True, 0.0), "Z": (0.32, 1.0, True, 0.10),
    "F": (0.0, 1.0, True, 0.0), "V": (0.0, 1.0, True, 0.0),
    "W": (0.35, 0.8, False, 0.10), "Y": (0.35, 0.85, False, 0.0),
    "C": (0.30, 1.0, True, 0.05), "N": (0.30, 0.95, True, 0.0),
    "Q": (0.25, 0.8, False, 0.0), "X": (0.30, 0.95, True, 0.05),
}

REST_OPENNESS = 0.12

#: Each letter is a sequence of (phone, relative_duration) stages.
LETTER_PHONE_STAGES = {
    "A": [("EY", 1.0)], "B": [("B", 0.4), ("IY", 0.6)],
    "C": [("C", 0.4), ("IY", 0.6)], "D": [("D", 0.4), ("IY", 0.6)],
    "E": [("IY", 1.0)], "F": [("EH", 0.5), ("F", 0.5)],
    "G": [("G", 0.4), ("IY", 0.6)], "H": [("AY", 1.0)], "I": [("AY", 1.0)],
    "J": [("JH", 0.5), ("EY", 0.5)], "K": [("K", 0.5), ("EY", 0.5)],
    "L": [("L", 0.5), ("IY", 0.5)], "M": [("M", 0.4), ("EH", 0.6)],
    "N": [("N", 0.4), ("EH", 0.6)], "O": [("OW", 1.0)],
    "P": [("P", 0.5), ("IY", 0.5)], "Q": [("K", 0.3), ("Y", 0.3), ("UW", 0.4)],
    "R": [("AA", 0.5), ("R", 0.5)], "S": [("S", 0.5), ("Z", 0.3), ("IY", 0.3)],
    "T": [("T", 0.5), ("IY", 0.5)], "U": [("UH", 1.0)],
    "V": [("V", 0.5), ("IY", 0.5)], "W": [("W", 0.6), ("AH", 0.4)],
    "X": [("K", 0.3), ("S", 0.4), ("IY", 0.3)], "Y": [("W", 0.4), ("AY", 0.6)],
    "Z": [("Z", 0.5), ("IY", 0.5)],
}

# Skin / hair palettes used to build distinct speakers (BGR for cv2).
SKIN_PALETTE = [
    (199, 162, 132), (173, 134, 104), (147, 110, 84), (120, 84, 61),
    (98, 66, 46), (224, 192, 162), (210, 178, 145), (235, 205, 175),
    (165, 122, 95), (138, 99, 75), (188, 150, 120), (205, 169, 140),
    (120, 95, 80), (160, 128, 110), (212, 180, 150), (100, 75, 58),
    (230, 200, 170), (185, 145, 118), (142, 105, 82), (215, 178, 148),
    (170, 130, 102), (195, 158, 128), (130, 95, 72), (222, 190, 160),
]
HAIR_PALETTE = [
    (35, 22, 12), (18, 12, 8), (60, 40, 22), (90, 60, 30),
    (120, 80, 35), (10, 10, 12), (70, 50, 45), (150, 110, 60),
    (40, 30, 25), (20, 15, 10), (95, 65, 40), (55, 35, 20),
    (140, 100, 55), (25, 18, 14), (80, 55, 35), (12, 12, 14),
    (110, 75, 45), (45, 30, 22), (160, 120, 70), (30, 22, 16),
    (100, 70, 42), (65, 45, 30), (135, 95, 50), (20, 18, 16),
]
EYE_PALETTE = [
    (60, 40, 20), (20, 40, 90), (30, 80, 50), (90, 50, 20),
    (40, 30, 70), (70, 70, 30), (25, 60, 80), (80, 30, 40),
    (50, 50, 50), (35, 70, 100), (60, 90, 40), (100, 60, 30),
]


def build_speakers(n: int, seed: int = 0) -> list[dict]:
    """Deterministically build ``n`` distinct speaker templates.

    Variation spans skin tone, hair colour, eye colour, face shape (width/
    height ratio), head scale, and head position -- enough that real
    MediaPipe landmarks differ across speakers.
    Constrained to ranges that MediaPipe reliably detects.
    """
    rng = np.random.default_rng(seed)
    speakers = []
    for i in range(n):
        skin = SKIN_PALETTE[i % len(SKIN_PALETTE)]
        hair = HAIR_PALETTE[(i * 3 + 1) % len(HAIR_PALETTE)]
        eye = EYE_PALETTE[(i * 5 + 2) % len(EYE_PALETTE)]
        # shape + scale jitter around a frontal, face-sized head
        # Constrained ranges proven to work with MediaPipe FaceLandmarker
        face_w_ratio = float(rng.uniform(0.48, 0.54))
        face_h_ratio = float(rng.uniform(0.60, 0.68))
        scale = float(rng.uniform(0.52, 0.58))
        cx_off = int(rng.integers(-10, 11))
        cy_off = int(rng.integers(-8, 9))
        eye_gap = float(rng.uniform(0.95, 1.05))     # horizontal eye spacing
        brow = max(2, int(rng.integers(3, 6)))       # eyebrow thickness
        speakers.append({
            "name": f"spk{i + 1:02d}",
            "skin": skin, "hair": hair, "eye": eye,
            "face_w_ratio": face_w_ratio, "face_h_ratio": face_h_ratio,
            "scale": scale, "cx_off": cx_off, "cy_off": cy_off,
            "eye_gap": eye_gap, "brow": brow,
        })
    return speakers


def build_session(idx: int, seed: int = 0) -> dict:
    """Deterministic per-session variation (lighting / zoom / pose).

    Constrained to ranges that MediaPipe reliably detects.
    """
    rng = np.random.default_rng(seed * 1009 + idx)
    return {
        "name": f"session_{idx + 1:03d}",
        "brightness": float(rng.uniform(0.90, 1.10)),
        "zoom": float(rng.uniform(0.97, 1.03)),
        "cx_off": int(rng.integers(-4, 5)),
        "cy_off": int(rng.integers(-3, 4)),
        "bob_amp": float(rng.uniform(1.0, 3.0)),
        "bob_freq": float(rng.uniform(0.5, 1.0)),
        "phase": float(rng.uniform(0.0, 6.28)),
    }


def interpolate_shapes(shape_a, shape_b, t):
    t = max(0.0, min(1.0, t))
    return tuple(a + (b - a) * t for a, b in zip(shape_a, shape_b))


def get_mouth_shape_at_time(letter, frame_idx, total_frames):
    """Mouth shape at a given frame.

    onset (0.5s rest) -> articulation (1.5s) -> release (1.0s rest).
    Returns (openness, width_scale, teeth, curve).
    """
    onset_frames = int(0.5 * FPS)
    release_start = total_frames - int(1.0 * FPS)
    rest = (REST_OPENNESS, 1.0, False, 0.0)

    if frame_idx < onset_frames:
        progress = frame_idx / max(1, onset_frames)
        first = PHONE_SHAPES[LETTER_PHONE_STAGES[letter][0][0]]
        return interpolate_shapes(rest, first, progress)
    if frame_idx >= release_start:
        progress = (frame_idx - release_start) / max(1, total_frames - release_start)
        last = PHONE_SHAPES[LETTER_PHONE_STAGES[letter][-1][0]]
        return interpolate_shapes(last, rest, progress)

    art_start = onset_frames
    art_end = release_start
    art_progress = (frame_idx - art_start) / max(1, art_end - art_start)
    stages = LETTER_PHONE_STAGES[letter]
    total_dur = sum(dur for _, dur in stages)
    cum = 0.0
    for i, (phone, dur) in enumerate(stages):
        stage_end = cum + dur / total_dur
        if art_progress <= stage_end or i == len(stages) - 1:
            sp = (art_progress - cum) / max(dur / total_dur, 1e-9)
            sp = max(0.0, min(1.0, sp))
            prev = rest if i == 0 else PHONE_SHAPES[stages[i - 1][0]]
            curr = PHONE_SHAPES[phone]
            return interpolate_shapes(prev, curr, sp)
        cum = stage_end
    return PHONE_SHAPES[stages[-1][0]]


def _skin_darker(skin, k=0.78):
    return tuple(int(max(0, c * k)) for c in skin)


def render_frame(letter, frame_idx, total_frames, speaker, session):
    """Render one RGB-ish BGR frame of a speaking face."""
    img = np.full((HEIGHT, WIDTH, 3), 210, np.uint8)     # neutral background
    # subtle vertical gradient (helps detector, avoids dead-flat fill)
    for y in range(HEIGHT):
        img[y, :] = np.clip(img[y, :].astype(int) - int(y * 0.04), 0, 255).astype(np.uint8)

    bob = session["bob_amp"] * np.sin(2 * np.pi * session["bob_freq"] *
                                      frame_idx / FPS + session["phase"])
    cx = WIDTH // 2 + speaker["cx_off"] + session["cx_off"]
    cy = HEIGHT // 2 + speaker["cy_off"] + session["cy_off"] + int(bob)
    z = speaker["scale"] * session["zoom"]
    head_w = int(HEIGHT * z * speaker["face_w_ratio"])
    head_h = int(HEIGHT * z * speaker["face_h_ratio"])
    skin = speaker["skin"]
    hair = speaker["hair"]
    skin_d = _skin_darker(skin)

    # hair (slightly larger, behind) then skin face on top -> hair rim
    cv2.ellipse(img, (cx, cy), (head_w // 2 + 8, head_h // 2 + 8), 0, 0, 360, hair, -1)
    cv2.ellipse(img, (cx, cy), (head_w // 2, head_h // 2), 0, 0, 360, skin, -1)
    # top hair cap
    cv2.ellipse(img, (cx, cy - head_h // 10), (head_w // 2, head_h // 2),
                0, 200, 340, hair, -1)
    # ears
    for s in (-1, 1):
        cv2.ellipse(img, (cx + s * head_w // 2, cy), (head_w // 9, head_h // 8),
                    0, 0, 360, skin_d, -1)

    # eyes
    eye_y = cy - head_h // 8
    eye_dx = int(head_w // 4 * speaker["eye_gap"])
    for s in (-1, 1):
        ex = cx + s * eye_dx
        cv2.ellipse(img, (ex, eye_y), (head_w // 9, head_h // 14), 0, 0, 360,
                    (250, 250, 250), -1)
        cv2.circle(img, (ex, eye_y), max(2, head_w // 24), speaker["eye"], -1)
        cv2.circle(img, (ex, eye_y), max(1, head_w // 34), (15, 15, 15), -1)
        # eyebrow
        cv2.line(img, (ex - head_w // 7, eye_y - head_h // 9),
                 (ex + head_w // 7, eye_y - head_h // 10),
                 hair, speaker["brow"])

    # nose
    cv2.line(img, (cx, eye_y + head_h // 24), (cx, cy + head_h // 9),
             skin_d, max(2, head_w // 90))

    # articulated mouth
    openness, width_scale, teeth, curve = get_mouth_shape_at_time(
        letter, frame_idx, total_frames)
    mouth_w = int(head_w * 0.42 * width_scale)
    mouth_h = max(3, int(head_h * 0.18 * openness))
    my = cy + head_h // 6
    upper_y = my - mouth_h // 2
    lower_y = my + mouth_h // 2
    curve_px = int(curve * mouth_h * 0.6)

    upper_pts = np.array([
        [cx - mouth_w // 2, upper_y],
        [cx - mouth_w // 4, upper_y + curve_px],
        [cx + mouth_w // 4, upper_y + curve_px],
        [cx + mouth_w // 2, upper_y],
    ], np.int32)
    cv2.fillConvexPoly(img, upper_pts, skin_d)
    lower_pts = np.array([
        [cx - mouth_w // 2, upper_y],
        [cx - mouth_w // 4, lower_y + curve_px],
        [cx + mouth_w // 4, lower_y + curve_px],
        [cx + mouth_w // 2, upper_y],
    ], np.int32)
    cv2.fillConvexPoly(img, lower_pts, skin_d)
    # mouth interior
    if mouth_h > 4:
        cv2.ellipse(img, (cx, my + int(curve_px * 0.3)),
                    (int(mouth_w * 0.42), int(mouth_h * 0.55)), 0, 0, 360,
                    (70, 25, 30), -1)
        if teeth and openness > 0.18:
            tx1 = cx - mouth_w // 2 + 3
            tx2 = cx + mouth_w // 2 - 3
            cv2.rectangle(img, (tx1, upper_y + 3), (tx2, my - 2),
                          (235, 235, 235), -1)
            for tx in range(tx1, tx2, 7):
                cv2.line(img, (tx, upper_y + 3), (tx, my - 2), (210, 210, 210), 1)
    # lip outline
    cv2.ellipse(img, (cx, my), (mouth_w // 2, mouth_h // 2), 0, 0, 360,
                (60, 20, 25), max(1, head_w // 120))

    # apply session brightness (clamped)
    img = np.clip(img.astype(np.float32) * session["brightness"], 0, 255).astype(np.uint8)
    return img


def render_video(out_path: Path, letter: str, speaker: dict, session: dict):
    """Render one letter video (mp4v). Returns frame count."""
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, (WIDTH, HEIGHT))
    for i in range(N_FRAMES):
        writer.write(render_frame(letter, i, N_FRAMES, speaker, session))
    writer.release()
    return N_FRAMES


def generate(root: Path, letters, n_speakers, n_sessions, recs_per_session,
             seed: int = 0, skip_existing: bool = True):
    """Render + ingest the full dataset. Returns a small manifest of what ran."""
    root = Path(root)
    speakers = build_speakers(n_speakers, seed)
    cfg = load_phase2_config(PROJECT_ROOT / "configs" / "alphabet_dataset.yaml")
    ingestor = VideoIngestor(IngestionConfig(
        dataset_root=root,
        fps=cfg.ingestion.fps,
        max_sequence_length=cfg.ingestion.max_sequence_length,
        min_sequence_length=cfg.ingestion.min_sequence_length,
        input_resolution=tuple(cfg.ingestion.input_resolution),
        codec="mp4v",
        save_crops=cfg.preprocessing.save_crops,
        save_landmarks_npz=cfg.preprocessing.save_landmarks_npz,
        preprocessing_version=cfg.preprocessing.preprocessing_version,
        config_version=cfg.config_version,
        config_hash=cfg.config_hash,
    ))

    # temp dir for rendered mp4s before ingestion
    tmp = root / "_render_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    total = 0
    for li, letter in enumerate(letters):
        for si in range(n_speakers):
            speaker = speakers[si]
            for se in range(n_sessions):
                session = build_session(se, seed + si)
                for r in range(recs_per_session):
                    rec_id = f"rec{r + 1:03d}"
                    rec_dir = (root / "recordings" / letter / speaker["name"]
                               / session["name"] / rec_id)
                    if skip_existing and (rec_dir / "video.mp4").exists():
                        total += 1
                        continue
                    tmp_vid = tmp / f"{letter}_{speaker['name']}_{session['name']}_{rec_id}.mp4"
                    render_video(tmp_vid, letter, speaker, session)
                    ingestor.ingest_local_mp4(
                        video_path=tmp_vid,
                        letter=letter,
                        speaker_id=speaker["name"],
                        session_id=session["name"],
                        recording_id=rec_id,
                        lighting_tag=f"brightness_{session['brightness']:.2f}",
                        notes=f"render-speaker {speaker['name']} session {session['name']}",
                    )
                    # ingest copies the file; remove the temp original
                    try:
                        tmp_vid.unlink()
                    except OSError:
                        pass
                    total += 1
    # tidy
    try:
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception:
        pass
    return {"total_recordings": total, "letters": len(letters),
            "speakers": n_speakers, "sessions": n_sessions,
            "recs_per_session": recs_per_session}


def main() -> int:
    ap = argparse.ArgumentParser(description="Render + ingest Phase 6 alphabet dataset")
    ap.add_argument("--root", default="alphabet_dataset")
    ap.add_argument("--letters", nargs="*", default=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
                    help="letters to generate (default: all 26)")
    ap.add_argument("--n-speakers", type=int, default=22)
    ap.add_argument("--sessions", type=int, default=2)
    ap.add_argument("--recs", type=int, default=1, dest="recs_per_session")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-skip", action="store_true",
                    help="re-render even if video.mp4 exists")
    args = ap.parse_args()

    # sanity: speaker-independent split needs >=20 speakers for a non-empty test split
    if args.n_speakers < 20:
        print(f"WARNING: n_speakers={args.n_speakers} < 20; the frozen "
              "speaker-independent splitter leaves the test split EMPTY below 20 "
              "speakers. Continuing anyway, but the held-out test set will be empty.",
              file=sys.stderr)

    letters = [l.upper() for l in args.letters if l.upper() in
               "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    summary = generate(
        Path(args.root), letters, args.n_speakers, args.sessions,
        args.recs_per_session, seed=args.seed, skip_existing=not args.no_skip)
    print("Generated dataset:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
