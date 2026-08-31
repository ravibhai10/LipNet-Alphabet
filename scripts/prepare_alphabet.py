#!/usr/bin/env python3
"""Dataset preparation for Lip Alphabet (multi-speaker GRID).

Parses GRID corpus `.align` files to find the LETTER token and extracts the
corresponding frames from matching `.mpg` videos. Each frame is mouth-cropped
via MediaPipe FaceLandmarker (IMAGE mode — VIDEO mode causes a sticky-box
artefact), converted to grayscale 46x140, and the clip is padded / trimmed
to exactly 30 frames.

Single-speaker (default — speaker s1):
    python scripts/prepare_alphabet.py

Multi-speaker:
    python scripts/prepare_alphabet.py --speakers s1 s2 s3
    python scripts/prepare_alphabet.py --speakers all
    python scripts/prepare_alphabet.py --speakers all --out-dir dataset/alphabet_multi

NOTE: GRID `.align` timestamps are in units of 1/25000 second (40 µs).
Frame conversion: round(value / 25000.0 * fps).
Verified against bbaf2n.mpg (75 frames @ 25 fps = 3 s; .align ends at 74500).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mediapipe as mp
from mediapipe.tasks.python import vision as mpv

LOGGER = logging.getLogger("prepare_alphabet")

# GRID sentence structure (after stripping sil):
#   <cmd> <color> <prep> <LETTER> <digit> <adv>
# Index 3 (zero-based) is the letter token.
CONTENT_WORD_LETTER_INDEX = 3

ALPHABET = [chr(ord("A") + i) for i in range(26)]

# Mouth crop geometry
MOUTH_W = 46
MOUTH_H = 140
SEQ_LEN = 30

# 24-point lip landmark set (MediaPipe FACEMESH indices)
LMS_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
]


def parse_align_file(path: Path) -> List[Tuple[int, int, str]]:
    """Parse a GRID `.align` file into (start_units, end_units, token) tuples.

    Timestamps are in 1/25000-s units (NOT milliseconds). Verified:
    bbaf2n.mpg is 75 frames @ 25 fps = 3 s; its .align ends at 74500,
    i.e. 74500 / 25000 = 2.98 s.
    """
    rows: List[Tuple[int, int, str]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split()
            if len(parts) < 3:
                continue
            try:
                start_units = int(parts[0])
                end_units = int(parts[1])
            except ValueError:
                continue
            rows.append((start_units, end_units, parts[2]))
    return rows


def find_letter_row(rows: List[Tuple[int, int, str]]) -> Optional[Tuple[int, int, str]]:
    """Return (start_units, end_units, LETTER) for the letter token, or None."""
    content = [(s, e, t) for (s, e, t) in rows if t != "sil"]
    if len(content) >= 6:
        s, e, t = content[CONTENT_WORD_LETTER_INDEX]
        if len(t) == 1 and t.isalpha():
            return (s, e, t.upper())
    for s, e, t in content:
        if len(t) == 1 and t.isalpha() and t.upper() in ALPHABET:
            return (s, e, t.upper())
    return None


def discover_video(video_dir: Path, video_id: str) -> Optional[Path]:
    """Locate a video file by stem (case-insensitive) with any common extension."""
    if not video_dir.is_dir():
        return None
    for ext in (".mpg", ".mp4", ".mov", ".avi"):
        for p in video_dir.iterdir():
            if p.stem.lower() == video_id.lower() and p.suffix.lower() == ext:
                return p
    return None

def _resize_with_pad(img, w, h):
    ih, iw = img.shape[:2]
    if ih == 0 or iw == 0: return np.zeros((h, w, 3), dtype=img.dtype)
    scale = min(w / max(iw, 1), h / max(ih, 1))
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), dtype=img.dtype)
    x_off = (w - nw) // 2; y_off = (h - nh) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
    return canvas






class MouthCropper:
    """MediaPipe FaceLandmarker in IMAGE mode (frame-independent)."""

    def __init__(self, model_path: Path) -> None:
        base = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mpv.FaceLandmarkerOptions(
            base_options=base,
            running_mode=mpv.RunningMode.IMAGE,  # not VIDEO -- VIDEO caches face position
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._det = mpv.FaceLandmarker.create_from_options(options)
        self._last_box = None

    def crop(self, bgr):
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        try:
            result = self._det.detect(img)
        except Exception as exc:
            LOGGER.warning("FaceLandmarker failed: %s", exc)
            if self._last_box is None: return None
            x1, y1, x2, y2 = self._last_box
        else:
            if not result.face_landmarks:
                if self._last_box is None: return None
                x1, y1, x2, y2 = self._last_box
            else:
                lms = result.face_landmarks[0]
                xs = [int(lms[i].x * w) for i in LMS_INDICES]
                ys = [int(lms[i].y * h) for i in LMS_INDICES]
                x1, x2 = max(0, min(xs) - 4), min(w, max(xs) + 4)
                y1, y2 = max(0, min(ys) - 4), min(h, max(ys) + 4)
                if x2 - x1 < 6 or y2 - y1 < 6:
                    if self._last_box is None: return None
                    x1, y1, x2, y2 = self._last_box
                else:
                    self._last_box = (x1, y1, x2, y2)
        mouth = bgr[y1:y2, x1:x2]
        if mouth.size == 0: return None
        resized = _resize_with_pad(mouth, MOUTH_W, MOUTH_H)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        if gray.shape != (MOUTH_H, MOUTH_W):
            gray = cv2.resize(gray, (MOUTH_W, MOUTH_H))
        return gray


def read_clip_window(video_path, start_units, end_units, cropper):
    """Extract (T, H, W) mouth crops. Timestamps in 1/25000-s units."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    s_f = max(0, min(int(round(start_units / 25000.0 * fps)), max(0, n_frames - 1)))
    e_f = max(s_f + 1, min(int(round(end_units / 25000.0 * fps)), n_frames))
    cap.set(cv2.CAP_PROP_POS_FRAMES, s_f)
    frames = []
    for _ in range(e_f - s_f):
        ret, frame = cap.read()
        if not ret or frame is None: break
        c = cropper.crop(frame)
        if c is not None: frames.append(c)
    cap.release()
    return np.stack(frames, axis=0) if frames else None


def to_fixed_length(frames, length):
    t = frames.shape[0]
    if t == length: return frames
    if t < length:
        pad = length - t; pl = pad // 2; pr = pad - pl
        parts = []
        if pl > 0: parts.append(np.repeat(frames[:1], pl, axis=0))
        parts.append(frames)
        if pr > 0: parts.append(np.repeat(frames[-1:], pr, axis=0))
        return np.concatenate(parts, axis=0)
    idx = np.linspace(0, t - 1, length, dtype=int)
    idx[0] = 0; idx[-1] = t - 1
    return frames[idx]


def process_one_speaker(align_dir, video_dir, out_dir, model_path,
                       speaker, max_per_letter, log_every):
    """Extract clips for one speaker. Returns (Counter, (sNL, sNV, sNF))."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for letter in ALPHABET: (out_dir / letter).mkdir(parents=True, exist_ok=True)
    cropper = MouthCropper(model_path)
    align_files = sorted(align_dir.glob("*.align"))
    LOGGER.info("[%s] Found %d align files", speaker, len(align_files))
    counts = Counter()
    saved_meta = []
    sNL = sNV = sNF = 0
    t0 = time.time()
    for n, align_path in enumerate(align_files, 1):
        if log_every and n % log_every == 0:
            LOGGER.info("[%s][%d/%d]  %.1f rec/s  %s", speaker, n,
                        len(align_files), n / max(time.time() - t0, 1e-6), dict(counts))
        rows = parse_align_file(align_path)
        lr = find_letter_row(rows)
        if lr is None: sNL += 1; continue
        start_u, end_u, letter = lr
        if max_per_letter and counts[letter] >= max_per_letter: continue
        vp = discover_video(video_dir, align_path.stem)
        if vp is None: sNV += 1; continue
        raw = read_clip_window(vp, start_u, end_u, cropper)
        if raw is None: sNF += 1; continue
        clip = to_fixed_length(raw, SEQ_LEN)
        if clip.shape != (SEQ_LEN, MOUTH_H, MOUTH_W):
            LOGGER.warning("Bad shape %s for %s", clip.shape, align_path.name)
            sNF += 1; continue
        idx = counts[letter] + 1
        out_path = out_dir / letter / f"clip_{speaker}_{idx:05d}.npy"
        np.save(out_path, clip.astype(np.uint8))
        saved_meta.append({"npy": str(out_path.relative_to(out_dir)),
                          "align": align_path.name, "letter": letter,
                          "speaker": speaker,
                          "start_units": int(start_u), "end_units": int(end_u)})
        counts[letter] = idx
    with open(out_dir / "index.jsonl", "a", encoding="utf-8") as fh:
        for m in saved_meta: fh.write(json.dumps(m) + "\n")
    return counts, (sNL, sNV, sNF)


def discover_speakers(data_root, requested):
    """Resolve speaker IDs. 'all' -> auto-discover s1..s34 from data_root."""
    if not requested: return []
    requested = list(requested)
    if "all" in requested:
        requested = [r for r in requested if r != "all"]
        auto = []
        for child in sorted(data_root.iterdir()):
            if not child.is_dir(): continue
            name = child.name
            if not (name.startswith("s") and name[1:].isdigit()): continue
            has_vid = (child / name).is_dir() and any((child / name).glob("*.mpg"))
            has_ali = (data_root / "alignments" / "alignments" / name).is_dir()
            if has_vid or has_ali: auto.append(name)
        requested = sorted(set(requested + auto))
    return sorted(set(requested))


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-speaker GRID alphabet clip extractor.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--speakers", nargs="+", default=["s1"],
                   help="Speaker IDs (e.g. s1 s2 s3) or 'all'  [default: s1]")
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data",
                   help="GRID dataset root  [default: data/]")
    p.add_argument("--out-dir", type=Path,
                   default=PROJECT_ROOT / "dataset" / "alphabet",
                   help="Output directory  [default: dataset/alphabet]")
    p.add_argument("--model", type=Path,
                   default=PROJECT_ROOT / "models" / "face_landmarker.task")
    p.add_argument("--max-per-letter", type=int, default=None)
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    if not args.model.exists():
        LOGGER.error("FaceLandmarker model missing: %s", args.model); return 2
    speakers = discover_speakers(args.data_root, args.speakers)
    if not speakers:
        LOGGER.error("No speakers found under %s  "
                     "(pass --speakers s1 s2 ... or --speakers all)", args.data_root)
        return 2
    LOGGER.info("Speakers: %s", speakers)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for letter in ALPHABET: (args.out_dir / letter).mkdir(parents=True, exist_ok=True)
    (args.out_dir / "index.jsonl").write_text("", encoding="utf-8")
    totals = Counter()
    sNL_t = sNV_t = sNF_t = 0
    t0 = time.time()
    for spk in speakers:
        align_dir = args.data_root / "alignments" / "alignments" / spk
        video_dir = args.data_root / spk / spk
        if not align_dir.exists():
            LOGGER.warning("[%s] no align dir, skipping", spk); continue
        if not video_dir.exists():
            LOGGER.warning("[%s] no video dir, skipping", spk); continue
        LOGGER.info("=" * 60)
        LOGGER.info("[%s] align=%s  video=%s", spk, align_dir, video_dir)
        counts, (sNL, sNV, sNF) = process_one_speaker(
            align_dir, video_dir, args.out_dir, args.model,
            spk, args.max_per_letter, args.log_every)
        sNL_t += sNL; sNV_t += sNV; sNF_t += sNF
        for l, c in counts.items(): totals[l] += c
        LOGGER.info("[%s] added %d clips  (no_letter=%d, no_video=%d, no_frames=%d)",
                    spk, sum(counts.values()), sNL, sNV, sNF)
    summary = {
        "speakers": speakers,
        "per_letter_count": {l: int(totals.get(l, 0)) for l in ALPHABET},
        "total_clips": int(sum(totals.values())),
        "skipped_no_letter": sNL_t,
        "skipped_no_video": sNV_t,
        "skipped_no_frames": sNF_t,
        "clip_shape": [SEQ_LEN, MOUTH_H, MOUTH_W],
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Per-letter counts (across all speakers) ===")
    for letter in ALPHABET: print(f"  {letter}: {totals.get(letter, 0)}")
    print(f"TOTAL: {summary['total_clips']}")
    return 0 if summary["total_clips"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
