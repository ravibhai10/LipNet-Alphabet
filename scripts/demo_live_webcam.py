#!/usr/bin/env python
"""Real-Time Live Webcam Lip-Reading Demo UI (OpenCV + MediaPipe).

High-Accuracy Edition:
- Affine Landmark Canonicalization: Aligns face & lips horizontally to neutralize head tilt.
- Visual Voice Activity Detection (V-VAD): Detects active speech vs. resting mouth to eliminate false positives.
- Sliding Window EMA Logit Smoothing: Smooths class probabilities across time to remove frame-by-frame flicker.
- Top-3 Candidate OSD: Overlays top-3 predictions with confidence bars.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import schema
from models.alphabet_model import AlphabetCNNBiLSTM

TARGET_FRAMES = 29
TARGET_HEIGHT = 140
TARGET_WIDTH = 46

# Lip and face alignment landmark indices (MediaPipe FaceMesh)
LIP_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
UPPER_LIP_CENTER = 13
LOWER_LIP_CENTER = 14
LEFT_MOUTH_CORNER = 78
RIGHT_MOUTH_CORNER = 308

LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263


def compute_mouth_openness(landmarks, w: int, h: int) -> float:
    """Compute normalized vertical mouth opening relative to mouth width."""
    upper = np.array([landmarks[UPPER_LIP_CENTER].x * w, landmarks[UPPER_LIP_CENTER].y * h])
    lower = np.array([landmarks[LOWER_LIP_CENTER].x * w, landmarks[LOWER_LIP_CENTER].y * h])
    left = np.array([landmarks[LEFT_MOUTH_CORNER].x * w, landmarks[LEFT_MOUTH_CORNER].y * h])
    right = np.array([landmarks[RIGHT_MOUTH_CORNER].x * w, landmarks[RIGHT_MOUTH_CORNER].y * h])

    vert_dist = np.linalg.norm(upper - lower)
    horiz_dist = np.linalg.norm(left - right)
    return float(vert_dist / max(horiz_dist, 1e-4))


def extract_mouth_roi_canonical(
    frame: np.ndarray,
    face_mesh: mp.solutions.face_mesh.FaceMesh,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], float]:
    """Extract affine-aligned mouth ROI and compute instant mouth openness."""
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    if not res.multi_face_landmarks:
        return None, None, 0.0

    landmarks = res.multi_face_landmarks[0].landmark
    openness = compute_mouth_openness(landmarks, w, h)

    # 1. Compute head roll angle via eye corners to neutralize head tilt
    lx, ly = landmarks[LEFT_EYE_CORNER].x * w, landmarks[LEFT_EYE_CORNER].y * h
    rx, ry = landmarks[RIGHT_EYE_CORNER].x * w, landmarks[RIGHT_EYE_CORNER].y * h
    d_x = rx - lx
    d_y = ry - ly
    angle = np.degrees(np.arctan2(d_y, d_x))

    # 2. Canonical rotation around face center if tilted > 2 degrees
    if abs(angle) > 2.0:
        center = (int(w / 2), int(h / 2))
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_frame = cv2.warpAffine(frame, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
        # Re-compute landmark coordinates in rotated frame
        rot_rgb = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2RGB)
        res_rot = face_mesh.process(rot_rgb)
        if res_rot.multi_face_landmarks:
            landmarks = res_rot.multi_face_landmarks[0].landmark
            frame_to_crop = rotated_frame
        else:
            frame_to_crop = frame
    else:
        frame_to_crop = frame

    xs = [landmarks[i].x * w for i in LIP_LANDMARKS]
    ys = [landmarks[i].y * h for i in LIP_LANDMARKS]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bw, bh = max_x - min_x, max_y - min_y

    x1 = int(max(0, min_x - bw * 0.35))
    y1 = int(max(0, min_y - bh * 0.45))
    x2 = int(min(w, max_x + bw * 0.35))
    y2 = int(min(h, max_y + bh * 0.45))

    if x2 <= x1 or y2 <= y1:
        return None, None, openness

    crop = frame_to_crop[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None, openness

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

    return resized, (x1, y1, x2, y2), openness


class PredictionStabilizer:
    """Exponential Moving Average (EMA) smoother and confidence gate."""

    def __init__(self, num_classes: int, alpha: float = 0.65, min_confidence: float = 0.28):
        self.num_classes = num_classes
        self.alpha = alpha
        self.min_confidence = min_confidence
        self.smoothed_probs: Optional[np.ndarray] = None
        self.stable_label: Optional[str] = None
        self.consecutive_count = 0
        self.candidate_label: Optional[str] = None

    def update(self, probs: np.ndarray, is_word_model: bool) -> Tuple[str, float, List[Tuple[str, float]]]:
        if self.smoothed_probs is None:
            self.smoothed_probs = probs.copy()
        else:
            self.smoothed_probs = self.alpha * probs + (1.0 - self.alpha) * self.smoothed_probs

        top3_indices = np.argsort(self.smoothed_probs)[-3:][::-1]
        top3 = []
        for idx in top3_indices:
            label_str = schema.get_word(int(idx)) if is_word_model else schema.idx_to_letter(int(idx))
            top3.append((label_str, float(self.smoothed_probs[idx])))

        best_idx = int(top3_indices[0])
        best_prob = float(self.smoothed_probs[best_idx])
        best_label = top3[0][0]

        # Confidence gating & hysteresis
        if best_prob >= self.min_confidence:
            if best_label == self.candidate_label:
                self.consecutive_count += 1
                if self.consecutive_count >= 2:
                    self.stable_label = best_label
            else:
                self.candidate_label = best_label
                self.consecutive_count = 1

        label_to_show = self.stable_label or best_label
        return label_to_show, best_prob * 100.0, top3

    def reset(self):
        self.smoothed_probs = None
        self.consecutive_count = 0


def run_live_demo(
    checkpoint_path: str,
    is_word_model: bool = False,
    camera_source: int | str = 0,
) -> None:
    """Run real-time lip reading inference with V-VAD and EMA logit stabilization."""
    device = torch.device("cpu")
    num_classes = schema.NUM_CLASSES if is_word_model else 26

    model = AlphabetCNNBiLSTM(num_classes=num_classes)
    ckpt_file = Path(checkpoint_path)
    if ckpt_file.exists():
        state = torch.load(str(ckpt_file), map_location=device)
        state_dict = state.get("model_state_dict", state.get("state_dict", state))
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint: {ckpt_file} (classes: {num_classes})")
    else:
        print(f"Warning: Checkpoint '{checkpoint_path}' not found. Using initialized weights for testing.")

    model.eval()

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        cam_id = int(camera_source)
    except ValueError:
        cam_id = camera_source

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"Error: Could not access video source: {camera_source}")
        face_mesh.close()
        return

    frame_buffer = deque(maxlen=TARGET_FRAMES)
    openness_buffer = deque(maxlen=TARGET_FRAMES)
    stabilizer = PredictionStabilizer(num_classes=num_classes, alpha=0.65)

    top_pred_text = "Waiting for speech..."
    confidence = 0.0
    top3_list: List[Tuple[str, float]] = []
    is_speaking = False

    mode_label = "500-Word LRW" if is_word_model else "A-Z Alphabet"
    print(f"\n=======================================================")
    print(f" Starting Live Lip-Reading Demo [{mode_label}]")
    print(f" Features: Head-Tilt Normalization + V-VAD + Logit Smoothing")
    print(f" Press 'q' on the preview window to exit.")
    print(f"=======================================================\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if isinstance(cam_id, int):
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            roi, bbox, openness = extract_mouth_roi_canonical(frame, face_mesh)

            if roi is not None and bbox is not None:
                x1, y1, x2, y2 = bbox
                # Normalize ROI pixel values to [0.0, 1.0]
                normalized_roi = roi.astype(np.float32) / 255.0
                frame_buffer.append(normalized_roi)
                openness_buffer.append(openness)

                # Visual Voice Activity Detection (V-VAD): check mouth motion energy
                if len(openness_buffer) >= 10:
                    openness_std = float(np.std(list(openness_buffer)))
                    openness_mean = float(np.mean(list(openness_buffer)))
                    # Active speaking creates dynamic variation in mouth aperture
                    is_speaking = (openness_std > 0.022) or (openness_mean > 0.28)
                else:
                    is_speaking = False

                # Box color: Green when speaking detected, Cyan when mouth idle
                box_color = (0, 255, 0) if is_speaking else (255, 200, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                # Thumbnail in top-right corner
                if h >= 320 and w >= 150:
                    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                    roi_thumb = cv2.resize(roi_bgr, (92, 280))
                    y_start, y_end = 20, 300
                    x_start, x_end = w - 112, w - 20
                    frame[y_start:y_end, x_start:x_end] = roi_thumb
                    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), box_color, 1)

            if len(frame_buffer) == TARGET_FRAMES:
                if is_speaking:
                    arr = np.stack(list(frame_buffer), axis=0)  # [T, H, W]
                    tensor_input = torch.from_numpy(arr).unsqueeze(1).unsqueeze(0)  # [1, T, 1, H, W]

                    with torch.no_grad():
                        logits = model(tensor_input)
                        probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
                        top_pred_text, confidence, top3_list = stabilizer.update(probs, is_word_model)
                else:
                    top_pred_text = "Idle (Ready to speak)"
                    confidence = 0.0
                    stabilizer.reset()

            # Render Premium On-Screen Display (OSD) Overlay
            osd_w = 480
            osd_h = 135 if top3_list else 95
            cv2.rectangle(frame, (20, 20), (20 + osd_w, 20 + osd_h), (15, 15, 15), -1)
            cv2.rectangle(frame, (20, 20), (20 + osd_w, 20 + osd_h), (80, 80, 80), 1)

            pred_color = (0, 255, 0) if is_speaking and confidence > 30 else (200, 200, 200)
            cv2.putText(frame, f"Pred: {top_pred_text}", (35, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, pred_color, 2)

            status_str = f"Conf: {confidence:.1f}% | V-VAD: {'ACTIVE' if is_speaking else 'IDLE'} | {mode_label}"
            cv2.putText(frame, status_str, (35, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            # Top 3 Candidates
            if is_speaking and top3_list:
                top3_str = "  ".join([f"{w}: {p*100:.0f}%" for w, p in top3_list])
                cv2.putText(frame, f"Top-3: {top3_str}", (35, 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 1)

            cv2.imshow("LipNet-AI — Live Showcase", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Webcam Lip-Reading Showcase")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/alphabet_classifier.pt",
        help="Path to trained PyTorch model checkpoint (.pt)",
    )
    parser.add_argument(
        "--word-model",
        action="store_true",
        help="Set flag when running 500-word LRW model",
    )
    parser.add_argument(
        "--camera-id",
        default="0",
        help="Camera device index (default: 0) or path to input video file",
    )
    args = parser.parse_args()

    run_live_demo(
        checkpoint_path=args.checkpoint,
        is_word_model=args.word_model,
        camera_source=args.camera_id,
    )
