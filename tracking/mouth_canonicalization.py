"""Similarity alignment from measured facial landmarks; no synthetic geometry."""
from __future__ import annotations
import cv2
import numpy as np


def mouth_transform(landmarks, frame_shape, size=96, margin=2.2):
    points = np.asarray(landmarks, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 292 or points.shape[1] < 2:
        raise ValueError("Full measured FaceMesh landmarks are required.")
    height, width = frame_shape[:2]
    pixels = points[:, :2] * [width, height]
    if not np.isfinite(pixels).all() or size < 1 or margin <= 0:
        raise ValueError("Invalid alignment coordinates or dimensions.")
    left, right = pixels[61], pixels[291]
    mouth_width = float(np.linalg.norm(right - left))
    if mouth_width < 2:
        raise ValueError("The measured mouth region is too small.")
    center = (pixels[13] + pixels[14]) * .5
    axis = pixels[263] - pixels[33]
    if np.linalg.norm(axis) < 2:
        axis = right - left
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
    transform = cv2.getRotationMatrix2D(tuple(center), angle, size / (mouth_width * margin))
    transform[:, 2] += np.asarray([size / 2, size / 2]) - center
    return transform


def aligned_mouth_crop(frame, landmarks, size=96, margin=2.2):
    matrix = mouth_transform(landmarks, frame.shape, size, margin)
    return cv2.warpAffine(frame, matrix, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def read_aligned_mouth_video(video_path, *, max_frames=500):
    from pathlib import Path
    from tracking.aligner import FaceTracker, TrackingConfig
    model_path = Path(__file__).resolve().parents[1] / "frontend/public/models/face_landmarker.task"
    tracker = FaceTracker(TrackingConfig(save_crops=False, save_landmarks_npz=False), model_path=model_path)
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if np.isfinite(fps) and fps > 0 else 25.
    crops, seen, consecutive_missing = [], 0, 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if seen >= max_frames:
                raise ValueError("Segment videos into clips of at most 500 frames.")
            result = tracker.process_frame(frame, seen, seen * 1000. / fps)
            seen += 1
            if result.has_face and result.landmarks is not None and "no_face" not in (result.quality_flags or []):
                crops.append(cv2.cvtColor(aligned_mouth_crop(frame, result.landmarks), cv2.COLOR_BGR2GRAY))
                consecutive_missing = 0
            else:
                consecutive_missing += 1
                if crops and consecutive_missing > 5:
                    raise ValueError("Face tracking was lost; split or rerecord the clip.")
    finally:
        capture.release()
        tracker.close()
    if len(crops) < 2 or len(crops) < seen * .8:
        raise ValueError("Too few reliably tracked mouth frames in the video.")
    return np.stack(crops), fps
