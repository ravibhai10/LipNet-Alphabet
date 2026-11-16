"""Decode lossless source frames and independently measure face/mouth quality."""
from __future__ import annotations
import base64
import hashlib
import io
import json
import struct
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

PREPROCESSING = {'id': 'adaptive-eyeaxis96-png-v1', 'size': 96, 'margin': 2.2,
                 'alignment': 'eye-axis', 'gray': 'opencv-BGR2GRAY', 'interpolation': 'linear',
                 'border': 'reject', 'source': 'lossless browser canvas PNG; no audio'}
MAX_BYTES = 64 * 1024 * 1024


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def track_frames(images, root):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(root / 'frontend/public/models/face_landmarker.task')),
        running_mode=vision.RunningMode.IMAGE, num_faces=2)
    with vision.FaceLandmarker.create_from_options(options) as tracker:
        return [tracker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).face_landmarks for frame in images]


def validate_capture(row, root: Path, tracker=track_frames):
    from tracking.mouth_canonicalization import mouth_transform
    if row['consent'] is not True:
        raise ValueError('Explicit recording and research-training consent is required.')
    stamp = datetime.fromisoformat(row['recorded_at'].replace('Z', '+00:00'))
    if stamp.tzinfo is None or abs((datetime.now(timezone.utc)-stamp).total_seconds()) > 86400:
        raise ValueError('Use a timezone-aware timestamp within the last day.')
    frames, times = row['frames'], row['timestamps_ms']
    if not 8 <= len(frames) <= 76 or len(times) != len(frames):
        raise ValueError('Capture 8–76 source frames with matching timestamps.')
    if any(type(t) not in (int, float) or not np.isfinite(t) or t < 0 or
           (i and t <= times[i-1]) for i, t in enumerate(times)):
        raise ValueError('Timestamps must be finite and strictly increasing.')
    duration = times[-1]-times[0]
    if not 0 < duration <= 5000 or max(np.diff(times)) > 250:
        raise ValueError('Capture exceeds five seconds or contains a frame gap.')
    fps = (len(times)-1)*1000/duration
    if fps < 8:
        raise ValueError('At least eight actual source samples per second are required.')
    images, raw_frames = [], []
    geometry = None
    for encoded in frames:
        if not encoded.startswith('data:image/png;base64,') or len(encoded) > 4_000_000:
            raise ValueError('Supply bounded, original lossless PNG frames.')
        try:
            raw = base64.b64decode(encoded.split(',', 1)[1], validate=True)
        except ValueError as exc:
            raise ValueError('Invalid PNG encoding.') from exc
        if len(raw) < 24 or raw[:8] != b'\x89PNG\r\n\x1a\n':
            raise ValueError('Invalid PNG source.')
        width, height = struct.unpack('>II', raw[16:24])
        if not 96 <= width <= 1280 or not 96 <= height <= 720:
            raise ValueError('Native source resolution must be between 96px and 1280×720. Select a supported camera mode.')
        if geometry and geometry != (width, height):
            raise ValueError('Source geometry changed during recording.')
        geometry = (width, height)
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape != (height, width, 3):
            raise ValueError('Source PNG cannot be decoded at its stated geometry.')
        raw_frames.append(raw); images.append(image)
    if sum(map(len, raw_frames)) > MAX_BYTES:
        raise ValueError('Recording exceeds 64 MiB. Record a shorter utterance.')
    landmarks = tracker(images, root)
    rois, evidence, warnings = [], [], []
    for image, faces in zip(images, landmarks):
        if len(faces) != 1:
            raise ValueError('Exactly one independently detected face is required in every frame.')
        points = np.array([(p.x, p.y) for p in faces[0]])
        matrix = mouth_transform(points, image.shape)
        inverse = cv2.invertAffineTransform(matrix)
        corners = np.array([[0, 0, 1], [95, 0, 1], [0, 95, 1], [95, 95, 1]]) @ inverse.T
        if (corners < 0).any() or (corners[:, 0] >= image.shape[1]).any() or (corners[:, 1] >= image.shape[0]).any():
            raise ValueError('The complete mouth crop must lie inside the source image.')
        roi = cv2.cvtColor(cv2.warpAffine(image, matrix, (96, 96)), cv2.COLOR_BGR2GRAY)
        brightness, contrast = float(roi.mean()), float(roi.std())
        if not 15 <= brightness <= 240 or contrast < 8:
            warnings.append('low_light_or_contrast')
        evidence.append({'face_count': 1, 'mouth_landmarks': True, 'transform': matrix.tolist(),
                         'brightness': brightness, 'contrast': contrast,
                         'occlusion': 'not_measured', 'landmarks': points.tolist()})
        rois.append(roi)
    if len(rois) != len(images):
        raise ValueError('Independent tracking did not cover the complete sequence.')
    rois = np.stack(rois)
    if float(np.abs(np.diff(rois.astype(float), axis=0)).mean()) < .5:
        warnings.append('short_or_low_motion')
    source_hash = digest(b''.join(hashlib.sha256(r).digest() for r in raw_frames))
    pixels_hash = digest(b''.join(image.tobytes() for image in images))
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED) as archive:
        for i, raw in enumerate(raw_frames):
            archive.writestr(f'{i:04d}.png', raw)
    roi_stream = io.BytesIO(); np.save(roi_stream, rois, allow_pickle=False)
    metadata = {k: v for k, v in row.items() if k != 'frames'}
    metadata.update(source_sha256=source_hash, pixels_sha256=pixels_hash, source_frame_hashes=[digest(r) for r in raw_frames],
                    frame_count=len(images), geometry=list(geometry), captured_fps=fps, duration_ms=duration,
                    preprocessing=PREPROCESSING, quality=evidence, warnings=sorted(set(warnings)),
                    landmarker_sha256=digest((root/'frontend/public/models/face_landmarker.task').read_bytes()),
                    roi_sha256=digest(roi_stream.getvalue()), device_attestation=False,
                    recording_format='ordered original PNG sequence; exact timestamps; lossless ZIP')
    return metadata, stream.getvalue(), roi_stream.getvalue()
