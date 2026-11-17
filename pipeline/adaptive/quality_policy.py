"""Documented engineering screening, not calibrated recognition guarantees.

Blur variance <10 gray-level squared is a quarantine heuristic. Similarity MAE
<3 on 8x8x8 ROI thumbnails flags review, never proves duplicate identity. Native
video/sample correspondence allows codec/time sampling differences: normalized
grayscale MAE <=.12 in at least 80% of samples. These versioned conservative
screens require independent validation and do not attest source or occlusion.
"""
import base64
import tempfile
from pathlib import Path
import cv2
import numpy as np

VERSION = 'private-quality-v2'


def similarity_signature(rois):
    indexes = np.linspace(0, len(rois)-1, 8, dtype=int)
    return np.stack([cv2.resize(rois[i], (8, 8), interpolation=cv2.INTER_AREA) for i in indexes]).astype(np.uint8).reshape(-1).tolist()


def near_duplicate(a, b):
    return len(a) == len(b) == 512 and float(np.abs(np.asarray(a, float)-np.asarray(b, float)).mean()) < 3


def correspondence(raw, suffix, images, timestamps):
    with tempfile.TemporaryDirectory(prefix='lipnet-correspondence-') as directory:
        path = Path(directory)/('source'+suffix); path.write_bytes(raw)
        cap = cv2.VideoCapture(str(path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError('Video timing unavailable for source correspondence.')
            wanted = [round(t * fps / 1000) for t in timestamps]
            samples = {}; index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if index in wanted:
                    samples[index] = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64))
                index += 1
                if index > 180:
                    raise ValueError('Source correspondence frame budget exceeded.')
        finally:
            cap.release()
    errors = []
    for target, source in zip(wanted, images):
        if target not in samples:
            errors.append(1.); continue
        sample = cv2.resize(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY), (64, 64)).astype(float)
        errors.append(float(np.abs(samples[target].astype(float)-sample).mean()/255))
    ratio = float(np.mean(np.asarray(errors) <= .12))
    if ratio < .8:
        raise ValueError('Source video does not correspond to supplied sampled frames/times.')
    return {'version': VERSION, 'usable_sample_ratio': ratio, 'normalized_mae': errors,
            'policy': '>=80% samples at normalized grayscale MAE<=.12; codec/timing heuristic, not device attestation'}
