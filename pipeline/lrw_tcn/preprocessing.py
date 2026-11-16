"""Publisher mean-face geometry with explicit landmark-source provenance.

Adapted from the pinned author's preprocessing/dataloader.py (Apache 2.0).
MediaPipe anatomical correspondences are an approximation to its 68-point
detector; they are never described as benchmark-equivalent landmarks.
"""
from pathlib import Path
import hashlib
import cv2
import numpy as np

REFERENCE = Path(__file__).with_name('20words_mean_face.npy')
STABLE = (28, 33, 36, 39, 42, 45, 48, 54)
MP68 = (234,93,132,58,172,136,150,149,152,378,379,365,397,288,361,323,454,
        70,63,105,66,107,336,296,334,293,300,168,6,197,1,98,97,2,326,327,
        33,160,158,133,153,144,362,385,387,263,373,380,
        61,40,37,0,267,270,291,321,314,17,84,91,78,81,13,311,308,402,14,178)
WEB_CONVENTION = 'lrw-meanface-mediapipe68-v1'


def resample_timed(frames, timestamps_ms, rate=25):
    """Sample on a real-time 25 Hz grid; do not stretch every word to 29 frames."""
    times = np.asarray(timestamps_ms, dtype=np.float64)
    if times.shape != (len(frames),) or not np.isfinite(times).all():
        raise ValueError('One finite capture timestamp per frame is required.')
    gaps = np.diff(times)
    if len(times) < 12 or np.any(gaps <= 0) or np.max(gaps) > 160:
        raise ValueError('Incomplete utterance or tracking gap: capture at least 12 ordered frames without gaps over 160 ms.')
    duration = (times[-1] - times[0]) / 1000
    if not .32 <= duration <= 3.0 or (len(times)-1)/duration < 10:
        raise ValueError('Capture a single complete word lasting 0.32–3.0 seconds at 10 Hz or faster.')
    targets = np.arange(times[0], times[-1] + 1e-6, 1000/rate)
    positions = np.interp(targets, times, np.arange(len(times)))
    lo = np.floor(positions).astype(int); hi = np.minimum(lo+1, len(frames)-1)
    alpha = (positions-lo)[:,None,None]
    sequence = np.asarray(frames)
    output = np.rint(sequence[lo]*(1-alpha) + sequence[hi]*alpha).clip(0,255).astype(np.uint8)
    return output, {'source_samples':len(times), 'source_duration_ms':round(duration*1000,2),
                    'measured_capture_hz':round((len(times)-1)/duration,2), 'target_hz':rate,
                    'model_frames':len(output), 'sampling_warning': 'Temporal interpolation adds no missing visual information.'}


def align_sequence(frames, landmarks, *, convention=WEB_CONVENTION):
    if convention not in (WEB_CONVENTION, 'publisher-68-v1'):
        raise ValueError('Unsupported landmark convention.')
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (len(frames),68,2) or not np.isfinite(points).all() or np.any(points < 0) or np.any(points > 1):
        raise ValueError('Every frame needs 68 measured normalized two-dimensional landmarks in the image.')
    if hashlib.sha256(REFERENCE.read_bytes()).hexdigest()!='dbf68b2044171e1160716df7c53e8bbfaa0ee8c61fb41171d04cb6092bb81422':
        raise ValueError('Publisher mean-face fingerprint mismatch.')
    reference = np.load(REFERENCE, allow_pickle=False)
    if reference.shape != (68,2):raise ValueError('Publisher reference geometry is invalid.')
    pixels = np.stack([p*np.array([f.shape[1], f.shape[0]]) for p,f in zip(points,frames)])
    crops = []
    for index, frame in enumerate(frames):
        if frame.ndim != 2 or frame.dtype != np.uint8:raise ValueError('Measured grayscale uint8 frames required.')
        margin = min(6,index,len(frames)-1-index)
        smoothed = pixels[index-margin:index+margin+1].mean(0)
        smoothed += pixels[index].mean(0) - smoothed.mean(0)
        if np.linalg.norm(smoothed[36]-smoothed[45]) < 15:raise ValueError('Face is too small; move closer to the camera.')
        matrix, _ = cv2.estimateAffinePartial2D(smoothed[list(STABLE)], reference[list(STABLE)], method=cv2.LMEDS)
        if matrix is None or not np.isfinite(matrix).all():raise ValueError('Stable face alignment could not be measured.')
        scale = np.linalg.norm(matrix[0,:2])
        if not .15 < scale < 12:raise ValueError('Face alignment scale is unreliable.')
        # Exact author geometry: 256-pixel mean face, 96-pixel mouth patch,
        # centered on the mean of transformed lip landmarks 48..67.
        transformed = smoothed @ matrix[:,:2].T + matrix[:,2]
        center = np.clip(transformed[48:68].mean(0),48,208)
        x,y = (int(round(c)) for c in center)
        warped = cv2.warpAffine(frame,matrix,(256,256),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        crop = warped[y-48:y+48,x-48:x+48]
        if crop.shape != (96,96):raise ValueError('Incomplete mouth region.')
        crops.append(crop)
    return np.stack(crops), {'convention':convention, 'publisher_mean_face':True,
        'alignment_equivalence_verified':convention=='publisher-68-v1',
        'limitation':'MediaPipe-to-68 anatomical mapping differs from publisher detector; webcam equivalence/accuracy is unverified.' if convention==WEB_CONVENTION else 'Caller must supply genuine publisher-compatible measured landmarks.'}


def prepare_webcam(frames, landmarks, timestamps_ms, face_counts):
    if len(face_counts) != len(frames) or any(type(n) is not int or n != 1 for n in face_counts):
        raise ValueError('Exactly one face is required in every captured frame.')
    if len(frames) < 12 or len(frames)>76:raise ValueError('Capture 12–76 real frames per word.')
    if any(min(f.shape)<96 or max(f.shape)>640 for f in frames):raise ValueError('Full-face frames must be 96–640 pixels per side.')
    means=np.array([f.mean() for f in frames]); deviations=np.array([f.std() for f in frames])
    if np.median(means)<15 or np.median(means)>240 or np.median(deviations)<8:
        raise ValueError('Lighting or contrast is insufficient; use even front lighting.')
    p=np.asarray(landmarks)
    if p.shape != (len(frames),68,2):raise ValueError('Landmark/frame count mismatch.')
    widths=np.linalg.norm(p[:,48]-p[:,54],axis=1)
    apertures=np.linalg.norm(p[:,62]-p[:,66],axis=1)/np.maximum(widths,1e-6)
    if np.ptp(apertures)<.025:raise ValueError('No clear mouth movement; capture one spoken word.')
    crops,alignment=align_sequence(frames,landmarks)
    sampled,temporal=resample_timed(crops,timestamps_ms)
    return sampled,{'alignment':alignment,'temporal':temporal,'lighting_mean':float(np.median(means)),
                    'mouth_aperture_range':float(np.ptp(apertures)),'quality_policy':'Heuristics; not calibrated accuracy guarantees.'}
