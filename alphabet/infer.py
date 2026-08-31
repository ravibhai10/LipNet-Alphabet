"""Live webcam lip-reading inference for the trained CNN-BiLSTM model."""

from __future__ import annotations
import argparse
import logging
import sys
import time
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "models"))
from alphabet_model import AlphabetCNNBiLSTM, T, H, W, NUM_CLASSES

LOGGER = logging.getLogger("infer")
IDX_TO_LABEL = {i: chr(ord("A") + i) for i in range(NUM_CLASSES)}
MOUTH_W, MOUTH_H = W, H
LMS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318,
       78, 191, 80, 81, 82, 13, 312, 311, 310, 415]


class LiveMouthCropper:
    def __init__(self, model_path: Path) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import vision as mpv
        base = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        opts = mpv.FaceLandmarkerOptions(base_options=base,
            running_mode=mpv.RunningMode.IMAGE, num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False)
        self.detector = mpv.FaceLandmarker.create_from_options(opts)
        self.mp = mp
        LOGGER.info("FaceLandmarker (IMAGE) loaded")

    def crop(self, bgr: np.ndarray) -> np.ndarray | None:
        if bgr is None or bgr.size == 0:
            return None
        H_, W_ = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        res = self.detector.detect(img)
        if not res.face_landmarks:
            return None
        lms = res.face_landmarks[0]
        xs = np.array([lms[k].x * W_ for k in LMS])
        ys = np.array([lms[k].y * H_ for k in LMS])
        x1, x2 = int(max(0, xs.min() - 5)), int(min(W_, xs.max() + 5))
        y1, y2 = int(max(0, ys.min() - 5)), int(min(H_, ys.max() + 5))
        if x2 - x1 < 5 or y2 - y1 < 5:
            return None
        mouth = bgr[y1:y2, x1:x2]
        h, w = mouth.shape[:2]


def load_model(ckpt_path: Path, device: torch.device) -> AlphabetCNNBiLSTM:
    model = AlphabetCNNBiLSTM().to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck))
    model.eval()
    return model


def predict(model: AlphabetCNNBiLSTM, clip: np.ndarray,
            device: torch.device):
    x = torch.from_numpy(clip.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(2)
    with torch.no_grad():
        probs = torch.softmax(model(x.to(device)), dim=-1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return IDX_TO_LABEL[idx], float(probs[idx]), probs


def pad_or_sample(frames: list, length: int) -> np.ndarray:
    t = len(frames)
    if t == 0:
        return np.zeros((length, MOUTH_H, MOUTH_W), dtype=np.uint8)
    if t == length:
        return np.stack(frames, axis=0)
    if t < length:
        pl = (length - t) // 2
        pr = length - t - pl
        parts = []
        if pl:
            parts.append(np.repeat(frames[:1], pl, axis=0))
        parts.append(np.stack(frames, axis=0))
        if pr:
            parts.append(np.repeat(frames[-1:], pr, axis=0))
        return np.concatenate(parts, axis=0)
    idx = np.linspace(0, t - 1, length, dtype=int)
    idx[0] = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path,
                   default=PROJECT_ROOT / "models" / "alphabet_classifier.pt")
    p.add_argument("--face-model", type=Path,
                   default=PROJECT_ROOT / "models" / "face_landmarker.task")
    p.add_argument("--device", default="auto")
    p.add_argument("--clip-length", type=float, default=1.5)
    p.add_argument("--cam", type=int, default=0)
    return p.parse_args()



def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    if not args.ckpt.exists():
        LOGGER.error("Checkpoint not found: %s", args.ckpt)
        return 1
    if not args.face_model.exists():
        LOGGER.error("Face model not found: %s", args.face_model)
        return 1
    device = torch.device("cuda" if args.device == "auto" and
                          torch.cuda.is_available() else
                          (args.device if args.device != "auto" else "cpu"))
    LOGGER.info("Device: %s", device)
    model = load_model(args.ckpt, device)
    cropper = LiveMouthCropper(args.face_model)
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        LOGGER.error("Cannot open camera %d", args.cam)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    win = "Lip Alphabet - SPACE=record, Q/ESC=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 900, 600)
    state = ["IDLE"]
    frames_buf: list = []
    recording_start = [0.0]
    last_result = [None]
    fps_dq = deque(maxlen=30)
    last_t = time.time()
    target_frames = int(args.clip_length * 30)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        now = time.time()
        fps_dq.append(now - last_t)
        last_t = now
        fps = 1.0 / max(np.mean(fps_dq), 1e-6)
        live_crop = cropper.crop(frame)
        if state[0] == "RECORDING":
            if live_crop is not None:
                frames_buf.append(live_crop)
            elapsed = now - recording_start[0]
            if len(frames_buf) >= target_frames or elapsed >= args.clip_length + 0.5:
                clip = pad_or_sample(frames_buf, T)
                letter, conf, probs = predict(model, clip, device)
                LOGGER.info("PRED = %s (conf=%.2f)", letter, conf)
                last_result[0] = (letter, conf, probs)
                frames_buf.clear()
                state[0] = "RESULT"
        vis = frame.copy()
        if live_crop is not None:
            vis_crop = cv2.resize(
                cv2.cvtColor(live_crop, cv2.COLOR_GRAY2BGR), (184, 240))
            vis[10:250, vis.shape[1] - 194:vis.shape[1] - 10] = vis_crop
            cv2.putText(vis, "mouth", (vis.shape[1] - 180, 268),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if state[0] == "IDLE":
            cv2.putText(vis, "SPACE = record  |  Q / ESC = quit",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (255, 255, 255), 2)
        elif state[0] == "RECORDING":
            rec_t = now - recording_start[0]
            cv2.putText(vis,
                       f"RECORDING ({rec_t:.1f}s / {args.clip_length:.1f}s)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 0, 255), 2)
        elif state[0] == "RESULT" and last_result[0]:
            letter, conf, probs = last_result[0]
            top3 = sorted(range(NUM_CLASSES), key=lambda i: -probs[i])[:3]
            cv2.putText(vis, f"PRED: {letter} ({conf * 100:.1f}%)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                       (0, 255, 0), 2)
            for i, idx in enumerate(top3):
                label = IDX_TO_LABEL[idx]
                p = probs[idx]
                cv2.putText(vis, f"  {label}: {p * 100:.1f}%",
                           (10, 60 + i * 24),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           (180, 180, 180), 1)
            cv2.putText(vis, "SPACE = record again  |  Q / ESC = quit",
                       (10, vis.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.putText(vis, f"{fps:.0f} fps",
                   (vis.shape[1] - 80, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.imshow(win, vis)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord(" ") and state[0] in ("IDLE", "RESULT"):
            frames_buf.clear()
            recording_start[0] = time.time()
            state[0] = "RECORDING"
            last_result[0] = None
    cap.release()
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

