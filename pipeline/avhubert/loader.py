"""Frozen CPU AV-HuBERT features using the official video-only source contract.

Raw videos are aligned with measured MediaPipe landmarks. Already aligned mouth
ROI videos can be supplied explicitly with input_kind="mouth_roi". The encoder
does not contain a trained letter/word classifier or a sentence decoder.
"""
from __future__ import annotations
import logging
import threading
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from src.video_training import resample_temporal

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

class ModelUnavailableError(RuntimeError):
    """Required trained weights or the official runtime are unavailable."""

class AVHubertLoader:
    def __init__(self, ckpt_path: str | Path | None = None, *, threads=4):
        self.ckpt_path = Path(ckpt_path) if ckpt_path is not None else None
        self._models = self._cfg = self._task = None
        self.available = False
        self.d_out = 768
        self.threads = threads
        self._lock = threading.Lock()

    @property
    def _models_root(self):
        return PROJECT_ROOT / "models" / "external" / "avhubert"

    def _resolve_ckpt(self):
        if self.ckpt_path is not None:
            if not self.ckpt_path.is_file():
                raise ModelUnavailableError(f"Explicit AV-HuBERT checkpoint not found: {self.ckpt_path}")
            return self.ckpt_path
        for name in ("base_vox_iter5.pt", "base_lrs3_iter5.pt", "avhubert_base.pt"):
            candidate = self._models_root / name
            if candidate.is_file():
                return candidate
        raise ModelUnavailableError(
            "No supported pretrained AV-HuBERT checkpoint found. Download the official "
            "Base checkpoint to models/external/avhubert/base_vox_iter5.pt."
        )

    def load(self):
        with self._lock:
            if self.available:
                return self
            self.ckpt_path = self._resolve_ckpt()
            try:
                # Import fairseq BEFORE adding the sibling user module directory:
                # the checkout also contains a namespace directory named fairseq.
                import fairseq
                from fairseq import checkpoint_utils
                from src.cpu_utils import configure_cpu_runtime
                configure_cpu_runtime(self.threads)
                user_dir = PROJECT_ROOT / "external" / "av_hubert" / "avhubert"
                if not user_dir.is_dir():
                    raise ModelUnavailableError("The official external/av_hubert/avhubert source is missing.")
                fairseq.utils.import_user_module(SimpleNamespace(user_dir=str(user_dir)))
                models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
                    [str(self.ckpt_path)], arg_overrides={}
                )
                if len(models) != 1 or not hasattr(models[0], "extract_finetune"):
                    raise ModelUnavailableError("Expected one pretrained AV-HuBERT feature encoder.")
                model = models[0].cpu().float().eval()
                model.requires_grad_(False)
                dimension = getattr(model, "encoder_embed_dim", None)
                if dimension is None:
                    dimension = getattr(getattr(model, "cfg", None), "encoder_embed_dim", None)
                if dimension is None:
                    raise ModelUnavailableError("The encoder did not report its embedding dimension.")
                dimension = int(dimension)
                if dimension not in (768, 1024):
                    raise ModelUnavailableError(f"Unexpected AV-HuBERT embedding width: {dimension}")
                self._models, self._cfg, self._task = models, cfg, task
                self.d_out = dimension
                self.available = True
            except ModelUnavailableError:
                raise
            except Exception as exc:
                raise ModelUnavailableError(
                    f"Cannot load official AV-HuBERT on CPU: {exc}. "
                    "Use the project's .venv-avhubert-cpu environment."
                ) from exc
            logger.info("AV-HuBERT loaded on CPU: %s (%d dimensions)", self.ckpt_path, self.d_out)
            return self

    @property
    def models(self):
        if not self.available:
            raise ModelUnavailableError("AV-HuBERT checkpoint not loaded.")
        return self._models

    @property
    def task(self):
        if not self.available:
            raise ModelUnavailableError("AV-HuBERT checkpoint not loaded.")
        return self._task

    def preprocessing(self):
        self.load()
        return {
            "fps": 25, "image_crop_size": int(self.task.cfg.image_crop_size),
            "image_mean": float(self.task.cfg.image_mean), "image_std": float(self.task.cfg.image_std),
            "channels": 1, "alignment": "measured-eye-axis-mouth-center-v1",
        }

    def prepare_rois(self, mouth_rois, *, source_fps=25., target_frames=None):
        if target_frames is not None and (not isinstance(target_frames, int) or not 1 <= target_frames <= 500):
            raise ValueError("target_frames must be an integer between 1 and 500.")
        self.load()
        frames = np.asarray(mouth_rois)
        if frames.ndim == 4 and frames.shape[-1] == 3:
            frames = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames])
        if frames.ndim != 3 or not 1 <= len(frames) <= 500:
            raise ValueError("Expected 1-500 grayscale mouth ROI frames [T,H,W].")
        if not np.isfinite(frames).all() or not np.isfinite(source_fps) or source_fps <= 0:
            raise ValueError("Invalid ROI values or frame rate.")
        integer_input = np.issubdtype(frames.dtype, np.integer)
        array = frames.astype(np.float32) / (255. if integer_input else 1.)
        if array.min() < 0 or array.max() > 1:
            raise ValueError("Floating mouth ROIs must be normalized to [0,1].")
        length = target_frames or max(1, round(len(array) * 25. / source_fps))
        array = resample_temporal(array, length)
        config = self.preprocessing()
        size = config["image_crop_size"]
        height, width = array.shape[1:]
        if min(height, width) < size:
            factor = size / min(height, width)
            array = np.stack([cv2.resize(frame, (round(width * factor), round(height * factor))) for frame in array])
            height, width = array.shape[1:]
        top, left = (height - size) // 2, (width - size) // 2
        array = array[:, top:top + size, left:left + size]
        array = (array - config["image_mean"]) / config["image_std"]
        import torch
        # Official source: video [batch,1,time,height,width], audio absent.
        return torch.from_numpy(np.ascontiguousarray(array))[None, None]

    def encode_frames(self, mouth_rois, *, source_fps=25., target_frames=None):
        import torch
        video = self.prepare_rois(mouth_rois, source_fps=source_fps, target_frames=target_frames)
        with torch.inference_mode():
            # The official pretraining extract_features() requires both
            # modalities. Its downstream extract_finetune() path explicitly
            # supports missing audio and is also valid with frozen weights.
            features, _ = self.models[0].extract_finetune(
                source={"video": video, "audio": None}, padding_mask=None, mask=False
            )
        if features.ndim != 3 or features.shape[0] != 1 or features.shape[-1] != self.d_out:
            raise ModelUnavailableError(f"Unexpected feature tensor shape: {tuple(features.shape)}")
        result = features[0].detach().cpu().float().numpy()
        if not np.isfinite(result).all():
            raise ModelUnavailableError("AV-HuBERT produced invalid features.")
        return result

    def encode(self, video_path: str | Path, *, input_kind="raw_video", target_frames=None):
        if input_kind == "raw_video":
            from tracking.mouth_canonicalization import read_aligned_mouth_video
            frames, fps = read_aligned_mouth_video(video_path)
        elif input_kind == "mouth_roi":
            capture = cv2.VideoCapture(str(video_path))
            fps = capture.get(cv2.CAP_PROP_FPS)
            fps = fps if np.isfinite(fps) and fps > 0 else 25.
            frames = []
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if len(frames) >= 500:
                        raise ValueError("Segment videos into clips of at most 500 frames.")
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            finally:
                capture.release()
            if not frames:
                raise ValueError(f"Cannot decode mouth ROI video: {video_path}")
        else:
            raise ValueError("input_kind must be raw_video or mouth_roi.")
        return self.encode_frames(frames, source_fps=fps, target_frames=target_frames)
