"""LivePipeline - real-time webcam -> Phase 2 -> Phase 3 -> Phase 4 -> Phase 5.

Wires the existing, frozen ML components into a streaming pipeline that
consumes CameraService frames and produces a stabilized A-Z prediction.

Pipeline:
    CameraService.frames (BGR)
      -> Phase 2:  MediaPipe FaceLandmarker (real landmarks only)
      -> Phase 3:  AppearanceExtractor (ResNet stem)   ->  [T, 96]
                   GeometryExtractor (MLP)             ->  [T, 64]
                   MotionExtractor (savgol + MLP)      ->  [T, 64]
                   GatedFusion                          ->  [T, 192]
      -> Phase 4:  TemporalEncoder (4x Conformer)      ->  [T, 192]
      -> Phase 5:  MultiTaskAlphabetModel CTC head     ->  [T, 27]
      -> GreedyCTC + uncertainty + abstain             -> PredictionResult

No duplicate decoders, no second implementation: every call goes through
the canonical alphabet/ modules.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from app.services.camera_service import (
    CameraService, CameraFrame, get_camera_service,
    CameraUnavailableError,
)
from app.services.temporal_buffer import (
    BufferedFrame, TemporalBuffer, BufferStats, PredictionStabilizer,
)
from app.services.inference_service import (
    InferenceService, ModelHandle, PredictionResult,
    get_inference_service,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    """Snapshot of the live pipeline, polled by the UI."""
    state: str = "IDLE"             # IDLE | NO_MODEL | NO_FACE | PROCESSING | CONFIDENT | LOW_CONFIDENCE | ABSTAIN | ERROR
    letter: str = ""
    confidence: float = 0.0
    entropy: float = 0.0
    margin: float = 0.0
    top_k: List[Tuple[str, float]] = field(default_factory=list)
    detail: str = ""
    error: Optional[str] = None
    buffer_size: int = 0
    buffer_capacity: int = 0
    buffer_has_min: bool = False
    face_recently: bool = False
    fps_camera: float = 0.0
    fps_inference: float = 0.0
    last_inference_ms: float = 0.0
    last_phase2_ms: float = 0.0
    last_phase3_ms: float = 0.0
    last_phase4_ms: float = 0.0
    last_phase5_ms: float = 0.0
    stabilize_count: int = 0
    frames_inferred: int = 0
    frames_with_face: int = 0


class LivePipeline:
    """Real webcam -> A-Z pipeline.

    Worker thread consumes CameraService frames, runs Phase 2-5 on each
    frame, and stores the latest result in `self._state`. The UI only
    reads `get_state()`.
    """

    def __init__(
        self,
        inference: Optional[InferenceService] = None,
        camera: Optional[CameraService] = None,
        buffer_max: int = 32,
        buffer_min: int = 8,
        no_face_timeout_s: float = 1.0,
    ):
        self._inference = inference or get_inference_service()
        self._camera = camera or get_camera_service()
        self._buffer = TemporalBuffer(
            max_frames=buffer_max,
            min_frames=buffer_min,
            no_face_timeout_s=no_face_timeout_s,
        )
        self._stabilizer = PredictionStabilizer(required_count=3, margin=0.05)

        self._handle: Optional[ModelHandle] = None
        self._ckpt_dir: Optional[Path] = None

        # Phase 2-3 components (lazy init, real-only)
        self._face_tracker = None
        self._aligner = None
        self._appearance_extractor = None
        self._geometry_extractor = None
        self._motion_extractor = None
        self._fusion = None
        self._phase_components_lock = threading.Lock()
        self._phase_init_failed: Optional[str] = None

        # Streaming state
        self._state = PipelineState()
        self._state_lock = threading.Lock()
        self._latest_result: PredictionResult = PredictionResult()
        self._frames_inferred: int = 0
        self._frames_with_face: int = 0
        self._last_inference_at: float = 0.0
        self._prev_inference_at_for_fps: float = 0.0
        self._inf_interval_s: float = 0.30     # ~3 Hz inference cadence
        self._last_scalars: Optional[np.ndarray] = None

        # Worker
        self._worker_thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self, ckpt_dir: Path) -> None:
        """Load (or reuse cached) checkpoint. Throws on hard failure."""
        with self._lock:
            new_dir = str(Path(ckpt_dir).resolve())
            if self._handle is not None and self._ckpt_dir is not None \
                    and str(self._ckpt_dir.resolve()) == new_dir:
                return  # already loaded
            self._handle = self._inference.load_checkpoint(ckpt_dir)
            self._ckpt_dir = Path(ckpt_dir).resolve()

        # Make sure we don't carry over an old prediction
        self._stabilizer.reset()
        self._latest_result = PredictionResult()
        with self._state_lock:
            self._state.error = None
            self._state.detail = f"Loaded {self._ckpt_dir.name}"

    def start(self) -> None:
        """Start the camera + worker."""
        if self._handle is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        if self._running.is_set():
            return
        try:
            self._init_phase_components()
        except Exception as exc:
            with self._state_lock:
                self._state.state = "ERROR"
                self._state.error = str(exc)
                self._state.detail = "Phase 2-3 init failed"
            raise
        try:
            self._camera.start()
        except CameraUnavailableError as e:
            with self._state_lock:
                self._state.state = "ERROR"
                self._state.error = str(e)
                self._state.detail = "Camera unavailable"
            raise
        self._buffer.clear()
        self._stabilizer.reset()
        self._last_scalars = None
        self._frames_inferred = 0
        self._frames_with_face = 0
        self._running.set()
        self._worker_thread = threading.Thread(
            target=self._run_loop, daemon=True, name="live-pipeline",
        )
        self._worker_thread.start()

    def stop(self) -> None:
        """Stop worker, camera, and clear buffer."""
        self._running.clear()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=3.0)
            self._worker_thread = None
        try:
            self._camera.stop()
        except Exception as exc:
            logger.warning("Camera stop error: %s", exc)
        self._buffer.clear()
        self._stabilizer.reset()
        self._last_scalars = None
        with self._state_lock:
            self._state.state = "IDLE"
            self._state.detail = "Stopped"

    def is_running(self) -> bool:
        return self._running.is_set()

    def get_state(self) -> PipelineState:
        with self._state_lock:
            return PipelineState(
                state=self._state.state,
                letter=self._state.letter,
                confidence=self._state.confidence,
                entropy=self._state.entropy,
                margin=self._state.margin,
                top_k=list(self._state.top_k),
                detail=self._state.detail,
                error=self._state.error,
                buffer_size=self._state.buffer_size,
                buffer_capacity=self._state.buffer_capacity,
                buffer_has_min=self._state.buffer_has_min,
                face_recently=self._state.face_recently,
                fps_camera=self._state.fps_camera,
                fps_inference=self._state.fps_inference,
                last_inference_ms=self._state.last_inference_ms,
                last_phase2_ms=self._state.last_phase2_ms,
                last_phase3_ms=self._state.last_phase3_ms,
                last_phase4_ms=self._state.last_phase4_ms,
                last_phase5_ms=self._state.last_phase5_ms,
                stabilize_count=self._state.stabilize_count,
                frames_inferred=self._state.frames_inferred,
                frames_with_face=self._state.frames_with_face,
            )

    def save_test_recording(self, out_dir: Path) -> int:
        """Save the most recent face frames to test_recordings/ (NOT to training)."""
        from datetime import datetime, timezone
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        frames = self._buffer.frames_chronological()
        if not frames:
            return 0
        for f in frames:
            if f.landmarks is None:
                continue
            np.savez_compressed(
                out_dir / f"frame_{f.frame_id:06d}.npz",
                landmarks=f.landmarks,
                mouth_landmarks=f.mouth_landmarks
                if f.mouth_landmarks is not None
                else np.zeros((0, 2)),
            )
        meta = {
            "recording_id": f"live_test_{int(time.time())}",
            "letter": "?",
            "speaker_id": "live_test_user",
            "session_id": "live_test",
            "data_source": "real_camera",
            "landmark_source": "mediapipe",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "frame_count": len(frames),
        }
        (out_dir / "metadata.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8",
        )
        return len(frames)

    # ------------------------------------------------------------------
    # Phase 2-3 component init (real only, lazy)
    # ------------------------------------------------------------------

    def _init_phase_components(self) -> None:
        with self._phase_components_lock:
            if self._face_tracker is not None:
                return
            from tracking.aligner import (
                FaceTracker, MouthAligner, TrackingConfig,
            )
            from repr.appearance import AppearanceConfig, AppearanceExtractor
            from repr.geometry import GeometryConfig, GeometryExtractor
            from repr.motion import MotionConfig, MotionExtractor
            from repr.fusion import FusionConfig, GatedFusion

            tracking_cfg = TrackingConfig()
            # Canonical 20-point mouth set (matches GeometryExtractor)
            tracking_cfg.mouth_landmark_indices = [
                78, 74, 81, 13, 312, 308, 402, 14, 61, 291,
                39, 37, 0, 267, 269, 391, 61, 0, 17, 291,
            ]
            tracking_cfg.num_mouth_landmarks = 20
            self._face_tracker = FaceTracker(tracking_cfg)
            self._aligner = MouthAligner(tracking_cfg)

            app_cfg = AppearanceConfig()
            geo_cfg = GeometryConfig()
            mot_cfg = MotionConfig(optical_flow_enabled=False)
            self._appearance_extractor = AppearanceExtractor(app_cfg, device="cpu")
            self._geometry_extractor = GeometryExtractor(geo_cfg, device="cpu")
            self._motion_extractor = MotionExtractor(mot_cfg, device="cpu")

            fusion_cfg = FusionConfig(use_context=False)
            self._fusion = GatedFusion(fusion_cfg).eval()

            self._load_dataset_stats()

    def _load_dataset_stats(self) -> None:
        # Try to use cached mean/std from any existing representation in
        # runs/repr_cache/. Without these, the per-stream MLPs still
        # work but accuracy degrades. We don't fail if absent.
        try:
            root = Path("alphabet_dataset")
            if not (root / "manifest.jsonl").exists():
                return
            cache_dir = root / "runs" / "repr_cache"
            if not cache_dir.exists():
                cache_dir = root / "repr_cache"
            if not cache_dir.exists():
                return
            sample_files = sorted(cache_dir.rglob("geometry.npy"))[:5]
            if not sample_files:
                return
            stacked = np.stack([np.load(f) for f in sample_files], axis=0)
            mean = stacked.mean(axis=(0, 1))
            std = np.maximum(stacked.std(axis=(0, 1)), 1e-6)
            self._geometry_extractor.scalars_mean = mean
            self._geometry_extractor.scalars_std = std
            logger.info("Loaded geometry stats from %d files", len(sample_files))
        except Exception as exc:
            logger.debug("Could not load dataset stats: %s", exc)

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        last_inference_at = 0.0
        while self._running.is_set():
            frame = self._camera.read()
            if frame is None:
                time.sleep(0.01)
                self._refresh_state_idle()
                continue

            try:
                t_p2_start = time.perf_counter()
                tracking = self._face_tracker.process_frame(
                    frame.bgr,
                    frame_index=frame.frame_id,
                    timestamp_ms=frame.timestamp * 1000.0,
                )
                p2_ms = (time.perf_counter() - t_p2_start) * 1000.0

                if not tracking.has_face or tracking.landmarks is None \
                        or tracking.appearance_crop is None:
                    self._buffer.clear()
                    self._last_scalars = None
                    with self._state_lock:
                        self._state.state = "NO_FACE"
                        self._state.detail = "No face detected"
                        self._state.last_phase2_ms = p2_ms
                    continue

                # Phase 3
                t_p3_start = time.perf_counter()
                fused_vec, mouth_lm = self._phase3_single_frame(
                    tracking.appearance_crop,
                    tracking.mouth_landmarks,
                )
                p3_ms = (time.perf_counter() - t_p3_start) * 1000.0
                if fused_vec is None:
                    continue

                self._buffer.append(BufferedFrame(
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    fused=fused_vec,
                    landmarks=tracking.landmarks,
                    mouth_landmarks=mouth_lm,
                ))
                self._frames_with_face += 1

                # Inference cadence
                now = time.perf_counter()
                if (now - last_inference_at) < self._inf_interval_s:
                    self._refresh_state_collecting(p2_ms, p3_ms)
                    continue
                if not self._buffer.is_ready(now):
                    self._refresh_state_collecting(p2_ms, p3_ms)
                    continue

                # Phase 4 + 5
                t_p4_start = time.perf_counter()
                stacked = self._buffer.stacked_fused()  # [T, 192]
                t_p4_end = time.perf_counter()
                p4_ms = (t_p4_end - t_p4_start) * 1000.0

                t_p5_start = t_p4_end
                result = self._handle.predict(stacked)
                t_p5_end = time.perf_counter()
                p5_ms = (t_p5_end - t_p5_start) * 1000.0
                total_ms = (t_p5_end - t_p4_start) * 1000.0

                self._apply_result(result, now, total_ms,
                                   p2_ms, p3_ms, p4_ms, p5_ms)
                last_inference_at = now
                self._frames_inferred += 1

            except Exception as exc:
                logger.exception("Pipeline worker error: %s", exc)
                with self._state_lock:
                    self._state.state = "ERROR"
                    self._state.error = str(exc)
                    self._state.detail = "Inference exception"
                time.sleep(0.05)

    def _phase3_single_frame(
        self,
        appearance_crop: np.ndarray,
        mouth_landmarks: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute the [192] fused vector for one frame (Phase 3)."""
        try:
            # Appearance: [3, 64, 64] normalized -> [96]
            app_tensor = self._appearance_extractor.normalize_crop(appearance_crop)
            app_tensor = app_tensor.unsqueeze(0).to("cpu")
            with torch.no_grad():
                app_emb = self._appearance_extractor.model(app_tensor).cpu().numpy()[0]

            # Geometry: mouth_landmarks [20, 2] -> scalars [20] -> [64]
            if mouth_landmarks is None or mouth_landmarks.shape != (20, 2):
                return None, None
            scalars = self._geometry_extractor.compute_scalars_batch(
                mouth_landmarks[None, ...]
            )[0]
            mean = getattr(self._geometry_extractor, "scalars_mean", None)
            std = getattr(self._geometry_extractor, "scalars_std", None)
            if mean is not None and std is not None:
                scalars = (scalars - mean) / (std + 1e-6)
            # extract_sequence takes a LIST of [20, 2] arrays
            geom_emb_full = self._geometry_extractor.extract_sequence(
                [mouth_landmarks], [scalars],
            )
            geom_emb = geom_emb_full[0]

            # Motion: needs at least 2 frames of history for savgol.
            # We feed the last scalar twice to satisfy the minimum length.
            if self._last_scalars is not None:
                lm_seq = [mouth_landmarks, mouth_landmarks]
                sc_seq = [self._last_scalars, scalars]
            else:
                lm_seq = [mouth_landmarks, mouth_landmarks]
                sc_seq = [scalars, scalars]
            self._last_scalars = scalars
            mot_emb_full = self._motion_extractor.extract_sequence(
                lm_seq, sc_seq,
            )
            mot_emb = mot_emb_full[-1]

            # Fusion: each stream is [1, 1, D] -> fused [1, 1, 192]
            with torch.no_grad():
                app_t = torch.from_numpy(app_emb).float().unsqueeze(0).unsqueeze(0)
                geo_t = torch.from_numpy(geom_emb).float().unsqueeze(0).unsqueeze(0)
                mot_t = torch.from_numpy(mot_emb).float().unsqueeze(0).unsqueeze(0)
                fused, _ = self._fusion(app_t, geo_t, mot_t)
                fused_vec = fused.squeeze(0).squeeze(0).numpy()
            return fused_vec, mouth_landmarks
        except Exception as exc:
            logger.exception("Phase 3 error: %s", exc)
            return None, None

    # ------------------------------------------------------------------
    # State refresh helpers
    # ------------------------------------------------------------------

    def _apply_result(
        self,
        result: PredictionResult,
        now: float,
        total_ms: float,
        p2_ms: float,
        p3_ms: float,
        p4_ms: float,
        p5_ms: float,
    ) -> None:
        self._latest_result = result
        # Stabilize display
        self._stabilizer.update(result.letter or "", result.confidence)
        display_letter = self._stabilizer.current or ""

        with self._state_lock:
            if result.state == "LOW_CONFIDENCE":
                ui_state = "LOW_CONFIDENCE"
            elif result.state == "ABSTAIN":
                ui_state = "ABSTAIN"
            elif display_letter:
                ui_state = "CONFIDENT"
            else:
                ui_state = "PROCESSING"
            self._state.state = ui_state
            self._state.letter = display_letter
            self._state.confidence = float(result.confidence)
            self._state.entropy = float(result.entropy)
            self._state.margin = float(result.margin)
            self._state.top_k = list(result.top_k)
            self._state.last_inference_ms = total_ms
            self._state.last_phase2_ms = p2_ms
            self._state.last_phase3_ms = p3_ms
            self._state.last_phase4_ms = p4_ms
            self._state.last_phase5_ms = p5_ms
            if self._prev_inference_at_for_fps > 0:
                dt = now - self._prev_inference_at_for_fps
                self._state.fps_inference = 1.0 / max(dt, 1e-6)
            self._prev_inference_at_for_fps = now
            self._state.stabilize_count = self._stabilizer._count
            stats = self._buffer.stats(now)
            self._state.buffer_size = stats.size
            self._state.buffer_capacity = stats.capacity
            self._state.buffer_has_min = stats.has_min
            self._state.face_recently = stats.has_face_recently
            self._state.fps_camera = self._camera.estimated_fps
            self._state.frames_inferred = self._frames_inferred
            self._state.frames_with_face = self._frames_with_face

    def _refresh_state_collecting(self, p2_ms: float, p3_ms: float) -> None:
        with self._state_lock:
            now = time.perf_counter()
            stats = self._buffer.stats(now)
            self._state.buffer_size = stats.size
            self._state.buffer_capacity = stats.capacity
            self._state.buffer_has_min = stats.has_min
            self._state.face_recently = stats.has_face_recently
            self._state.fps_camera = self._camera.estimated_fps
            self._state.last_phase2_ms = p2_ms
            self._state.last_phase3_ms = p3_ms
            if self._state.state not in ("ERROR", "ABSTAIN", "CONFIDENT",
                                          "LOW_CONFIDENCE"):
                if not stats.has_face_recently:
                    self._state.state = "NO_FACE"
                    self._state.detail = "Face lost"
                elif not stats.has_min:
                    self._state.state = "PROCESSING"
                    self._state.detail = (
                        f"Collecting frames ({stats.size}/{stats.capacity})"
                    )
                else:
                    self._state.state = "PROCESSING"
                    self._state.detail = "Inference pending"

    def _refresh_state_idle(self) -> None:
        with self._state_lock:
            now = time.perf_counter()
            stats = self._buffer.stats(now)
            self._state.buffer_size = stats.size
            self._state.buffer_capacity = stats.capacity
            self._state.buffer_has_min = stats.has_min
            self._state.face_recently = stats.has_face_recently
            self._state.fps_camera = self._camera.estimated_fps
            if self._frames_with_face == 0 and self._state.state != "ERROR":
                self._state.state = "NO_FACE"
                self._state.detail = "No face detected"


_live_pipeline: Optional[LivePipeline] = None
_live_pipeline_lock = threading.Lock()


def get_live_pipeline() -> LivePipeline:
    global _live_pipeline
    with _live_pipeline_lock:
        if _live_pipeline is None:
            _live_pipeline = LivePipeline()
        return _live_pipeline


def shutdown_live_pipeline() -> None:
    global _live_pipeline
    with _live_pipeline_lock:
        if _live_pipeline is not None:
            _live_pipeline.stop()
            _live_pipeline = None