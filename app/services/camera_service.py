"""CameraService - webcam capture with clean lifecycle management.

Thread-safe camera capture. Camera is owned by a dedicated daemon thread,
frames are put into a bounded queue, and the caller reads from the queue.
Never call VideoCapture from the Streamlit render thread.
"""
from __future__ import annotations

import logging
import threading
import time
import queue
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraState(str, Enum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class CameraFrame:
    """One captured frame with its capture timestamp."""
    bgr: np.ndarray          # [H, W, 3] BGR from OpenCV
    timestamp: float         # monotonic seconds from time.perf_counter()
    frame_id: int            # monotonically increasing


class CameraService:
    """Thread-safe webcam capture.

    Owned by a single background thread. The caller reads frames from
    the public `frames` queue. This design avoids holding any lock
    during Streamlit rendering.

    Lifecycle:
        start(camera_index)  -> spawns capture thread
        read()               -> returns CameraFrame or None
        stop()               -> stops capture thread, releases camera
        __del__              -> defensive stop if caller forgets
    """

    QUEUE_SIZE = 4          # keep only a few frames; old ones are dropped
    FPS_SAMPLE_SIZE = 30   # rolling window for FPS estimation

    def __init__(self):
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._state = CameraState.STOPPED
        self._state_lock = threading.Lock()
        self._fps_history: deque[float] = deque(maxlen=self.FPS_SAMPLE_SIZE)
        self._last_ts = 0.0
        self._frame_id = 0
        self._error_msg: Optional[str] = None

        # Public queue: caller reads here. Never holds a lock.
        self.frames: queue.Queue[CameraFrame] = queue.Queue(
            maxsize=self.QUEUE_SIZE
        )

    @property
    def state(self) -> CameraState:
        with self._state_lock:
            return self._state

    @property
    def error_msg(self) -> Optional[str]:
        with self._state_lock:
            return self._error_msg

    @property
    def estimated_fps(self) -> float:
        if len(self._fps_history) < 2:
            return 0.0
        return 1.0 / (sum(self._fps_history) / len(self._fps_history))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        """Start the capture thread.

        Args:
            camera_index: OpenCV camera index.
            width, height: Frame resolution.
            fps: Requested frames-per-second.

        Raises:
            RuntimeError: if already running.
            CameraUnavailableError: if camera cannot be opened.
        """
        with self._state_lock:
            if self._state in (CameraState.STARTING, CameraState.RUNNING):
                raise RuntimeError(f"Camera already {self._state.value}")
            self._state = CameraState.STARTING
            self._error_msg = None

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            cap.release()
            with self._state_lock:
                self._state = CameraState.ERROR
                self._error_msg = (
                    f"Camera index {camera_index} unavailable. "
                    "Ensure no other application is using the camera."
                )
            raise CameraUnavailableError(self._error_msg)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        cap.set(cv2.CAP_PROP_FPS, float(fps))

        # Verify settings took
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        logger.info(
            "Camera opened: index=%d  resolution=%dx%d  fps=%.1f",
            camera_index, actual_w, actual_h, actual_fps,
        )

        self._cap = cap
        self._running.set()
        self._thread = threading.Thread(
            target=self._capture_loop,
            args=(camera_index,),
            daemon=True,
            name="camera-capture",
        )
        self._thread.start()

        # Wait briefly for first frame
        for _ in range(30):
            time.sleep(0.05)
            if not self._running.is_set():
                break
            if not self.frames.empty():
                break

        with self._state_lock:
            if self._state != CameraState.ERROR:
                self._state = CameraState.RUNNING

    def read(self) -> Optional[CameraFrame]:
        """Return the oldest frame in the queue, or None if empty."""
        try:
            return self.frames.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        """Stop capture thread and release camera. Idempotent."""
        if not self._running.is_set() and self._state == CameraState.STOPPED:
            return

        self._running.clear()

        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as exc:
                logger.warning("Error releasing camera: %s", exc)
            self._cap = None

        # Drain queue
        while True:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                break

        with self._state_lock:
            self._state = CameraState.STOPPED

        logger.info("Camera stopped")

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Capture loop (runs in dedicated thread)
    # ------------------------------------------------------------------

    def _capture_loop(self, camera_index: int) -> None:
        cap = self._cap
        if cap is None:
            return

        ts_prev = 0.0
        fid = 0

        while self._running.is_set():
            ret, bgr = cap.read()
            if not ret:
                # Camera disconnected or error
                logger.warning("Camera read returned False")
                self._running.clear()
                with self._state_lock:
                    self._state = CameraState.ERROR
                    self._error_msg = "Camera disconnected or read error."
                break

            ts = time.perf_counter()
            if ts_prev > 0:
                dt = ts - ts_prev
                if 0 < dt < 5.0:          # sanity guard against wild values
                    self._fps_history.append(dt)
            ts_prev = ts
            fid += 1

            frame = CameraFrame(bgr=bgr, timestamp=ts, frame_id=fid)

            # Non-blocking put; drop oldest if full
            try:
                self.frames.put_nowait(frame)
            except queue.Full:
                try:
                    self.frames.get_nowait()   # discard oldest
                except queue.Empty:
                    pass
                try:
                    self.frames.put_nowait(frame)
                except queue.Full:
                    pass

        # Drain remaining frames before exit with safety counter
        for _ in range(10):
            ret, _ = cap.read()
            if not ret:
                break


class CameraUnavailableError(RuntimeError):
    """Raised when the webcam cannot be opened."""


# Singleton camera (owned by the live pipeline)
_camera_service: Optional[CameraService] = None
_camera_lock = threading.Lock()


def get_camera_service() -> CameraService:
    global _camera_service
    with _camera_lock:
        if _camera_service is None:
            _camera_service = CameraService()
        return _camera_service


def shutdown_camera_service() -> None:
    """Stop and discard the global camera service."""
    global _camera_service
    with _camera_lock:
        if _camera_service is not None:
            _camera_service.stop()
            _camera_service = None