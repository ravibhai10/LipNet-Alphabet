"""TemporalBuffer - bounded, chronologically-ordered frame buffer.

Maintains a sliding window of representation frames for live inference.
Frames are kept in arrival order; old frames are dropped when the buffer
is full. No two frames with the same id may coexist.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class BufferedFrame:
    """One temporal frame in the buffer."""
    frame_id: int
    timestamp: float
    fused: np.ndarray           # [192] Phase 3 fused vector (after Phase 2+3)
    landmarks: Optional[np.ndarray] = None  # [478, 3] MediaPipe for debug
    mouth_landmarks: Optional[np.ndarray] = None  # [20, 2] normalized


@dataclass
class BufferStats:
    """Snapshot of the temporal buffer."""
    size: int
    capacity: int
    has_min: bool               # at least min_frames queued
    has_face_recently: bool     # face seen within `no_face_timeout_s`
    age_s: float                # age of oldest frame, 0 if empty


class TemporalBuffer:
    """Bounded temporal frame buffer.

    Frames are kept in arrival order. When the buffer is full, the
    oldest frame is dropped. Thread-safety is provided for the
    single-producer (capture) / single-consumer (inference) pattern
    typical of live pipelines.
    """

    def __init__(
        self,
        max_frames: int = 64,
        min_frames: int = 8,
        no_face_timeout_s: float = 1.0,
    ):
        if max_frames < min_frames:
            raise ValueError(
                f"max_frames ({max_frames}) must be >= min_frames ({min_frames})"
            )
        self._buf: deque[BufferedFrame] = deque(maxlen=max_frames)
        self._min = min_frames
        self._no_face_timeout = no_face_timeout_s
        self._last_face_ts: float = 0.0
        self._lock_serial = 0     # monotonic counter, no lock needed
        self._ids: set = set()    # duplicate-id guard
        self._last_append_ts: float = 0.0

    @property
    def max_frames(self) -> int:
        return self._buf.maxlen or 0

    @property
    def min_frames(self) -> int:
        return self._min

    def __len__(self) -> int:
        return len(self._buf)

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def append(self, frame: BufferedFrame) -> bool:
        """Append one frame. Returns True if accepted.

        - Duplicate frame_id is silently rejected.
        - Old frames are auto-evicted when max_frames is reached.
        - The face timestamp is updated iff a non-None landmark set
          is present.
        """
        if frame is None:
            return False
        if frame.frame_id in self._ids:
            return False

        # deque(maxlen=...) already drops oldest; clean up its id
        if self._buf.maxlen is not None and len(self._buf) >= self._buf.maxlen:
            evicted = self._buf[0]    # about to be dropped
            self._ids.discard(evicted.frame_id)

        self._buf.append(frame)
        self._ids.add(frame.frame_id)
        self._last_append_ts = frame.timestamp

        if frame.landmarks is not None and len(frame.landmarks) > 0:
            self._last_face_ts = frame.timestamp

        return True

    def clear(self) -> None:
        """Drop all frames and reset state."""
        self._buf.clear()
        self._ids.clear()
        self._last_face_ts = 0.0
        self._last_append_ts = 0.0

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def is_ready(self, now_ts: float) -> bool:
        """Buffer has enough frames to attempt inference."""
        if len(self._buf) < self._min:
            return False
        if not self.has_face_recently(now_ts):
            return False
        return True

    def has_face_recently(self, now_ts: float) -> bool:
        if self._last_face_ts == 0.0:
            return False
        return (now_ts - self._last_face_ts) <= self._no_face_timeout

    def stats(self, now_ts: float) -> BufferStats:
        return BufferStats(
            size=len(self._buf),
            capacity=self._buf.maxlen or 0,
            has_min=len(self._buf) >= self._min,
            has_face_recently=self.has_face_recently(now_ts),
            age_s=(now_ts - self._last_append_ts) if self._last_append_ts > 0 else 0.0,
        )

    def stacked_fused(self) -> np.ndarray:
        """Return the current temporal sequence as [T, 192] float32."""
        if not self._buf:
            return np.zeros((0, 192), dtype=np.float32)
        return np.stack([f.fused for f in self._buf], axis=0).astype(np.float32)

    def frames_chronological(self) -> List[BufferedFrame]:
        """Return a snapshot of the buffer in chronological order."""
        return list(self._buf)


class PredictionStabilizer:
    """Reduce label flicker on the displayed prediction.

    The displayed label only changes once the same label has been
    observed for ``required_count`` consecutive stable inferences OR
    the confidence drop is above a margin.  This isolates the
    UI from raw per-frame jitter.
    """

    def __init__(self, required_count: int = 3, margin: float = 0.05):
        if required_count < 1:
            raise ValueError("required_count must be >= 1")
        self._required = required_count
        self._margin = float(margin)
        self._current: Optional[str] = None
        self._current_conf: float = 0.0
        self._count = 0

    def update(self, label: str, confidence: float) -> Optional[str]:
        """Feed one inference result.

        Returns the new stable label when it changes, else None.
        """
        if self._current is None:
            self._current = label
            self._current_conf = confidence
            self._count = 1
            return label

        if label == self._current:
            self._count += 1
            self._current_conf = max(self._current_conf, confidence)
            return None

        # Different label: accept only if seen >= required_count and conf jump
        if self._count >= self._required and confidence > self._current_conf + self._margin:
            prev = self._current
            self._current = label
            self._current_conf = confidence
            self._count = 1
            return prev
        return None

    def reset(self) -> None:
        self._current = None
        self._current_conf = 0.0
        self._count = 0

    @property
    def current(self) -> Optional[str]:
        return self._current

    @property
    def current_confidence(self) -> float:
        return self._current_conf