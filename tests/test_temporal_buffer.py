"""Tests for TemporalBuffer and PredictionStabilizer.

Covers: chronological ordering, max length, insufficient frames,
reset, duplicate rejection, prediction stabilization.
"""
from __future__ import annotations

import time
import pytest
import numpy as np

from app.services.temporal_buffer import (
    TemporalBuffer, BufferedFrame, BufferStats, PredictionStabilizer,
)


class TestTemporalBufferBasics:
    """Structural correctness of the buffer."""

    def test_empty_buffer_has_zero_length(self):
        buf = TemporalBuffer(max_frames=10, min_frames=5)
        assert len(buf) == 0
        assert buf.max_frames == 10
        assert buf.min_frames == 5

    def test_rejects_max_less_than_min(self):
        with pytest.raises(ValueError, match="max_frames"):
            TemporalBuffer(max_frames=3, min_frames=5)

    def test_is_ready_false_when_empty(self):
        buf = TemporalBuffer(max_frames=10, min_frames=5)
        assert buf.is_ready(time.perf_counter()) is False


class TestTemporalBufferAppend:
    """Frame append behavior."""

    def _make(self, frame_id=0, fused_dim=192):
        return BufferedFrame(
            frame_id=frame_id,
            timestamp=time.perf_counter(),
            fused=np.zeros(fused_dim, dtype=np.float32),
            landmarks=None,
            mouth_landmarks=None,
        )

    def test_single_append_increments_len(self):
        buf = TemporalBuffer(max_frames=10, min_frames=1)
        buf.append(self._make(1))
        assert len(buf) == 1

    def test_duplicate_frame_id_rejected(self):
        buf = TemporalBuffer(max_frames=10, min_frames=1)
        buf.append(self._make(42))
        buf.append(self._make(42))  # same id
        assert len(buf) == 1

    def test_append_none_returns_false(self):
        buf = TemporalBuffer(max_frames=10, min_frames=1)
        assert buf.append(None) is False
        assert len(buf) == 0

    def test_old_frames_evicted_when_full(self):
        buf = TemporalBuffer(max_frames=5, min_frames=1)
        for i in range(7):
            buf.append(self._make(i))
        assert len(buf) == 5
        ids = [f.frame_id for f in buf.frames_chronological()]
        # Oldest 2 should be gone
        assert ids == [2, 3, 4, 5, 6]

    def test_chronological_order_preserved(self):
        buf = TemporalBuffer(max_frames=20, min_frames=1)
        for i in range(10):
            buf.append(self._make(i))
        ids = [f.frame_id for f in buf.frames_chronological()]
        assert ids == list(range(10))  # monotonically increasing

    def test_clear_resets_buffer(self):
        buf = TemporalBuffer(max_frames=10, min_frames=1)
        for i in range(5):
            buf.append(self._make(i))
        assert len(buf) == 5
        buf.clear()
        assert len(buf) == 0
        # Face timestamp also reset
        assert buf.is_ready(time.perf_counter() + 10) is False


class TestTemporalBufferFaceTracking:
    """Face-timestamp bookkeeping."""

    def _make_with_landmarks(self, frame_id, ts):
        return BufferedFrame(
            frame_id=frame_id,
            timestamp=ts,
            fused=np.zeros(192, dtype=np.float32),
            landmarks=np.zeros((478, 3), dtype=np.float32),
            mouth_landmarks=np.zeros((20, 2), dtype=np.float32),
        )

    def _make_no_face(self, frame_id, ts):
        return BufferedFrame(
            frame_id=frame_id,
            timestamp=ts,
            fused=np.zeros(192, dtype=np.float32),
            landmarks=None,
            mouth_landmarks=None,
        )

    def test_has_face_recently_true_after_face_frame(self):
        buf = TemporalBuffer(max_frames=10, min_frames=1, no_face_timeout_s=1.0)
        now = time.perf_counter()
        buf.append(self._make_with_landmarks(1, now))
        assert buf.has_face_recently(now) is True

    def test_has_face_recently_false_after_timeout(self):
        buf = TemporalBuffer(max_frames=10, min_frames=1, no_face_timeout_s=0.5)
        past = time.perf_counter() - 2.0
        buf.append(self._make_with_landmarks(1, past))
        assert buf.has_face_recently(time.perf_counter()) is False

    def test_no_face_frame_does_not_update_timestamp(self):
        buf = TemporalBuffer(max_frames=10, min_frames=1, no_face_timeout_s=1.0)
        now = time.perf_counter()
        buf.append(self._make_no_face(1, now))
        assert buf.has_face_recently(now) is False

    def test_is_ready_false_without_recent_face(self):
        buf = TemporalBuffer(max_frames=10, min_frames=5, no_face_timeout_s=0.5)
        now = time.perf_counter()
        for i in range(5):
            buf.append(self._make_no_face(i, now - 2.0))
        assert buf.is_ready(time.perf_counter()) is False

    def test_is_ready_true_with_enough_frames_and_recent_face(self):
        buf = TemporalBuffer(max_frames=10, min_frames=3, no_face_timeout_s=10.0)
        now = time.perf_counter()
        for i in range(5):
            buf.append(self._make_with_landmarks(i, now))
        assert buf.is_ready(now) is True


class TestTemporalBufferStackedFused:
    """stacked_fused output shape."""

    def test_stacked_fused_returns_correct_shape(self):
        buf = TemporalBuffer(max_frames=20, min_frames=1)
        now = time.perf_counter()
        for i in range(6):
            frame = BufferedFrame(
                frame_id=i, timestamp=now,
                fused=np.full(192, float(i), dtype=np.float32),
                landmarks=None, mouth_landmarks=None,
            )
            buf.append(frame)
        stacked = buf.stacked_fused()
        assert stacked.shape == (6, 192)
        assert np.allclose(stacked[0], 0.0)
        assert np.allclose(stacked[3], 3.0)

    def test_stacked_fused_empty_returns_zero_array(self):
        buf = TemporalBuffer(max_frames=10, min_frames=1)
        stacked = buf.stacked_fused()
        assert stacked.shape == (0, 192)


class TestBufferStats:
    """BufferStats snapshot."""

    def test_stats_reflect_current_state(self):
        buf = TemporalBuffer(max_frames=10, min_frames=3, no_face_timeout_s=1.0)
        now = time.perf_counter()
        for i in range(4):
            buf.append(BufferedFrame(
                frame_id=i, timestamp=now,
                fused=np.zeros(192, dtype=np.float32),
                landmarks=np.zeros((478, 3), dtype=np.float32),
            ))
        stats = buf.stats(now)
        assert stats.size == 4
        assert stats.capacity == 10
        assert stats.has_min is True
        assert stats.has_face_recently is True


class TestPredictionStabilizer:
    """PredictionStabilizer reduces label flicker."""

    def test_first_label_is_returned(self):
        stab = PredictionStabilizer(required_count=1)
        result = stab.update("A", 0.9)
        assert result == "A"
        assert stab.current == "A"

    def test_same_label_accumulates(self):
        stab = PredictionStabilizer(required_count=3)
        assert stab.update("A", 0.9) == "A"  # count=1
        assert stab.update("A", 0.85) is None   # count=2
        assert stab.update("A", 0.80) is None   # count=3 -> stable
        assert stab.current == "A"

    def test_different_label_blocked_until_required_count(self):
        stab = PredictionStabilizer(required_count=3)
        stab.update("A", 0.9)
        stab.update("A", 0.9)
        stab.update("B", 0.5)   # count resets to 1 for B
        assert stab.current == "A"
        assert stab.update("B", 0.5) is None   # count=2
        assert stab.current == "A"

    def test_confidence_margin_allows_early_switch(self):
        stab = PredictionStabilizer(required_count=2, margin=0.05)
        # First lock in "A" for 2 frames (required_count)
        stab.update("A", 0.9)
        stab.update("A", 0.9)
        # Now "B" with a much higher confidence jump (> margin)
        result = stab.update("B", 0.99)
        # Should switch to B because conf jump > margin AND count>=required
        assert stab.current == "B"
        assert result == "A"  # returns prev

    def test_reset_clears_state(self):
        stab = PredictionStabilizer(required_count=2)
        stab.update("A", 0.9)
        stab.update("A", 0.9)
        assert stab.current == "A"
        stab.reset()
        assert stab.current is None
        assert stab.current_confidence == 0.0

    def test_rejects_invalid_required_count(self):
        with pytest.raises(ValueError, match="required_count"):
            PredictionStabilizer(required_count=0)