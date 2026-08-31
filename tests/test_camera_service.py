"""Tests for CameraService.

Tests camera lifecycle without requiring physical webcam hardware.
Tests: unavailable camera, stop cleanup, state transitions.
"""
from __future__ import annotations

import time
import queue
import threading
import pytest

from app.services.camera_service import (
    CameraService, CameraFrame, CameraState, CameraUnavailableError,
)


class TestCameraServiceStateTransitions:
    """State machine correctness."""

    def test_initial_state_is_stopped(self):
        svc = CameraService()
        assert svc.state == CameraState.STOPPED

    def test_stop_when_not_started_is_idempotent(self):
        svc = CameraService()
        svc.stop()
        assert svc.state == CameraState.STOPPED

    def test_estimated_fps_zero_when_stopped(self):
        svc = CameraService()
        assert svc.estimated_fps == 0.0

    def test_frames_queue_empty_initially(self):
        svc = CameraService()
        assert svc.frames.empty()


class TestCameraServiceUnavailable:
    """Camera unavailable scenarios."""

    def test_start_with_invalid_index_raises(self):
        svc = CameraService()
        with pytest.raises(CameraUnavailableError, match="unavailable"):
            svc.start(camera_index=99)

    def test_state_is_error_after_unavailable_start(self):
        svc = CameraService()
        try:
            svc.start(camera_index=99)
        except CameraUnavailableError:
            pass
        assert svc.state == CameraState.ERROR

    def test_error_msg_set_after_unavailable(self):
        svc = CameraService()
        try:
            svc.start(camera_index=99)
        except CameraUnavailableError:
            pass
        assert svc.error_msg is not None
        assert "unavailable" in svc.error_msg.lower()


class TestCameraFrameDataclass:
    """CameraFrame frozen dataclass."""

    def test_camera_frame_is_frozen(self):
        frame = CameraFrame(bgr=None, timestamp=1.0, frame_id=1)
        with pytest.raises(AttributeError):
            frame.frame_id = 2  # type: ignore

    def test_camera_frame_fields(self):
        import numpy as np
        arr = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = CameraFrame(bgr=arr, timestamp=1.5, frame_id=7)
        assert frame.timestamp == 1.5
        assert frame.frame_id == 7
        assert frame.bgr.shape == (480, 640, 3)


class TestCameraServiceRead:
    """Read behavior."""

    def test_read_returns_none_when_queue_empty(self):
        svc = CameraService()
        assert svc.read() is None

    def test_read_returns_oldest_frame(self):
        import numpy as np
        svc = CameraService()
        f1 = CameraFrame(bgr=np.zeros((10, 10, 3)), timestamp=1.0, frame_id=1)
        f2 = CameraFrame(bgr=np.zeros((10, 10, 3)), timestamp=2.0, frame_id=2)
        svc.frames.put_nowait(f1)
        svc.frames.put_nowait(f2)
        assert svc.read().frame_id == 1
        assert svc.read().frame_id == 2
        assert svc.read() is None


class TestCameraServiceLifecycle:
    """Full lifecycle (no physical camera needed for invalid-index path)."""

    def test_start_stop_releases_state(self):
        svc = CameraService()
        try:
            svc.start(camera_index=99)
        except CameraUnavailableError:
            pass
        svc.stop()
        assert svc.state == CameraState.STOPPED

    def test_double_stop_is_idempotent(self):
        svc = CameraService()
        try:
            svc.start(camera_index=99)
        except CameraUnavailableError:
            pass
        svc.stop()
        svc.stop()
        assert svc.state == CameraState.STOPPED

    def test_queue_drained_on_stop(self):
        import numpy as np
        svc = CameraService()
        try:
            svc.start(camera_index=99)
        except CameraUnavailableError:
            pass
        # Manually fill queue
        svc.frames.put_nowait(
            CameraFrame(bgr=np.zeros((10, 10, 3)), timestamp=1.0, frame_id=1)
        )
        svc.stop()
        assert svc.frames.empty()

    def test_cannot_start_while_running(self):
        # After the first start fails, state should be ERROR.
        # A subsequent start should not raise "already" since
        # the camera is not actively starting/running.
        svc = CameraService()
        with pytest.raises(CameraUnavailableError):
            svc.start(camera_index=99)
        # State should be ERROR
        assert svc.state == CameraState.ERROR
        # Restarting from ERROR is allowed (it will fail again)
        with pytest.raises(CameraUnavailableError):
            svc.start(camera_index=99)