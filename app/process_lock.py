"""Cross-process guard for the single-process backend deployment profile."""
from __future__ import annotations

import os
from pathlib import Path
import threading

from app.settings import settings

_lock_handle = None
_guard = threading.RLock()


def acquire_process_lock():
    """Acquire the private-store lock once per Python process."""
    global _lock_handle
    with _guard:
        if _lock_handle is not None:
            return _lock_handle

        lock_path = settings().private_root / 'backend-process.lock'
        if not lock_path.parent.is_dir():
            raise RuntimeError('Private storage directory is missing; configure it before launch.')
        handle = lock_path.open('a+b')
        handle.seek(0)
        if lock_path.stat().st_size == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            handle.close()
            raise RuntimeError('Another LipNet backend process holds the private deployment lock.') from error
        _lock_handle = handle
        return handle


def release_process_lock():
    global _lock_handle
    with _guard:
        if _lock_handle is not None:
            _lock_handle.close()
            _lock_handle = None
