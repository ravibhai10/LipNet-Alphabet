"""
JobManager — thread-safe background job abstraction for training and inference.

States: QUEUED | RUNNING | COMPLETED | FAILED | CANCELLED

Each job runs in a dedicated daemon thread. Workers are bounded (max 1 training
at a time). Job state is stored in a dict keyed by job_id, accessible from the
Streamlit UI without holding locks during rendering.
"""
from __future__ import annotations

import json
import logging
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class JobState(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class Job:
    job_id: str
    name: str
    state: JobState = JobState.QUEUED
    created_at: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    progress: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()


class JobManager:
    """Bounded worker pool. max_workers=1 by default (one training job at a time).

    Job state is read-only for consumers (Streamlit UI reads from self.jobs dict
    without acquiring the lock). Only the worker thread writes to job state.
    The main thread writes only job.state transitions and result/error.
    """

    MAX_WORKERS = 1

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: Dict[str, Job] = {}
        self._workers: Dict[str, threading.Thread] = {}
        self._cancelled: set = set()

    def submit(
        self,
        name: str,
        target: Callable[..., Any],
        args: tuple = (),
        kwargs: Optional[Dict] = None,
    ) -> str:
        """Submit a job. Returns job_id. Raises RuntimeError if at capacity."""
        kwargs = kwargs or {}
        with self._lock:
            running = [j for j in self._jobs.values() if j.state == JobState.RUNNING]
            if len(running) >= self.MAX_WORKERS:
                raise RuntimeError(
                    f"Worker pool at capacity ({self.MAX_WORKERS} job(s) running). "
                    "Wait for the current job to finish before submitting a new one."
                )
            job_id = str(uuid.uuid4())[:8]
            job = Job(job_id=job_id, name=name)
            self._jobs[job_id] = job
            thread = threading.Thread(
                target=self._worker, args=(job_id, target, args, kwargs),
                daemon=True, name=f"job-{job_id}",
            )
            self._workers[job_id] = thread
            self._cancelled.discard(job_id)
            thread.start()
            logger.info("Job %s (%s) submitted", job_id, name)
            return job_id

    def _worker(self, job_id: str, target: Callable, args: tuple, kwargs: Dict):
        job = self._jobs[job_id]
        if job_id in self._cancelled:
            job.state = JobState.CANCELLED
            job.finished_at = datetime.utcnow().isoformat()
            return
        job.state = JobState.RUNNING
        job.started_at = datetime.utcnow().isoformat()
        try:
            result = target(*args, **kwargs)
            if job_id in self._cancelled:
                job.state = JobState.CANCELLED
            else:
                job.state = JobState.COMPLETED
                job.result = result
        except Exception as e:
            if job_id in self._cancelled:
                job.state = JobState.CANCELLED
            else:
                job.state = JobState.FAILED
                job.error = str(e)
                job.traceback = traceback.format_exc()
                logger.exception("Job %s failed", job_id)
        finally:
            job.finished_at = datetime.utcnow().isoformat()

    def cancel(self, job_id: str) -> bool:
        """Request cancellation. Returns True if the job was found and cancelled."""
        with self._lock:
            if job_id not in self._jobs:
                return False
            job = self._jobs[job_id]
            if job.state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED):
                return False
            self._cancelled.add(job_id)
            if job.state == JobState.RUNNING:
                job.state = JobState.CANCELLED
                job.finished_at = datetime.utcnow().isoformat()
            return True

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at)

    def latest_job(self) -> Optional[Job]:
        jobs = self.list_jobs()
        return jobs[-1] if jobs else None

    def is_running(self) -> bool:
        with self._lock:
            return any(j.state == JobState.RUNNING for j in self._jobs.values())

    def clear_finished(self):
        """Remove completed/failed/cancelled jobs from the registry."""
        with self._lock:
            self._jobs = {
                k: v for k, v in self._jobs.items()
                if v.state not in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED)
            }


# ── singleton ─────────────────────────────────────────────────────────────────
_job_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager
