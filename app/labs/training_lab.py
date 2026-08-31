"""
TrainingLab - background training of frozen Phase 2-5 architecture.

Displays:
  - Pre-training gate results
  - Dataset summary
  - Frozen architecture summary
  - Training controls (epochs, batch size, lr, resume)
  - Live job status (via JobManager)
"""
from __future__ import annotations

import time
import streamlit as st
from pathlib import Path
from typing import Optional

from app.services.training_service import get_training_service
from app.services.dataset_service import get_dataset_service
from app.services.job_manager import get_job_manager, JobState


def render():
    st.header("Training Lab")
    st.caption("Trains the frozen Phase 2-5 pipeline using real webcam data")
    ts = get_training_service()
    ds = get_dataset_service()

    if "train_last_job_id" not in st.session_state:
        st.session_state.train_last_job_id = None

    # ── pre-training check ──────────────────────────────────────────────
    check = ts.pre_training_check()
    _render_gate(check)

    # ── architecture summary ────────────────────────────────────────────
    with st.expander("Frozen Architecture (read-only)"):
        st.code(
            "Real video\n"
            "  -> MediaPipe FaceLandmarker\n"
            "  -> Mouth alignment\n"
            "  -> Appearance T x 96\n"
            "  -> Geometry T x 64\n"
            "  -> Motion T x 64\n"
            "  -> Gated Fusion T x 192\n"
            "  -> 4 x Conformer (d=192, h=4, ffn=768, k=7)\n"
            "  -> CTC (27 classes: 0=blank, 1-26=A-Z)",
            language="text"
        )

    # ── controls ────────────────────────────────────────────────────────
    if not check.passed:
        st.warning("Training is blocked. Resolve the gate errors above before proceeding.")
        return

    st.subheader("Training Configuration")
    config = ts.get_config()
    c1, c2, c3 = st.columns(3)
    epochs = c1.number_input("Epochs", min_value=1, max_value=500, value=int(config.training.epochs))
    batch_size = c2.number_input("Batch size", min_value=1, max_value=64, value=int(config.training.batch_size))
    lr = c3.number_input("Learning rate", min_value=1e-6, max_value=1e-1, value=float(config.training.lr), format="%.4f")
    resume = st.checkbox("Resume from latest checkpoint", value=False)

    jm = get_job_manager()
    is_running = jm.is_running()

    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        start_disabled = is_running
        if st.button("START TRAINING", type="primary", use_container_width=True, disabled=start_disabled):
            cfg = ts.get_config()
            cfg.training.epochs = int(epochs)
            cfg.training.batch_size = int(batch_size)
            cfg.training.lr = float(lr)
            try:
                job_id = jm.submit(
                    name="training",
                    target=ts.train,
                    kwargs={"config": cfg, "max_epochs": int(epochs),
                            "batch_size": int(batch_size), "lr": float(lr)},
                )
                st.session_state.train_last_job_id = job_id
                st.success(f"Job submitted: {job_id}")
            except RuntimeError as e:
                st.error(str(e))
    with c_btn2:
        cancel_disabled = not is_running
        if st.button("CANCEL", use_container_width=True, disabled=cancel_disabled):
            job = jm.latest_job()
            if job and jm.cancel(job.job_id):
                st.warning(f"Job {job.job_id} cancelled.")

    # ── status ──────────────────────────────────────────────────────────
    job = None
    if st.session_state.train_last_job_id:
        job = jm.get(st.session_state.train_last_job_id)
    if job is None:
        job = jm.latest_job()

    if job is not None:
        st.subheader("Job Status")
def _render_gate(check):
    if check.passed:
        st.success("GATE PASSED - training is allowed to start")
    else:
        st.error("TRAINING BLOCKED")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recordings", check.dataset_recordings)
    c2.metric("Speakers", check.dataset_speakers)
    splits = check.dataset_splits
    c3.metric("Train/Val/Test", f"{splits.get('train', 0)}/{splits.get('val', 0)}/{splits.get('test', 0)}")
    c4.metric("Data Source", "real_camera" if check.data_source_ok else "FAIL")
    if check.errors:
        with st.expander("Errors"):
            for e in check.errors:
                st.code(e)
    if check.warnings:
        with st.expander("Warnings"):
            for w in check.warnings:
                st.code(w)


def _render_job(job):
    state = job.state
    if state == JobState.RUNNING:
        st.info(f"RUNNING (job {job.job_id})")
    elif state == JobState.QUEUED:
        st.info(f"QUEUED (job {job.job_id})")
    elif state == JobState.COMPLETED:
        st.success(f"COMPLETED (job {job.job_id})")
    elif state == JobState.FAILED:
        st.error(f"FAILED (job {job.job_id})")
    elif state == JobState.CANCELLED:
        st.warning(f"CANCELLED (job {job.job_id})")
    c1, c2 = st.columns(2)
    c1.caption(f"Created: {job.created_at}")
    if job.started_at:
        c2.caption(f"Started: {job.started_at}")
    if job.progress:
        prog = job.progress
        st.code(f"State: {prog.get('state', '?')}\n"
                f"Epoch: {prog.get('epoch', 0)}\n"
                f"Note: {prog.get('note', '')}")
    if job.error:
        st.error(f"Error: {job.error}")
        if job.traceback:
            with st.expander("Traceback"):
                st.code(job.traceback)
    if job.result is not None:
        st.success("Result available")
        with st.expander("Summary"):
            st.json({"epochs_run": job.result.get("epochs_run"),
                    "device": job.result.get("device")})

        _render_job(job)
