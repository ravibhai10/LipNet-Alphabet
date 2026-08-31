"""LiveTestLab - real webcam inference using trained checkpoint.

State machine: IDLE | NO_MODEL | NO_FACE | PROCESSING | CONFIDENT
              | LOW_CONFIDENCE | ABSTAIN | ERROR

Model loads ONCE per checkpoint, then a background worker thread
consumes CameraService frames, runs Phase 2-5, and a UI render
loop reads the latest result.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from app.services.inference_service import get_inference_service
from app.services.live_pipeline import get_live_pipeline, shutdown_live_pipeline


def _ensure_session_state() -> None:
    if "live_test_ckpt" not in st.session_state:
        st.session_state.live_test_ckpt = None
    if "live_test_running" not in st.session_state:
        st.session_state.live_test_running = False
    if "live_test_last_state" not in st.session_state:
        st.session_state.live_test_last_state = None
    if "live_test_saved_paths" not in st.session_state:
        st.session_state.live_test_saved_paths = []


def render() -> None:
    st.header("Live Test Lab")
    st.caption("Real webcam -> MediaPipe -> Phase 3 -> Conformer -> CTC -> A-Z")
    _ensure_session_state()

    isvc = get_inference_service()
    pipeline = get_live_pipeline()

    # Model selection ----------------------------------------------------
    st.subheader("1. Model")
    checkpoints = isvc.list_checkpoints()
    if not checkpoints:
        st.warning(
            "**NO TRAINED MODEL AVAILABLE** — Train a model in Training Lab first. "
            "(Looking under `runs/letter_decoder/`)"
        )
        return

    ckpt_options = [str(d) for d in checkpoints]
    options_with_meta = []
    for p in ckpt_options:
        summary = isvc.get_checkpoint_summary(p)
        tag = p
        if summary:
            acc = summary.get("best_val_accuracy") or summary.get("val_accuracy")
            if acc is not None:
                tag = f"{Path(p).name}  (val acc {float(acc):.3f})"
            else:
                tag = Path(p).name
        else:
            tag = Path(p).name
        options_with_meta.append(tag)

    selected_idx = st.selectbox(
        "Checkpoint",
        range(len(ckpt_options)),
        format_func=lambda i: options_with_meta[i],
    )
    selected_ckpt = ckpt_options[selected_idx]

    # Load model
    if st.session_state.live_test_ckpt != selected_ckpt:
        with st.spinner("Loading checkpoint..."):
            try:
                pipeline.load_model(Path(selected_ckpt))
                st.session_state.live_test_ckpt = selected_ckpt
                st.success(f"Model loaded: {Path(selected_ckpt).name}")
            except FileNotFoundError:
                st.error("Checkpoint file missing.")
                return
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                return

    # Controls -----------------------------------------------------------
    st.subheader("2. Camera Controls")
    col_cam, col_btn = st.columns([1, 2])
    with col_cam:
        camera_index = st.number_input("Camera index", min_value=0, max_value=5, value=0)
    with col_cam:
        width = st.selectbox("Width", [320, 640, 1280], index=1)
        height = st.selectbox("Height", [240, 480, 720], index=1)
    with col_cam:
        target_fps = st.slider("Target FPS", 10, 30, 25)

    cols = st.columns(4)
    start_clicked = cols[0].button(
        "START",
        type="primary",
        disabled=st.session_state.live_test_running,
    )
    stop_clicked = cols[1].button(
        "STOP",
        disabled=not st.session_state.live_test_running,
    )
    save_clicked = cols[2].button("Save test recording (NOT training data)")
    reset_clicked = cols[3].button("Reset buffer")

    if start_clicked:
        try:
            pipeline.start()
            st.session_state.live_test_running = True
        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Failed to start: {e}")
    if stop_clicked:
        pipeline.stop()
        st.session_state.live_test_running = False
    if reset_clicked:
        pipeline._buffer.clear()
        st.info("Buffer reset.")
    if save_clicked:
        out = Path("test_recordings") / f"live_{int(time.time())}"
        try:
            n = pipeline.save_test_recording(out)
            st.session_state.live_test_saved_paths.append(str(out))
            st.success(f"Saved {n} frames to {out} (NOT in alphabet_dataset).")
        except Exception as e:
            st.error(f"Save failed: {e}")

    # Live status --------------------------------------------------------
    st.subheader("3. Live Status")
    state = pipeline.get_state()

    col_state, col_letter, col_conf = st.columns(3)
    with col_state:
        st.metric("State", state.state)
    with col_letter:
        st.metric("Predicted Letter",
                  state.letter if state.letter else "—")
    with col_conf:
        st.metric("Confidence",
                  f"{state.confidence * 100:.1f}%" if state.confidence else "—")

    col_e, col_m, col_stab = st.columns(3)
    col_e.metric("Entropy", f"{state.entropy:.3f}" if state.entropy == state.entropy else "—")
    col_m.metric("Margin",
                 f"{state.margin:.3f}" if state.margin == state.margin else "—")
    col_stab.metric("Stabilize Count", state.stabilize_count)

    st.markdown("##### Temporal buffer")
    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    col_b1.metric("Frames", f"{state.buffer_size}/{state.buffer_capacity}")
    col_b2.metric("Min ready", "✓" if state.buffer_has_min else "✗")
    col_b3.metric("Face recent", "✓" if state.face_recently else "✗")
    col_b4.metric("Camera FPS", f"{state.fps_camera:.1f}")

    st.markdown("##### Top-5")
    if state.top_k:
        st.table([
            {"Letter": l, "Probability": f"{p * 100:.1f}%"}
            for l, p in state.top_k
        ])
    else:
        st.caption("No prediction yet.")

    st.markdown("##### Per-phase timing (ms)")
    col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns(5)
    col_t1.metric("Phase 2 (MediaPipe)", f"{state.last_phase2_ms:.1f}")
    col_t2.metric("Phase 3 (Repr.)", f"{state.last_phase3_ms:.1f}")
    col_t3.metric("Phase 4 (Conf.)", f"{state.last_phase4_ms:.1f}")
    col_t4.metric("Phase 5 (CTC)", f"{state.last_phase5_ms:.1f}")
    col_t5.metric("Total", f"{state.last_inference_ms:.1f}")

    st.caption(
        f"Inferences: {state.frames_inferred} | "
        f"Frames with face: {state.frames_with_face} | "
        f"Inference FPS: {state.fps_inference:.2f}"
    )

    if state.error:
        st.error(f"Error: {state.error}")
    if state.detail:
        st.caption(state.detail)

    if st.session_state.live_test_saved_paths:
        with st.expander("Saved test recordings (NOT in training data)"):
            for p in st.session_state.live_test_saved_paths:
                st.code(p, language="text")

    # Auto-refresh while running -----------------------------------------
    if st.session_state.live_test_running and pipeline.is_running():
        time.sleep(0.25)
        st.rerun()
    elif st.session_state.live_test_running and not pipeline.is_running():
        st.session_state.live_test_running = False
        st.warning("Pipeline is no longer running.")