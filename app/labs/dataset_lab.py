"""
DatasetLab - record real webcam samples, view stats, validate dataset.
"""
from __future__ import annotations

import streamlit as st
from typing import Optional

from app.services.dataset_service import get_dataset_service, RecordingResult


def render():
    st.header("Dataset Lab")
    st.caption("Record real webcam samples - data_source=real_camera enforced")
    ds = get_dataset_service()
    tab_record, tab_stats, tab_validate = st.tabs(["Record", "Stats", "Validate"])
    with tab_record:
        _record(ds)
    with tab_stats:
        _stats(ds)
    with tab_validate:
        _validate(ds)


def _record(ds):
    col1, col2, col3 = st.columns(3)
    letter = col1.selectbox("Letter", list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    speaker_id = col2.text_input("Speaker ID", value="speaker_001", key="dl_spk")
    session_id = col3.text_input("Session ID", value="session_001", key="dl_sess")
    col_d, col_f, col_c = st.columns(3)
    duration = col_d.slider("Duration (s)", 1.0, 10.0, 3.0, 0.5)
    fps = col_f.selectbox("FPS", [15, 24, 30, 60], index=2)
    device = col_c.selectbox("Camera", list(range(5)), index=0)
    if st.button("START RECORDING", type="primary", use_container_width=True):
        if not speaker_id.strip():
            st.error("speaker_id is required.")
        elif not session_id.strip():
            st.error("session_id is required.")
        else:
            with st.spinner(f"Recording {letter}..."):
                result = ds.record(letter=letter, speaker_id=speaker_id.strip(),
                                 session_id=session_id.strip(), duration_seconds=duration,
                                 fps=fps, device_index=device, show_preview=True)
                if result.success:
                    st.success(f"Saved: {result.recording_id}")
                    if result.validation_errors:
                        for e in result.validation_errors:
                            st.code(e)
                else:
                    st.error(f"Failed: {result.error}")
    stats = ds.get_stats()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recordings", stats["recordings"])
    m2.metric("Speakers", stats["speakers"])
    m3.metric("Sessions", stats["sessions"])
    m4.metric("Letters", stats["letters"])
    if stats["missing_letters"]:
        st.caption(f"Missing: {', '.join(stats['missing_letters'])}")


def _stats(ds):
    s = ds.get_stats()
    c1, c2 = st.columns(2)
    c1.metric("Total Recordings", s["recordings"])
    c1.metric("Speakers", s["speakers"])
    c1.metric("Sessions", s["sessions"])
    c2.metric("Letters Covered", f"{s['letters']}/26")
    c2.metric("Min Class", s["min_count"])
    c2.metric("Max Class", s["max_count"])
    if s["missing_letters"]:
        st.warning(f"Missing: {', '.join(s['missing_letters'])}")
    if s["letter_counts"]:
        st.bar_chart(s["letter_counts"])


def _validate(ds):
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Build Manifest", use_container_width=True):
            try:
                n = ds.build_manifest()
                st.success(f"Manifest: {n} entries.")
            except Exception as e:
                st.error(f"Error: {e}")
    with c2:
        if st.button("Create Splits", use_container_width=True):
            try:
                counts = ds.create_splits()
                st.success(f"Splits: {counts}")
            except Exception as e:
                st.error(f"Error: {e}")
    if st.button("VALIDATE DATASET", type="primary", use_container_width=True):
        with st.spinner("Validating..."):
            report = ds.validate()
            if report.has_critical_errors():
                st.error(f"FAILED - {report.critical_errors} critical errors")
            else:
                st.success("PASSED")
            for r in report.results:
                icon = "OK" if r.passed else "FAIL"
                with st.expander(f"[{icon}] {r.check_name}"):
                    st.code(r.message)
                    if r.details:
                        for d in r.details[:5]:
                            st.code(d)

        _validate(ds)
