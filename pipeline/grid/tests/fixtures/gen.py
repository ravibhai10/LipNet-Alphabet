"""Synthetic GRID-like fixtures for unit tests (no full corpus required).

Creates minimal tree shaped like one GRID talker and writes placeholder
audio/video/alignment files so tests can run offline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path


def make_fixture(name: str = "fixture_t01") -> Path:
    """Create a synthetic GRID talker tree and return the root dir."""
    fd, tmp = tempfile.mkstemp(prefix="grid_fixture_", suffix=".tmp")
    import os
    os.close(fd)
    root = Path(tmp).resolve()
    (root / "audio" / "25kHz").mkdir(parents=True, exist_ok=True)
    (root / "video").mkdir(parents=True, exist_ok=True)
    (root / "alignment").mkdir(parents=True, exist_ok=True)

    (root / "audio" / "25kHz" / "T01_a.wav").write_bytes(b"fake-audio")
    (root / "video" / "T01_v.mp4").write_bytes(b"fake-video")
    (root / "alignment" / "T01.align").write_bytes(
        b"0\t1000\tput\n1100\t2000\tred\n2100\t3000\tat\n3100\t4000\t9\n4100\t5000\tnow\n"
    )
    (root / "T01_U0001").mkdir(parents=True, exist_ok=True)

    manifest = root / "manifest.jsonl"
    manifest.write_text(
        '{"talk_id":"T01","speaker_id":"grid_T01","utterance_id":"T01_U0001",'
        '"video_path":"T01_v.mp4","audio_path":"audio/25kHz/T01_a.wav",'
        '"alignment_path":"T01.align","frame_count":0,"fps":0.0,'
        '"transcript":"put red at 9 now","alignment_status":"ok",'
        '"validation_status":"ok","value":"put red at 9 now"}\n',
        encoding="utf-8",
    )
    return root


def cleanup(path: Path) -> None:
    import shutil
    shutil.rmtree(path, ignore_errors=True)
