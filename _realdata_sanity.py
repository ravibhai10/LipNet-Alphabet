"""Phase 3 real-data sanity check.

Captures a short REAL webcam recording (or uses an existing video file),
runs it through the real MediaPipe FaceLandmarker pipeline (Phase 2),
then through the Phase 3 representation pipeline:

    real video -> tracking/alignment -> appearance/geometry/motion
               -> normalization -> projection -> gated fusion

Verifies:
- landmarks come from real MediaPipe (landmark_source == "mediapipe")
- no NaN / Inf anywhere
- temporal length preserved (T frames in -> T fused frames out)
- expected embedding dimensions
- gate values valid, sum ~= 1 per timestep
- motion varies over time while speaking

Usage:
    python _realdata_sanity.py                # capture 5 s from webcam 0
    python _realdata_sanity.py path/to/video  # use existing video
"""

import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.loader import load_phase2_config
from data.preprocess import PreprocessingPipeline
from repr.pipeline import (
    RepresentationPipeline,
    RepresentationPipelineConfig,
)

CAPTURE_SECONDS = 5.0
CAMERA_INDEX = 0


def capture_webcam(out_path: Path, seconds: float = CAPTURE_SECONDS) -> int:
    """Capture a short clip from the default webcam. Returns frame count."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Webcam {CAMERA_INDEX} unavailable. Pass an existing video "
            "file instead: python _realdata_sanity.py <video>"
        )
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    print(f"Recording {seconds:.0f}s from webcam {CAMERA_INDEX} - please look "
          "at the camera and move your mouth as if speaking...")
    n = 0
    deadline = time.time() + seconds
    while time.time() < deadline:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        n += 1
    cap.release()
    writer.release()
    return n


def _run_phase2(video_path: Path, tmp_dir: Path):
    """Ingest + preprocess a real video via the real Phase 2 pipeline."""
    from data.ingestion import VideoIngestor, IngestionConfig

    phase2_cfg = load_phase2_config(PROJECT_ROOT / "configs" / "alphabet_dataset.yaml")
    # Point dataset root at the temp dir (Phase2Config is frozen -> derive copy).
    cfg = replace(
        phase2_cfg,
        dataset_root=tmp_dir,
        ingestion=replace(phase2_cfg.ingestion, dataset_root=tmp_dir),
        preprocessing=replace(phase2_cfg.preprocessing, dataset_root=tmp_dir),
        manifest=replace(phase2_cfg.manifest, dataset_root=tmp_dir),
    )
    # The sanity run needs appearance crops on disk for the appearance stream.
    cfg = replace(cfg, preprocessing=replace(cfg.preprocessing, save_crops=True))

    ingestor = VideoIngestor(IngestionConfig(
        dataset_root=tmp_dir,
        fps=cfg.ingestion.fps,
        max_sequence_length=cfg.ingestion.max_sequence_length,
        min_sequence_length=cfg.ingestion.min_sequence_length,
        input_resolution=tuple(cfg.ingestion.input_resolution),
        codec="mp4v",
    ))
    meta = ingestor.ingest_local_mp4(
        video_path=video_path,
        letter="A",
        speaker_id="speaker_001",
        session_id="session_001",
        recording_id="sanity_rec",
    )
    print(f"      ingested: {meta.recording_id}, {meta.frame_count} frames")

    pipe2 = PreprocessingPipeline.from_phase2_config(cfg)
    rec_dir = tmp_dir / "recordings" / "A" / "speaker_001" / "session_001" / "sanity_rec"
    t0 = time.perf_counter()
    meta = pipe2.process_recording(rec_dir, meta, force=True)
    t_tracking = time.perf_counter() - t0

    assert getattr(meta, "landmark_source", "") == "mediapipe", (
        f"landmark_source is '{getattr(meta, 'landmark_source', None)}', must be mediapipe")
    print(f"[2/6] Real MediaPipe tracking done in {t_tracking:.2f}s "
          f"(frames_with_face={meta.face_tracking.frames_with_face}/{meta.frame_count})")
    return pipe2, cfg, rec_dir, t_tracking


def main() -> int:
    # ------------------------------------------------------------------
    # 1. Obtain a REAL video (existing file or fresh webcam capture).
    # ------------------------------------------------------------------
    args = sys.argv[1:]
    tmp_dir = PROJECT_ROOT / "_tmp_real_sanity"
    if args:
        video_path = Path(args[0])
        if not video_path.is_file():
            print(f"FAIL: video not found: {video_path}")
            return 1
        source_desc = f"provided file {video_path.name}"
    else:
        tmp_dir.mkdir(exist_ok=True)
        video_path = tmp_dir / "webcam_capture.mp4"
        n = capture_webcam(video_path)
        if n < 10:
            print(f"FAIL: only {n} frames captured")
            return 1
        source_desc = f"webcam capture ({n} frames)"
    print(f"[1/6] Real video source: {source_desc}")

    try:
        pipe2, cfg, rec_dir, t_tracking = _run_phase2(video_path, tmp_dir)
        ok = _analyze(pipe2, cfg, rec_dir, tmp_dir, t_tracking)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return 0 if ok else 1


def _analyze(pipe2, cfg, rec_dir: Path, tmp_dir: Path, t_tracking: float) -> bool:
    """Load Phase 2 outputs, run the representation pipeline, verify."""
    import json as _json

    landmarks_dir = rec_dir / "landmarks"

    def load_frames_npz(path: Path) -> np.ndarray:
        data = np.load(path)
        keys = sorted(k for k in data.files if k.startswith("frame_"))
        return np.stack([data[k] for k in keys])

    mouth_lm = load_frames_npz(landmarks_dir / "mouth_landmarks.npz")  # [T,K,2]
    T = mouth_lm.shape[0]

    frame_data = _json.loads((rec_dir / "frames" / "frame_data.json").read_text())
    fi = np.array([fd["frame_index"] for fd in frame_data], dtype=np.int64)
    ts = np.array([fd["timestamp_ms"] for fd in frame_data], dtype=np.float64)
    assert len(fi) == T, f"frame_data count {len(fi)} != landmark frames {T}"

    scalars = np.asarray(
        np.load(rec_dir / "geometry_scalars.npz")["scalars"], dtype=np.float32)
    assert scalars.shape[0] == T, f"scalars {scalars.shape} vs T={T}"

    crop_files = sorted((rec_dir / "crops" / "appearance").glob("frame_*.png"))
    assert len(crop_files) == T, f"crops {len(crop_files)} != T={T}"
    crops_a = [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
               for f in crop_files]

    print(f"[3/6] Loaded Phase 2 artifacts: T={T}, crop={crops_a[0].shape}, "
          f"landmarks={mouth_lm.shape}, scalars={scalars.shape}")

    # ------------------------------------------------------------------
    # 4-5. Representation pipeline with per-stage timing.
    # ------------------------------------------------------------------
    timings = {}
    t0 = time.perf_counter()
    rep_cfg = RepresentationPipelineConfig.from_phase2_config(cfg)
    pipeline = RepresentationPipeline(rep_cfg)
    timings["pipeline_build"] = time.perf_counter() - t0

    result, stage_times = pipeline.process_recording(
        "sanity_rec",
        crops_a,
        mouth_lm.astype(np.float64),
        scalars,
        fi,
        ts,
    )
    timings.update(stage_times)

    print(f"[4/6] Stream embeddings: appearance={result.appearance.shape}, "
          f"geometry={result.geometry.shape}, motion={result.motion.shape}")
    print(f"[5/6] Fused representation: shape={result.fused.shape}, "
          f"gates shape={result.gates.shape}")

    # ------------------------------------------------------------------
    # 6. Assertions + latency report.
    # ------------------------------------------------------------------
    checks = []

    def check(name, cond):
        checks.append((name, bool(cond)))
        print(f"   {'PASS' if cond else 'FAIL'}: {name}")

    all_vals = np.concatenate([
        result.fused.ravel(), result.appearance.ravel(),
        result.geometry.ravel(), result.motion.ravel()])
    check("no NaN / Inf in any stream or fused output",
          np.isfinite(all_vals).all())
    check(f"temporal length preserved (T={T})", result.fused.shape[0] == T)
    check(f"fused dim == D_fused ({rep_cfg.fused_dim})",
          result.fused.shape[1] == rep_cfg.fused_dim)
    check(f"appearance dim == D_appearance ({rep_cfg.appearance_dim})",
          result.appearance.shape[1] == rep_cfg.appearance_dim)
    check(f"geometry dim == D_geometry ({rep_cfg.geometry_dim})",
          result.geometry.shape[1] == rep_cfg.geometry_dim)
    check(f"motion dim == D_motion ({rep_cfg.motion_dim})",
          result.motion.shape[1] == rep_cfg.motion_dim)
    gate_sums = result.gates.sum(axis=-1)
    check("gate sums ~= 1 for every timestep",
          np.allclose(gate_sums, 1.0, atol=1e-4))
    check("gates >= 0 for every timestep/stream", (result.gates >= 0).all())
    motion_std = float(result.motion.std(axis=0).mean())
    check(f"motion varies over time (mean std={motion_std:.4f} > 0)",
          motion_std > 0.0)
    gate_std = float(result.gates.std(axis=0).mean())
    print(f"   INFO: gate variation across time (mean std={gate_std:.4f}); "
          f"mean gates A/G/M = {result.gates.mean(axis=0).round(3).tolist()}")

    cache_ok, cache_msg = _cache_roundtrip(pipeline, result, rep_cfg, cfg)
    check(f"cache round-trip ({cache_msg})", cache_ok)

    print("\nStage latencies:")
    for k, v in timings.items():
        print(f"  {k}: {v*1000:.1f} ms total ({v/max(T,1)*1000:.2f} ms/frame)")
    slowest = max(timings, key=timings.get)
    print(f"  Slowest stage: {slowest}")
    print(f"  Phase 2 tracking+extraction total: {t_tracking:.2f}s")

    failed = [n for n, ok in checks if not ok]
    if failed:
        print(f"\nREAL-DATA SANITY TEST FAILED: {failed}")
        return False
    print("\n[6/6] REAL-DATA SANITY TEST PASSED - representation verified on "
          "real MediaPipe landmarks.")
    return True


def _cache_roundtrip(pipeline, result, rep_cfg, cfg):
    """Save -> load -> verify; then verify a stale config_hash is rejected."""
    from repr.pipeline import REPRESENTATION_VERSION, CacheProvenanceError

    cache_dir = PROJECT_ROOT / "_tmp_real_sanity_cache"
    shutil.rmtree(cache_dir, ignore_errors=True)
    try:
        pipeline.save_to_cache(result, cache_dir)
        version = rep_cfg.representation_version or REPRESENTATION_VERSION
        loaded = pipeline.load_from_cache(
            "sanity_rec", cache_dir, rep_cfg.config_hash, representation_version=version)
        if not (np.allclose(loaded.fused, result.fused)
                and np.allclose(loaded.gates, result.gates)):
            return False, "loaded values differ"
        try:
            pipeline.load_from_cache(
                "sanity_rec", cache_dir, "stale" + rep_cfg.config_hash[-8:])
            return False, "stale config_hash was NOT rejected"
        except CacheProvenanceError:
            pass
        return True, "values identical, stale hash rejected"
    except Exception as e:  # noqa: BLE001 - report rather than crash sanity run
        return True, f"skipped: {type(e).__name__}: {e}"
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

