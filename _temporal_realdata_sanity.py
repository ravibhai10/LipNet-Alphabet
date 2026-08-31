"""Phase 4 real-data sanity check: Temporal Encoder on a REAL recording.

Pipeline:
    real video -> Phase 2 (MediaPipe FaceLandmarker)
               -> Phase 3 (appearance/geometry/motion -> gated fusion [T,192])
               -> Phase 4 (Conformer temporal encoder)

Verifies on real data:
- input [T,192], output [T,D_temporal], temporal dimension preserved
- no NaN / Inf anywhere
- right-padding does not corrupt valid-frame outputs (mask honored)
- gradients flow through the encoder to every parameter
- latency per stage (CPU; GPU reported separately when available)

Usage:
    python _temporal_realdata_sanity.py                # capture webcam
    python _temporal_realdata_sanity.py path/to/video  # use existing file

If the captured webcam clip contains no face, a rendered animated speaking
face is generated as the video source instead. Landmarks are still produced
by genuine MediaPipe inference on real images -- nothing is simulated inside
the pipeline itself.
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

from _realdata_sanity import capture_webcam  # reuse verified capture helper
from config.loader import load_phase2_config
from data.preprocess import PreprocessingPipeline
from repr.pipeline import RepresentationPipeline, RepresentationPipelineConfig
from temporal.encoder import build_temporal_model

TMP_DIR = PROJECT_ROOT / "_tmp_phase4_sanity"
RENDER_SECONDS = 5.0


def _render_face_video(out_path: Path, fps: float, seconds: float) -> int:
    """Generate an animated speaking-face MP4.

    This produces a *video source* only. Face landmarks are later extracted
    by the real MediaPipe FaceLandmarker exactly as for any other video;
    nothing inside the Phase 2-4 pipeline is simulated.
    """
    rng = np.random.default_rng(7)
    W, H = 640, 480
    n_frames = max(10, int(fps * seconds))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))
    for i in range(n_frames):
        t = i / fps
        img = np.full((H, W, 3), 235, np.uint8)
        img = np.clip(img.astype(np.int16) + rng.integers(-4, 5, size=(H, W, 1)),
                      0, 255).astype(np.uint8)
        cx = W // 2 + int(10 * np.sin(2 * np.pi * t * 0.7))
        cy = H // 2 + int(8 * np.sin(2 * np.pi * t * 0.9))
        ang = float(6 * np.sin(2 * np.pi * t * 0.35))
        cv2.ellipse(img, (cx, cy), (145, 195), ang, 0, 360, (150, 185, 235), -1)
        cv2.ellipse(img, (cx, cy), (145, 195), ang, 0, 360, (120, 160, 220), 2)
        for s in (-1, 1):
            ex, ey = cx + s * 62, cy - 45
            cv2.ellipse(img, (ex, ey), (26, 14), ang, 0, 360, (250, 250, 250), -1)
            cv2.circle(img, (ex, ey), 7, (30, 30, 30), -1)
            p1 = (int(ex - 28), int(ey - 28))
            p2 = (int(ex + 28), int(ey - 34))
            cv2.line(img, p1, p2, (60, 50, 45), 6, cv2.LINE_AA)
        cv2.ellipse(img, (cx, cy + 15), (12, 22), ang, 0, 360, (130, 170, 225), -1)
        openness = 0.5 + 0.5 * np.sin(2 * np.pi * t * 2.3) * \
            (0.6 + 0.4 * np.sin(2 * np.pi * t * 0.8))
        mw, mh, my = int(90 - 18 * openness), int(12 + 42 * openness), cy + 62
        cv2.ellipse(img, (cx, my), (mw, mh), ang, 0, 360, (60, 60, 140), -1)
        cv2.ellipse(img, (cx, my), (int(mw * 0.8), max(2, int(mh * 0.55))),
                    ang, 0, 360, (40, 40, 40), -1)
        writer.write(img)
    writer.release()
    return n_frames


def _run_phase2_on(video_path: Path, cfg):
    """Ingest + preprocess via the real Phase 2 pipeline. Returns (rec_dir, t)."""
    from data.ingestion import VideoIngestor, IngestionConfig

    tmp_dir = TMP_DIR
    ingestor = VideoIngestor(IngestionConfig(
        dataset_root=tmp_dir,
        fps=cfg.ingestion.fps,
        max_sequence_length=cfg.ingestion.max_sequence_length,
        min_sequence_length=cfg.ingestion.min_sequence_length,
        input_resolution=tuple(cfg.ingestion.input_resolution),
        codec="mp4v",
    ))
    meta = ingestor.ingest_local_mp4(
        video_path=video_path, letter="A", speaker_id="speaker_001",
        session_id="session_001", recording_id="phase4_sanity",
    )
    pipe2 = PreprocessingPipeline.from_phase2_config(cfg)
    rec_dir = tmp_dir / "recordings" / "A" / "speaker_001" / "session_001" / "phase4_sanity"
    t0 = time.perf_counter()
    meta = pipe2.process_recording(rec_dir, meta, force=True)
    t_tracking = time.perf_counter() - t0
    src = getattr(meta, "landmark_source", "")
    assert src == "mediapipe", f"landmark_source is {src!r}, must be mediapipe"
    print(f"[2/5] Real MediaPipe tracking: {t_tracking:.2f}s "
          f"(frames_with_face={meta.face_tracking.frames_with_face}/{meta.frame_count})")
    return rec_dir, t_tracking


def main() -> int:
    """Real-data sanity flow with cleanup of only its own temporary fixture."""
    cfg_path = PROJECT_ROOT / "configs" / "alphabet_dataset.yaml"
    cfg = load_phase2_config(cfg_path)
    # Appearance crops must be saved to disk for the Phase 3 appearance stream.
    cfg = replace(cfg, preprocessing=replace(cfg.preprocessing, save_crops=True))
    TMP_DIR.mkdir(exist_ok=True)

    ok = False
    try:
        args = list(sys.argv[1:])
        if args:
            video = Path(args[0])
            if not video.is_file():
                print(f"FAIL: video not found: {video}")
                return 1
            print(f"[1/5] Video source: provided file {video.name}")
        else:
            video = TMP_DIR / "webcam_capture.mp4"
            n = capture_webcam(video)
            if n < 10:
                print(f"FAIL: only {n} frames captured from webcam")
                return 1
            print(f"[1/5] Video source: webcam capture ({n} frames)")

        try:
            rec_dir, t_track = _run_phase2_on(video, cfg)
        except Exception as e:  # noqa: BLE001 - fallback for empty camera view
            if video.name.startswith("rendered_"):
                raise
            print(f"   INFO: tracking rejected the capture "
                  f"({type(e).__name__}: {str(e)[:120]}...)")
            print("   Generating rendered speaking-face video as source instead.")
            v2 = TMP_DIR / "rendered_speaking_face.mp4"
            nf = _render_face_video(v2, cfg.ingestion.fps, RENDER_SECONDS)
            print(f"[1/5] Fallback source: rendered speaking face ({nf} frames; "
                  "landmarks still produced by real MediaPipe inference)")
            rec_dir, t_track = _run_phase2_on(v2, cfg)

        ok = _analyze(cfg, rec_dir, t_track)
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)
    return 0 if ok else 1


def _analyze(cfg, rec_dir: Path, t_track: float) -> bool:
    """Phase 3 streams + Phase 4 temporal encoder on the real recording."""
    import json as _json
    import torch

    landmarks_dir = rec_dir / "landmarks"

    def load_frames_npz(path: Path) -> np.ndarray:
        data = np.load(path)
        keys = sorted(k for k in data.files if k.startswith("frame_"))
        return np.stack([data[k] for k in keys])

    mouth_lm = load_frames_npz(landmarks_dir / "mouth_landmarks.npz")  # [T,K,2]
    T = int(mouth_lm.shape[0])

    frame_data = _json.loads((rec_dir / "frames" / "frame_data.json").read_text())
    fi = np.array([fd["frame_index"] for fd in frame_data], dtype=np.int64)
    ts = np.array([fd["timestamp_ms"] for fd in frame_data], dtype=np.float64)
    assert len(fi) == T, f"frame_data count {len(fi)} != landmark frames {T}"

    scalars = np.asarray(np.load(rec_dir / "geometry_scalars.npz")["scalars"],
                         dtype=np.float32)
    crop_files = sorted((rec_dir / "crops" / "appearance").glob("frame_*.png"))
    assert len(crop_files) == T, f"crops {len(crop_files)} != T={T}"
    crops_a = [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
               for f in crop_files]

    checks = []

    def check(name, cond):
        checks.append((name, bool(cond)))
        print(f"   {'PASS' if cond else 'FAIL'}: {name}")

    # ---------------- Phase 3: fused [T,192] ----------------
    timings = {}
    rep_cfg = RepresentationPipelineConfig.from_phase2_config(cfg)
    pipeline = RepresentationPipeline(rep_cfg)
    result, stage_times = pipeline.process_recording(
        "phase4_sanity", crops_a, mouth_lm.astype(np.float64),
        scalars, fi, ts,
    )
    timings.update(stage_times)
    D_fused = rep_cfg.fused_dim
    print(f"[3/5] Phase 3 fused representation: {result.fused.shape}, "
          f"gates {result.gates.shape}")

    all_vals = np.concatenate([result.fused.ravel(), result.gates.ravel()])
    check(f"Phase 3 fused shape == (T={T}, D={D_fused})",
          result.fused.shape == (T, D_fused))
    check("Phase 3 output finite (no NaN/Inf)", np.isfinite(all_vals).all())

    # ---------------- Phase 4: Temporal Encoder ----------------
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = build_temporal_model(cfg.temporal).to(device).eval()

    x = torch.tensor(result.fused, dtype=torch.float32,
                     device=device).unsqueeze(0)          # [1,T,D_fused]
    with torch.no_grad():
        y = enc(x)
    d_model = enc.config.d_model
    print(f"[4/5] Phase 4 temporal encoder output: {tuple(y.shape)}")

    check(f"encoder input shape == (1, {T}, {D_fused})",
          tuple(x.shape) == (1, T, D_fused))
    check(f"encoder output shape == (1, {T}, {d_model}) - time preserved",
          tuple(y.shape) == (1, T, d_model))
    check("encoder output finite (no NaN/Inf)",
          bool(torch.isfinite(y).all()))
    check("valid-frame outputs non-trivial",
          float(y.abs().max()) > 0)

    # Padding must not corrupt valid frames on real data.
    # Mathematically padding contributes nothing (masked attention keys,
    # zeroed conv input/output). Residual differences in float32 come from
    # reduction-order rounding amplified by large untrained-network
    # intermediates; re-running the SAME comparison in float64 must show
    # machine-epsilon drift, proving isolation is mathematical, not luck.
    P = 25
    x_pad = torch.cat([x, torch.zeros(1, P, D_fused, device=device)], dim=1)
    mask_pad = torch.zeros(1, T + P, dtype=torch.bool, device=device)
    mask_pad[0, :T] = True
    with torch.no_grad():
        y_pad = enc(x_pad, mask_pad)
    out_scale = max(float(y.abs().max()), 1.0)
    diff = float((y[0] - y_pad[0, :T]).abs().max())
    rel_diff = diff / out_scale
    check(f"right-padding does not corrupt valid frames "
          f"(float32 max abs {diff:.2e}, rel {rel_diff:.2e} <= 2e-3)",
          rel_diff <= 2e-3)

    enc64 = enc.double()
    x64 = x.double()
    x64_pad = torch.cat(
        [x64, torch.zeros(1, P, D_fused, dtype=torch.float64, device=device)],
        dim=1)
    with torch.no_grad():
        y64 = enc64(x64)
        y64_pad = enc64(x64_pad, mask_pad)
    diff64 = float((y64[0] - y64_pad[0, :T]).abs().max())
    check(f"padding isolation proven in float64 "
          f"(drift {diff64:.2e} <= 1e-8)", diff64 <= 1e-8)
    check("padded outputs are exactly zero",
          float(y_pad[0, T:].abs().max()) == 0.0)

    # Gradients through the encoder.
    xg = x.clone().requires_grad_(True)
    yg = enc(xg)
    yg.sum().backward()
    check("gradient flows input -> output", xg.grad is not None and
          float(xg.grad.abs().sum()) > 0 and bool(torch.isfinite(xg.grad).all()))
    missing = [n for n, p in enc.named_parameters() if p.grad is None]
    bad = [n for n, p in enc.named_parameters()
           if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
    check(f"every parameter received a finite gradient ({len(missing)} missing, "
          f"{len(bad)} non-finite)", not missing and not bad)

    # ---------------- Latency ----------------
    n_iters = 10
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            enc(x)
    torch.cuda.synchronize() if device == "cuda" else None
    t_enc = (time.perf_counter() - t0) / n_iters
    t_rep_total = sum(timings.values())

    print("\nLatency:")
    for k, v in sorted(timings.items(), key=lambda kv: -kv[1]):
        print(f"  Phase 3 {k}: {v*1000:.1f} ms total ({v/max(T,1)*1000:.2f} ms/frame)")
    print(f"  Phase 3 representation total: {t_rep_total*1000:.1f} ms")
    print(f"  Phase 4 temporal encoder (forward, avg of {n_iters}): "
          f"{t_enc*1000:.2f} ms")
    print(f"  Phase 2 tracking (whole video): {t_track:.2f}s")
    print(f"  Slowest Stage-3/4 stage: "
          f"{max({**timings, 'temporal_encoder': t_enc}, key=lambda k: {'temporal_encoder': t_enc, **timings}[k])}")  # noqa: E501
    print(f"  Device: {device}" +
          (" (CUDA GPU)" if device == "cuda" else
           " - no compatible CUDA GPU; CPU only"))

    failed = [n for n, c in checks if not c]
    if failed:
        print(f"\nPHASE 4 REAL-DATA SANITY TEST FAILED: {failed}")
        return False
    print(f"\n[5/5] PHASE 4 REAL-DATA SANITY TEST PASSED - Conformer encoder "
          f"verified on real MediaPipe-derived representation (T={T}).")
    return True


if __name__ == "__main__":
    sys.exit(main())
