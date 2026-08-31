"""
CLI entry point for alphabet dataset tools.

Provides:
- python -m alphabet validate-dataset <dataset_root>
- python -m alphabet inspect-dataset <dataset_root>
- python -m alphabet record --speaker-id X --session-id Y A    (real webcam, real_camera)
- python -m alphabet ingest <args>
- python -m alphabet preprocess <dataset_root>                  (writes manifest.jsonl)
"""

import sys
import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

from data.validate import validate_dataset
from data.inspect import inspect_dataset
from data.ingestion import VideoIngestor, IngestionConfig
from data.schema import SourceType


def validate_cmd(args: argparse.Namespace) -> int:
    """Validate dataset command."""
    return validate_dataset(args.dataset_root)


def inspect_cmd(args: argparse.Namespace) -> int:
    """Inspect dataset command."""
    return inspect_dataset(args.dataset_root)


def ingest_cmd(args: argparse.Namespace) -> int:
    """Ingest video command."""
    config = IngestionConfig(
        dataset_root=Path(args.dataset_root),
        fps=args.fps,
        max_sequence_length=args.max_len,
        min_sequence_length=args.min_len,
        input_resolution=(args.width, args.height),
        codec="mp4v",
        blur_threshold=100.0,
        brightness_range=(20, 235),
        min_mouth_size=400,
        max_pose_angle=30.0,
        duplicate_ssim_threshold=0.98,
        save_frames=False,
        save_landmarks_npz=True,
        save_crops=False,
        preprocessing_version=args.version,
        config_version="1.0",
        config_hash="",
    )

    ingestor = VideoIngestor(config)

    if args.source == "webcam":
        metadata = ingestor.record_webcam(
            letter=args.letter,
            speaker_id=args.speaker,
            session_id=args.session,
            recording_id=args.recording_id,
            duration_seconds=args.duration,
            lighting_tag=args.lighting,
            accent_tag=args.accent,
            notes=args.notes,
            device_index=args.device,
            show_preview=args.preview,
        )
    else:  # local_mp4
        metadata = ingestor.ingest_local_mp4(
            video_path=Path(args.video_path),
            letter=args.letter,
            speaker_id=args.speaker,
            session_id=args.session,
            recording_id=args.recording_id,
            lighting_tag=args.lighting,
            accent_tag=args.accent,
            notes=args.notes,
        )

    print(f"✅ Ingested: {metadata.recording_id} ({metadata.letter})")
    print(f"   Speaker: {metadata.speaker_id}, Session: {metadata.session_id}")
    print(f"   Frames: {metadata.frame_count}, Duration: {metadata.duration_ms:.0f}ms")
    return 0


def preprocess_cmd(args: argparse.Namespace) -> int:
    """Preprocess dataset command.

    Configuration precedence (highest first):
        CLI overrides  >  YAML (--config)  >  typed dataclass defaults
    """
    from config.loader import load_phase2_config

    overrides: Dict[str, Any] = {}
    if args.fps is not None:
        overrides["video.fps"] = args.fps
    if args.crop_margin is not None:
        overrides["mouth_alignment.crop_margin_multiplier"] = args.crop_margin
    if args.no_resume:
        overrides["preprocessing.resume"] = False
    if args.force_all:
        overrides["preprocessing.resume"] = False

    cfg = load_phase2_config(args.config, overrides or None)

    # The CLI positional wins over YAML dataset_root.
    # Phase2Config is frozen by design, so derive an updated copy.
    root = Path(args.dataset_root)
    cfg = replace(
        cfg,
        dataset_root=root,
        ingestion=replace(cfg.ingestion, dataset_root=root),
        preprocessing=replace(cfg.preprocessing, dataset_root=root),
        manifest=replace(cfg.manifest, dataset_root=root),
    )

    from data.preprocess import PreprocessingPipeline

    pipeline = PreprocessingPipeline.from_phase2_config(cfg)

    print(f"Config: {args.config}  (config_version={cfg.config_version}, "
          f"config_hash={cfg.config_hash[:12]}...)")
    print(f"Dataset root: {cfg.dataset_root}")
    print(f"crop_margin_multiplier={cfg.tracking.crop_margin_multiplier}, "
          f"fps={cfg.ingestion.fps}")

    results = pipeline.process_all(force=args.force_all)
    print(f"Processed {len(results)} recording(s)")

    summary = pipeline.build_manifest_and_splits()
    print(f"Manifest entries: {summary['entries']}")
    return 0


# ---------------------------------------------------------------------------
# Phase 5: letter-decoder commands
# ---------------------------------------------------------------------------

def _decoder_context(args):
    """Load both the Phase 5 decoder config and the frozen Phase 2-4 config."""
    from alphabet.config import load_decoder_config
    from config.loader import load_phase2_config

    return load_decoder_config(args.config), load_phase2_config(args.config)


def cache_reprs_cmd(args) -> int:
    """Build the Phase 3 representation cache used by decoder training."""
    from alphabet.cache import build_representation_cache

    summary = build_representation_cache(
        dataset_root=Path(args.dataset_root),
        yaml_config_path=Path(args.config),
        ablation=args.ablation,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        limit=args.limit,
        splits=args.splits.split(",") if args.splits else None,
    )
    print(f"Representation cache: {summary['n_ok']} ok, "
          f"{summary['n_failed']} failed -> {summary['cache_dir']} "
          f"(hash {summary['config_hash'][:12]}..., ablation={summary['ablation']})")
    for f in summary["failures"]:
        print(f"  FAILED {f['recording']}: {f['error']}")
    return 0 if summary["n_failed"] == 0 and summary["n_ok"] > 0 else 1


def train_decoder_cmd(args) -> int:
    """Train the letter-CTC decoder (Phase 5 primary task)."""
    from alphabet.train import train_decoder

    dec_cfg, phase2_cfg = _decoder_context(args)
    overrides = {}
    if args.max_epochs:
        overrides["training.epochs"] = args.max_epochs
    if overrides:
        from alphabet.config import load_decoder_config
        dec_cfg = load_decoder_config(args.config, overrides)
    dec_cfg.training.checkpoint_dir = args.checkpoint_dir or dec_cfg.training.checkpoint_dir
    if args.ablation:
        dec_cfg.ablation = args.ablation
    summary = train_decoder(
        dec_cfg, phase2_cfg,
        limit_train=args.limit_train, limit_val=args.limit_val,
        max_steps=args.max_steps, device=args.device,
        resume=not args.no_resume, run_test=args.run_test,
    )
    print("Training summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


def sanity_decoder_cmd(args) -> int:
    """§25 real-data sanity: Phase 2 -> 3 (cache) -> 4 -> decoder."""
    from alphabet.sanity import real_data_sanity

    dec_cfg, phase2_cfg = _decoder_context(args)
    result = real_data_sanity(dec_cfg, phase2_cfg, n_max=args.n_max,
                              device=args.device)
    print(f"REAL-DATA SANITY: {result['status']}")
    if result.get("reason"):
        print(f"  reason: {result['reason']}")
    for c in result.get("checks", []):
        print(f"   {'PASS' if c['ok'] else 'FAIL'}: {c['check']} {c.get('info', '')}")
    return 0 if result["status"] == "PASS" else 1


def overfit_decoder_cmd(args) -> int:
    """§26 tiny-subset overfit: data -> target -> model -> loss -> gradients."""
    from alphabet.sanity import tiny_overfit

    dec_cfg, phase2_cfg = _decoder_context(args)
    result = tiny_overfit(dec_cfg, phase2_cfg, steps=args.steps,
                          device=args.device)
    print(f"TINY-SUBSET OVERFIT: {result['status']} ({result['data_source']})")
    print(f"  steps={result['steps_run']} final_loss={result['final_loss']:.4f} "
          f"final_accuracy={result['final_accuracy']:.2f}")
    return 0 if result["status"] == "PASS" else 1
def record_cmd(args: argparse.Namespace) -> int:
    """Record real webcam video to the production dataset.

    This command is the RECOMMENDED way to add real human recordings to
    alphabet_dataset/ (production). It sets data_source="real_camera"
    automatically, ensuring the recording passes the production data gate.

    Raises a clear error if:
    - Webcam cannot be opened
    - No frames are captured
    - Letter / speaker / session ID is invalid
    """
    config = IngestionConfig(
        dataset_root=Path(args.dataset_root),
        fps=args.fps,
        max_sequence_length=76,
        min_sequence_length=15,
        input_resolution=(args.width, args.height),
        codec="mp4v",
        preprocessing_version="1.0",
        config_version="1.0",
    )

    ingestor = VideoIngestor(config)

    # Validate letter
    from data.schema import Letter
    if not Letter.is_valid(args.letter):
        print(f"ERROR: Invalid letter '{args.letter}'. Must be A-Z (uppercase).")
        return 1
    if not args.speaker_id:
        print("ERROR: --speaker-id is required.")
        return 1
    if not args.session_id:
        print("ERROR: --session-id is required.")
        return 1

    try:
        metadata = ingestor.record_webcam(
            letter=args.letter.upper(),
            speaker_id=args.speaker_id,
            session_id=args.session_id,
            recording_id=args.recording_id,
            duration_seconds=args.duration,
            lighting_tag=args.lighting,
            accent_tag=args.accent,
            notes=args.notes,
            device_index=args.device,
            show_preview=not args.no_preview,
        )
        print(f"OK Recording saved:")
        print(f"   Letter:    {metadata.letter}")
        print(f"   Speaker:   {metadata.speaker_id}")
        print(f"   Session:   {metadata.session_id}")
        print(f"   Recording: {metadata.recording_id}")
        print(f"   Frames:    {metadata.frame_count} ({metadata.duration_ms:.0f}ms)")
        print(f"   data_source: {metadata.data_source}")
        print(f"   Source:    {metadata.source}")
        return 0
    except RuntimeError as e:
        print(f"FAIL Recording failed: {e}")
        return 1
    except ValueError as e:
        print(f"FAIL Invalid input: {e}")
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="alphabet",
        description="Alphabet Visual Speech Dataset Tools (Phase 2)"
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s 0.2.0 (Phase 2)"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # validate-dataset
    validate_parser = subparsers.add_parser(
        "validate-dataset",
        help="Validate dataset integrity"
    )
    validate_parser.add_argument("dataset_root", help="Path to dataset root directory")
    validate_parser.set_defaults(func=validate_cmd)

    # inspect-dataset
    inspect_parser = subparsers.add_parser(
        "inspect-dataset",
        help="Inspect dataset statistics"
    )
    inspect_parser.add_argument("dataset_root", help="Path to dataset root directory")
    inspect_parser.set_defaults(func=inspect_cmd)

    # record - primary production recording command
    record_parser = subparsers.add_parser(
        "record",
        help="Record real webcam video to the production dataset (data_source=real_camera)"
    )
    record_parser.add_argument(
        "--dataset-root", default="alphabet_dataset",
        help="Dataset root directory [default: alphabet_dataset]"
    )
    record_parser.add_argument(
        "letter", help="Letter label (A-Z)"
    )
    record_parser.add_argument(
        "--speaker-id", required=True,
        help="Speaker identifier (e.g. speaker_001)"
    )
    record_parser.add_argument(
        "--session-id", required=True,
        help="Session identifier (e.g. session_001)"
    )
    record_parser.add_argument(
        "--recording-id", help="Custom recording ID [default: auto UUID]"
    )
    record_parser.add_argument(
        "--duration", type=float, default=3.0,
        help="Recording duration in seconds [default: 3.0]"
    )
    record_parser.add_argument(
        "--fps", type=int, default=30, help="Target FPS [default: 30]"
    )
    record_parser.add_argument(
        "--width", type=int, default=1280, help="Video width [default: 1280]"
    )
    record_parser.add_argument(
        "--height", type=int, default=720, help="Video height [default: 720]"
    )
    record_parser.add_argument(
        "--device", type=int, default=0, help="Webcam device index [default: 0]"
    )
    record_parser.add_argument(
        "--no-preview", action="store_true",
        help="Disable preview window during recording"
    )
    record_parser.add_argument(
        "--lighting", help="Lighting condition tag (e.g. natural, fluorescent, dim)"
    )
    record_parser.add_argument(
        "--accent", help="Accent/language tag (e.g. american, british)"
    )
    record_parser.add_argument(
        "--notes", help="Additional notes"
    )
    record_parser.set_defaults(func=record_cmd)

    # ingest
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest video into dataset"
    )
    ingest_parser.add_argument("dataset_root", help="Path to dataset root directory")
    ingest_parser.add_argument("letter", help="Letter label (A-Z)")
    ingest_parser.add_argument("speaker", help="Speaker ID")
    ingest_parser.add_argument("session", help="Session ID")
    ingest_parser.add_argument(
        "--source", choices=["webcam", "local_mp4"], default="webcam",
        help="Input source type"
    )
    ingest_parser.add_argument("--video-path", help="Path to MP4 file (for local_mp4)")
    ingest_parser.add_argument("--recording-id", help="Custom recording ID")
    ingest_parser.add_argument("--duration", type=float, default=3.0,
                               help="Recording duration in seconds (webcam)")
    ingest_parser.add_argument("--lighting", help="Lighting condition tag")
    ingest_parser.add_argument("--accent", help="Accent/language tag")
    ingest_parser.add_argument("--notes", help="Additional notes")
    ingest_parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    ingest_parser.add_argument("--max-len", type=int, default=76, help="Max sequence length")
    ingest_parser.add_argument("--min-len", type=int, default=15, help="Min sequence length")
    ingest_parser.add_argument("--width", type=int, default=1280, help="Video width")
    ingest_parser.add_argument("--height", type=int, default=720, help="Video height")
    ingest_parser.add_argument("--device", type=int, default=0, help="Webcam device index")
    ingest_parser.add_argument("--preview", action="store_true", help="Show preview window")
    ingest_parser.add_argument("--version", default="1.0", help="Preprocessing version")
    ingest_parser.set_defaults(func=ingest_cmd)

    # preprocess
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Preprocess dataset (tracking + representations + manifest + splits)"
    )
    preprocess_parser.add_argument("dataset_root", help="Path to dataset root directory")
    preprocess_parser.add_argument(
        "--config", default="configs/alphabet_dataset.yaml",
        help="Path to YAML configuration file"
    )
    preprocess_parser.add_argument("--fps", type=int, default=None,
                                   help="Override video.fps from YAML")
    preprocess_parser.add_argument("--crop-margin", type=float, default=None,
                                   help="Override mouth_alignment.crop_margin_multiplier")
    preprocess_parser.add_argument("--no-resume", action="store_true",
                                   help="Disable resume (reprocess everything)")
    preprocess_parser.add_argument("--force-all", action="store_true",
                                   help="Force reprocessing of all recordings")
    preprocess_parser.set_defaults(func=preprocess_cmd)

    # cache-reprs (Phase 5)
    cache_parser = subparsers.add_parser(
        "cache-reprs",
        help="Build the Phase 3 fused-representation cache for decoder training"
    )
    cache_parser.add_argument("dataset_root", help="Path to dataset root directory")
    cache_parser.add_argument("--config", default="configs/alphabet_dataset.yaml",
                              help="Path to YAML configuration file")
    cache_parser.add_argument("--ablation", default="all",
                              help="Stream ablation mode (Phase 3 switches)")
    cache_parser.add_argument("--cache-dir", default=None,
                              help="Override default runs/repr_cache/<ablation>")
    cache_parser.add_argument("--limit", type=int, default=None,
                              help="Only cache the first N manifest entries")
    cache_parser.add_argument("--splits", default=None,
                              help="Comma-separated splits to cache (default all)")
    cache_parser.set_defaults(func=cache_reprs_cmd)

    # train-decoder (Phase 5)
    train_parser = subparsers.add_parser(
        "train-decoder",
        help="Train the letter-CTC sequence decoder"
    )
    train_parser.add_argument("--config", default="configs/alphabet_dataset.yaml",
                              help="Path to YAML configuration file")
    train_parser.add_argument("--max-epochs", type=int, default=None)
    train_parser.add_argument("--max-steps", type=int, default=None,
                              help="Hard optimizer-step cap (debugging)")
    train_parser.add_argument("--checkpoint-dir", default=None)
    train_parser.add_argument("--ablation", default=None,
                              help="Stream ablation mode (Phase 3 switches)")
    train_parser.add_argument("--limit-train", type=int, default=None)
    train_parser.add_argument("--limit-val", type=int, default=None)
    train_parser.add_argument("--device", default=None,
                              help="e.g. cuda / cpu (default: auto)")
    train_parser.add_argument("--no-resume", action="store_true")
    train_parser.add_argument("--run-test", action="store_true",
                              help="Evaluate the test split AFTER training "
                                   "(test stays untouched until then)")
    train_parser.set_defaults(func=train_decoder_cmd)

    # sanity-decoder (Phase 5, §25)
    sanity_parser = subparsers.add_parser(
        "sanity-decoder",
        help="Real-data sanity: Phase 2 -> 3 (cache) -> 4 -> decoder"
    )
    sanity_parser.add_argument("--config", default="configs/alphabet_dataset.yaml")
    sanity_parser.add_argument("--n-max", type=int, default=8)
    sanity_parser.add_argument("--device", default=None)
    sanity_parser.set_defaults(func=sanity_decoder_cmd)

    # overfit-decoder (Phase 5, §26)
    overfit_parser = subparsers.add_parser(
        "overfit-decoder",
        help="Tiny-subset overfit: verifies data->target->model->loss wiring"
    )
    overfit_parser.add_argument("--config", default="configs/alphabet_dataset.yaml")
    overfit_parser.add_argument("--steps", type=int, default=300)
    overfit_parser.add_argument("--device", default=None)
    overfit_parser.set_defaults(func=overfit_decoder_cmd)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())