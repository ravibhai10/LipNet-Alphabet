"""
Dataset validation command.

Implements: python -m alphabet validate-dataset

Checks:
- missing files
- corrupted videos
- invalid metadata
- missing frames
- missing landmarks
- timestamp ordering
- duplicate recording IDs
- duplicate speaker/session IDs
- class labels outside A-Z
- train/test leakage
- inconsistent preprocessing versions
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass
import logging
import json

import numpy as np

from data.schema import (
    RecordingMetadata, ManifestEntry, Letter, Split,
    validate_metadata, validate_manifest_entry, validate_frame_ordering,
    QualityFlag,
    LANDMARK_SOURCE_MEDIAPIPE
)
from data.manifest import ManifestManager, ManifestConfig
from data.ingestion import get_video_info, validate_video

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    check_name: str
    passed: bool
    message: str
    severity: str  # "error", "warning", "info"
    details: List[str] = None

    def __post_init__(self):
        if self.details is None:
            self.details = []


@dataclass
class ValidationReport:
    """Complete validation report."""
    results: List[ValidationResult]
    critical_errors: int = 0
    warnings: int = 0

    def add_result(self, result: ValidationResult) -> None:
        self.results.append(result)
        if result.severity == "error":
            self.critical_errors += 1
        elif result.severity == "warning":
            self.warnings += 1

    def has_critical_errors(self) -> bool:
        return self.critical_errors > 0

    def summary(self) -> str:
        lines = [
            f"Validation Report",
            f"=================",
            f"Total checks: {len(self.results)}",
            f"Critical errors: {self.critical_errors}",
            f"Warnings: {self.warnings}",
            f"",
        ]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{status}] {r.check_name}: {r.message}")
            if r.details:
                for d in r.details:
                    lines.append(f"    - {d}")
        return "\n".join(lines)


class DatasetValidator:
    """Validate alphabet dataset integrity."""

    def __init__(self, dataset_root: Path):
        self.dataset_root = Path(dataset_root)
        self.manifest_path = self.dataset_root / "manifest.jsonl"
        self.recordings_root = self.dataset_root / "recordings"

    def validate_all(self) -> ValidationReport:
        """Run all validation checks."""
        report = ValidationReport(results=[])

        # Run checks in order
        report.add_result(self._check_manifest_exists())
        report.add_result(self._check_recordings_directory())

        # Load manifest for further checks
        entries = self._load_manifest()
        if entries:
            report.add_result(self._check_manifest_entries(entries))
            report.add_result(self._check_duplicate_recording_ids(entries))
            report.add_result(self._check_duplicate_speaker_sessions(entries))
            report.add_result(self._check_letters(entries))
            report.add_result(self._check_missing_files(entries))
            report.add_result(self._check_video_integrity(entries))
            report.add_result(self._check_metadata_validity(entries))
            report.add_result(self._check_frame_ordering(entries))
            report.add_result(self._check_landmarks_exist(entries))
            report.add_result(self._check_splits(entries))
            report.add_result(self._check_preprocessing_versions(entries))
            report.add_result(self._check_train_test_leakage(entries))
            report.add_result(self._check_landmark_source(entries))
            report.add_result(self._check_data_source(entries))  # HARD DATA GATE (§6)
            report.add_result(self._check_motion_content(entries))

        return report

    def _load_manifest(self) -> List[ManifestEntry]:
        """Load manifest entries."""
        if not self.manifest_path.exists():
            return []

        manager = ManifestManager(ManifestConfig(
            dataset_root=self.dataset_root,
            manifest_path=self.manifest_path,
        ))
        return manager.load_manifest()

    def _check_manifest_exists(self) -> ValidationResult:
        """Check if manifest.jsonl exists."""
        if self.manifest_path.exists():
            return ValidationResult(
                check_name="manifest_exists",
                passed=True,
                message="Manifest file exists",
                severity="info"
            )
        return ValidationResult(
            check_name="manifest_exists",
            passed=False,
            message="Manifest file missing: manifest.jsonl",
            severity="error"
        )

    def _check_recordings_directory(self) -> ValidationResult:
        """Check if recordings directory exists."""
        if self.recordings_root.exists():
            return ValidationResult(
                check_name="recordings_directory",
                passed=True,
                message="Recordings directory exists",
                severity="info"
            )
        return ValidationResult(
            check_name="recordings_directory",
            passed=False,
            message="Recordings directory missing",
            severity="error"
        )

    def _check_manifest_entries(self, entries: List[ManifestEntry]) -> ValidationResult:
        """Validate each manifest entry."""
        errors = []
        for entry in entries:
            entry_errors = validate_manifest_entry(entry)
            if entry_errors:
                errors.append(f"{entry.recording_id}: {', '.join(entry_errors)}")

        if errors:
            return ValidationResult(
                check_name="manifest_entries",
                passed=False,
                message=f"{len(errors)} manifest entries have validation errors",
                severity="error",
                details=errors[:10]  # Show first 10
            )
        return ValidationResult(
            check_name="manifest_entries",
            passed=True,
            message=f"All {len(entries)} manifest entries valid",
            severity="info"
        )

    def _check_duplicate_recording_ids(self, entries: List[ManifestEntry]) -> ValidationResult:
        """Check for duplicate recordings within the same (speaker, session, letter).

        ``recording_id`` is scoped to its session directory (per DESIGN.md the
        layout is ``<LETTER>/<speaker>/<session>/<recording>``), so the unique
        identity is the scoped tuple, not ``recording_id`` alone. Two manifest
        entries describing the same physical recording are flagged here.
        """
        seen = {}
        duplicates = []

        for entry in entries:
            key = (entry.letter, entry.speaker_id, entry.session_id, entry.recording_id)
            if key in seen:
                duplicates.append(
                    f"{entry.recording_id} ({key[:3]}): "
                    f"{seen[key]} and {entry.path}"
                )
            else:
                seen[key] = entry.path

        if duplicates:
            return ValidationResult(
                check_name="duplicate_recording_ids",
                passed=False,
                message=f"Found {len(duplicates)} duplicate recordings",
                severity="error",
                details=duplicates[:10]
            )
        return ValidationResult(
            check_name="duplicate_recording_ids",
            passed=True,
            message="No duplicate recordings (scoped by letter/speaker/session)",
            severity="info"
        )

    def _check_duplicate_speaker_sessions(self, entries: List[ManifestEntry]) -> ValidationResult:
        """Check for duplicate speaker/session/letter/recording combinations."""
        seen = {}
        duplicates = []

        for entry in entries:
            key = (entry.speaker_id, entry.session_id, entry.letter, entry.recording_id)
            if key in seen:
                duplicates.append(
                    f"{key}: {seen[key]} and {entry.path}"
                )
            else:
                seen[key] = entry.path

        if duplicates:
            return ValidationResult(
                check_name="duplicate_speaker_sessions",
                passed=False,
                message=f"Found {len(duplicates)} duplicate speaker/session/letter/recording combinations",
                severity="warning",
                details=duplicates[:10]
            )
        return ValidationResult(
            check_name="duplicate_speaker_sessions",
            passed=True,
            message="No duplicate speaker/session/letter/recording combinations",
            severity="info"
        )

    def _check_letters(self, entries: List[ManifestEntry]) -> ValidationResult:
        """Check all letters are valid A-Z."""
        invalid = []
        for entry in entries:
            if not Letter.is_valid(entry.letter):
                invalid.append(f"{entry.recording_id}: '{entry.letter}'")

        if invalid:
            return ValidationResult(
                check_name="valid_letters",
                passed=False,
                message=f"Found {len(invalid)} recordings with invalid letters",
                severity="error",
                details=invalid[:10]
            )
        return ValidationResult(
            check_name="valid_letters",
            passed=True,
            message="All letters are valid A-Z",
            severity="info"
        )

    def _check_missing_files(self, entries: List[ManifestEntry]) -> ValidationResult:
        """Check for missing required files."""
        missing = []

        for entry in entries:
            rec_path = self.dataset_root / entry.path

            # Check directory
            if not rec_path.exists():
                missing.append(f"{entry.recording_id}: directory missing")
                continue

            # Check video
            video_path = rec_path / "video.mp4"
            if not video_path.exists():
                missing.append(f"{entry.recording_id}: video.mp4 missing")

            # Check metadata
            metadata_path = rec_path / "metadata.json"
            if not metadata_path.exists():
                missing.append(f"{entry.recording_id}: metadata.json missing")

        if missing:
            return ValidationResult(
                check_name="missing_files",
                passed=False,
                message=f"Found {len(missing)} recordings with missing files",
                severity="error",
                details=missing[:20]
            )
        return ValidationResult(
            check_name="missing_files",
            passed=True,
            message="All required files present",
            severity="info"
        )

    def _check_video_integrity(self, entries: List[ManifestEntry]) -> ValidationResult:
        """Check video files can be decoded."""
        corrupted = []

        for entry in entries:
            video_path = self.dataset_root / entry.path / "video.mp4"
            if not video_path.exists():
                continue

            is_valid, error = validate_video(video_path)
            if not is_valid:
                corrupted.append(f"{entry.recording_id}: {error}")

        if corrupted:
            return ValidationResult(
                check_name="video_integrity",
                passed=False,
                message=f"Found {len(corrupted)} corrupted/invalid videos",
                severity="error",
                details=corrupted[:10]
            )
        return ValidationResult(
            check_name="video_integrity",
            passed=True,
            message="All videos are decodable",
            severity="info"
        )

    def _check_metadata_validity(self, entries: List[ManifestEntry]) -> ValidationResult:
        """Validate metadata.json for each recording."""
        invalid = []

        for entry in entries:
            metadata_path = self.dataset_root / entry.path / "metadata.json"
            if not metadata_path.exists():
                continue

            try:
                metadata = RecordingMetadata.load(metadata_path)
                errors = validate_metadata(metadata)
                if errors:
                    invalid.append(f"{entry.recording_id}: {', '.join(errors)}")
            except Exception as e:
                invalid.append(f"{entry.recording_id}: Failed to load metadata: {e}")

        if invalid:
            return ValidationResult(
                check_name="metadata_validity",
                passed=False,
                message=f"Found {len(invalid)} recordings with invalid metadata",
                severity="error",
                details=invalid[:10]
            )
        return ValidationResult(
            check_name="metadata_validity",
            passed=True,
            message="All metadata valid",
            severity="info"
        )

    def _check_frame_ordering(self, entries: List[ManifestEntry]) -> ValidationResult:
        """Check frame timestamp ordering."""
        issues = []

        for entry in entries:
            frames_path = self.dataset_root / entry.path / "frames" / "frame_data.json"
            if not frames_path.exists():
                continue

            try:
                with open(frames_path) as f:
                    frame_data = json.load(f)

                # Convert to FrameData objects for validation
                from data.schema import FrameData
                frames = [FrameData.from_dict(fd) for fd in frame_data]

                order_errors = validate_frame_ordering(frames)
                if order_errors:
                    issues.append(f"{entry.recording_id}: {', '.join(order_errors)}")
            except Exception as e:
                issues.append(f"{entry.recording_id}: Failed to validate frame ordering: {e}")

        if issues:
            return ValidationResult(
                check_name="frame_ordering",
                passed=False,
                message=f"Found {len(issues)} recordings with frame ordering issues",
                severity="error",
                details=issues[:10]
            )
        return ValidationResult(
            check_name="frame_ordering",
            passed=True,
            message="All frame sequences chronologically ordered",
            severity="info"
        )

    def _check_landmarks_exist(self, entries: List[ManifestEntry]) -> ValidationResult:
        """Check that landmark files exist for processed recordings."""
        missing = []

        for entry in entries:
            landmarks_dir = self.dataset_root / entry.path / "landmarks"
            if not landmarks_dir.exists():
                missing.append(f"{entry.recording_id}: landmarks directory missing")
                continue

            required = [
                "mouth_landmarks.npz",
                "mouth_bboxes.npz",
                "head_poses.npz",
            ]

            for req in required:
                if not (landmarks_dir / req).exists():
                    missing.append(f"{entry.recording_id}: {req} missing")

        if missing:
            return ValidationResult(
                check_name="landmarks_exist",
                passed=False,
                message=f"Found {len(missing)} missing landmark files",
                severity="warning",
                details=missing[:20]
            )
        return ValidationResult(
            check_name="landmarks_exist",
            passed=True,
            message="All landmark files present",
            severity="info"
        )

    def _check_splits(self, entries: List[ManifestEntry]) -> ValidationResult:
        """Check split assignments are valid."""
        issues = []

        for entry in entries:
            if entry.split and entry.split not in [s.value for s in Split]:
                issues.append(f"{entry.recording_id}: invalid split '{entry.split}'")

        if issues:
            return ValidationResult(
                check_name="valid_splits",
                passed=False,
                message=f"Found {len(issues)} recordings with invalid splits",
                severity="error",
                details=issues[:10]
            )
        return ValidationResult(
            check_name="valid_splits",
            passed=True,
            message="All splits valid",
            severity="info"
        )

    def _check_preprocessing_versions(self, entries: List[ManifestEntry]) -> ValidationResult:
        """Check preprocessing version consistency."""
        versions = set()
        for entry in entries:
            versions.add(entry.preprocessing_version)

        if len(versions) > 1:
            return ValidationResult(
                check_name="preprocessing_versions",
                passed=False,
                message=f"Multiple preprocessing versions found: {sorted(versions)}",
                severity="warning",
                details=[f"Version: {v}" for v in sorted(versions)]
            )
        return ValidationResult(
            check_name="preprocessing_versions",
            passed=True,
            message=f"Single preprocessing version: {next(iter(versions)) if versions else 'none'}",
            severity="info"
        )

    def _check_train_test_leakage(self, entries: List[ManifestEntry]) -> ValidationResult:
        """Check for train/test leakage (same speaker in multiple splits)."""
        speaker_splits = {}
        leaks = []

        for entry in entries:
            if not entry.split:
                continue

            if entry.speaker_id in speaker_splits:
                if speaker_splits[entry.speaker_id] != entry.split:
                    leaks.append(
                        f"Speaker {entry.speaker_id} in both "
                        f"{speaker_splits[entry.speaker_id]} and {entry.split}"
                    )
            else:
                speaker_splits[entry.speaker_id] = entry.split

        if leaks:
            return ValidationResult(
                check_name="train_test_leakage",
                passed=False,
                message=f"Found {len(leaks)} speaker leakage cases",
                severity="error",
                details=leaks[:10]
            )
        return ValidationResult(
            check_name="train_test_leakage",
            passed=True,
            message="No train/test leakage detected",
            severity="info"
        )

    def _check_landmark_source(self, entries: List[ManifestEntry]) -> ValidationResult:
        """Reject synthetic or missing landmark provenance in production data."""
        bad = []

        for entry in entries:
            source = getattr(entry, "landmark_source", None)
            if not source:
                bad.append(f"{entry.recording_id}: landmark_source missing")
            elif source != LANDMARK_SOURCE_MEDIAPIPE:
                bad.append(
                    f"{entry.recording_id}: forbidden landmark_source '{source}'"
                )
                continue

            # Cross-check the recording metadata as well.
            metadata_path = self.dataset_root / entry.path / "metadata.json"
            if metadata_path.exists():
                try:
                    metadata = RecordingMetadata.load(metadata_path)
                    if metadata.landmark_source != LANDMARK_SOURCE_MEDIAPIPE:
                        bad.append(
                            f"{entry.recording_id}: metadata declares "
                            f"landmark_source '{metadata.landmark_source}'"
                        )
                except Exception as e:  # noqa: BLE001
                    bad.append(f"{entry.recording_id}: cannot read metadata: {e}")

        if bad:
            return ValidationResult(
                check_name="landmark_source",
                passed=False,
                message=(
                    "REAL_FACE_LANDMARKS_UNAVAILABLE / synthetic landmarks "
                    f"in production dataset ({len(bad)} recordings)"
                ),
                severity="error",
                details=bad[:10],
            )
        return ValidationResult(
            check_name="landmark_source",
            passed=True,
            message=f"All {len(entries)} recordings use real MediaPipe landmarks",
            severity="info",
        )

    def _check_data_source(self, entries: List[ManifestEntry]) -> ValidationResult:
        """REJECT synthetic/test data in production dataset (§6 DATA GATE).

        Each manifest entry MUST have an explicit ``data_source``. The previous
        implementation defaulted missing values to ``"real_camera"`` (via
        ``getattr(..., DATA_SOURCE_REAL_CAMERA)``), which silently accepted
        synthetic recordings whose metadata.json had no ``data_source`` field.
        This version requires every entry to have an explicit, real value.
        """
        from data.schema import (
            DATA_SOURCE_REAL_CAMERA, DATA_SOURCE_SYNTHETIC_TEST,
            DATA_SOURCE_UNSPECIFIED, REAL_DATA_ONLY_SOURCES,
            TEST_SYNTHETIC_SOURCES,
        )
        bad = []
        for entry in entries:
            # Read from entry; default sentinel __UNSPECIFIED__ will fail.
            src = getattr(entry, "data_source", DATA_SOURCE_UNSPECIFIED)
            if src == DATA_SOURCE_UNSPECIFIED or src is None or src == "":
                bad.append(
                    f"{entry.recording_id}: data_source MISSING "
                    f"(must be 'real_camera' for production)"
                )
            elif src in TEST_SYNTHETIC_SOURCES:
                bad.append(
                    f"{entry.recording_id}: synthetic data_source '{src}' "
                    f"REJECTED in production dataset "
                    f"(use tests/fixtures/ or synthetic_dataset/ only)"
                )
            elif src not in REAL_DATA_ONLY_SOURCES:
                bad.append(
                    f"{entry.recording_id}: unknown data_source '{src}' "
                    f"(must be 'real_camera')"
                )
            # Cross-check metadata file directly
            metadata_path = self.dataset_root / entry.path / "metadata.json"
            if metadata_path.exists():
                try:
                    meta = json.load(open(metadata_path))
                    meta_src = meta.get("data_source", DATA_SOURCE_UNSPECIFIED)
                    if meta_src == DATA_SOURCE_UNSPECIFIED:
                        bad.append(
                            f"{entry.recording_id}: metadata.json has NO data_source field"
                        )
                    elif meta_src in TEST_SYNTHETIC_SOURCES:
                        bad.append(
                            f"{entry.recording_id}: metadata.json data_source "
                            f"'{meta_src}' is synthetic; production must be "
                            f"'{DATA_SOURCE_REAL_CAMERA}'"
                        )
                    elif meta_src not in REAL_DATA_ONLY_SOURCES:
                        bad.append(
                            f"{entry.recording_id}: metadata.json data_source "
                            f"'{meta_src}' is not 'real_camera'"
                        )
                except Exception:
                    pass  # can't read, other validators catch
        if bad:
            return ValidationResult(
                check_name="data_source",
                passed=False,
                message=(
                    "PRODUCTION DATA GATE FAILED "
                    f"({len(bad)} recordings have missing/non-real data_source)"
                ),
                severity="error",
                details=bad[:10],
            )
        return ValidationResult(
            check_name="data_source",
            passed=True,
            message=f"All {len(entries)} recordings use real_camera data_source",
            severity="info",
        )

    def _check_motion_content(self, entries: List[ManifestEntry]) -> ValidationResult:
        """Detect zero-motion sequences where motion should exist.

        A recording whose motion embeddings are exactly all zeros indicates a
        broken pipeline (e.g. simulated/constant landmarks). Legitimate static
        segments with LOW (but non-zero) motion are NOT rejected.
        """
        zero_motion = []
        checked = 0

        for entry in entries:
            motion_path = self.dataset_root / entry.path / "motion_embeddings.npz"
            if not motion_path.exists():
                continue

            try:
                data = np.load(motion_path)
                key = "embeddings" if "embeddings" in data.files else data.files[0]
                emb = np.asarray(data[key], dtype=np.float64)

                # Need at least 2 frames for any temporal variation to be
                # possible; single-frame sequences are skipped.
                if emb.shape[0] < 2:
                    continue

                checked += 1
                variation = float(np.abs(np.diff(emb, axis=0)).max())
                if variation == 0.0:
                    zero_motion.append(
                        f"{entry.recording_id}: motion embeddings are constant "
                        "(all-zero temporal variation)"
                    )
            except Exception as e:  # noqa: BLE001
                zero_motion.append(f"{entry.recording_id}: cannot read motion: {e}")

        if zero_motion:
            return ValidationResult(
                check_name="motion_content",
                passed=False,
                message=(
                    f"{len(zero_motion)} recordings have zero/constant motion "
                    "despite multiple frames — likely fake or broken landmarks"
                ),
                severity="warning",
                details=zero_motion[:10],
            )
        return ValidationResult(
            check_name="motion_content",
            passed=True,
            message=(
                f"Motion content plausible in {checked} checked recordings"
                if checked else "No multi-frame motion files to check"
            ),
            severity="info",
        )


def validate_dataset(dataset_root: str) -> int:
    """
    Main validation entry point.

    Args:
        dataset_root: Path to dataset root directory

    Returns:
        Exit code (0 = success, 1 = critical errors)
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    validator = DatasetValidator(Path(dataset_root))
    report = validator.validate_all()

    print(report.summary())

    if report.has_critical_errors():
        print("\n❌ VALIDATION FAILED: Critical errors found")
        return 1
    elif report.warnings > 0:
        print("\n⚠️  VALIDATION PASSED WITH WARNINGS")
        return 0
    else:
        print("\n✅ VALIDATION PASSED")
        return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m data.validate <dataset_root>")
        sys.exit(1)

    sys.exit(validate_dataset(sys.argv[1]))