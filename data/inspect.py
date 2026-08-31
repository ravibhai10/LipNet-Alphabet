"""
Dataset inspection command.

Implements: python -m alphabet inspect-dataset

Outputs concise terminal information:
- Total recordings
- Total speakers
- Total sessions
- Recordings per letter
- Frames per recording
- Sequence length distribution
- Missing-face percentage
- Missing-mouth percentage
- Quality statistics
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter
import logging

from data.schema import RecordingMetadata, ManifestEntry
from data.manifest import ManifestManager, ManifestConfig

logger = logging.getLogger(__name__)


class DatasetInspector:
    """Inspect dataset and print statistics."""

    def __init__(self, dataset_root: Path):
        self.dataset_root = Path(dataset_root)
        self.manifest_manager = ManifestManager(ManifestConfig(
            dataset_root=dataset_root,
            manifest_path=dataset_root / "manifest.jsonl",
        ))

    def inspect(self) -> Dict[str, Any]:
        """Run inspection and return statistics."""
        entries = self.manifest_manager.load_manifest()

        if not entries:
            return {"error": "No manifest entries found"}

        # Basic counts
        total_recordings = len(entries)
        speakers = set(e.speaker_id for e in entries)
        sessions = set(f"{e.speaker_id}/{e.session_id}" for e in entries)

        # Recordings per letter
        letter_counts = Counter(e.letter for e in entries)

        # Frames per recording
        frame_counts = [e.frame_count for e in entries]
        valid_frame_counts = [e.valid_frame_count for e in entries]

        # Sequence length distribution
        length_dist = self._compute_length_distribution(frame_counts)

        # Missing face/mouth percentages
        total_frames = sum(frame_counts)
        total_valid_frames = sum(valid_frame_counts)
        missing_face_pct = ((total_frames - total_valid_frames) / total_frames * 100) if total_frames > 0 else 0

        # Quality statistics from metadata
        quality_stats = self._compute_quality_stats(entries)

        # Split distribution
        split_counts = Counter(e.split or "unassigned" for e in entries)

        # Duration statistics
        durations = [e.duration_ms for e in entries]

        stats = {
            "total_recordings": total_recordings,
            "total_speakers": len(speakers),
            "total_sessions": len(sessions),
            "recordings_per_letter": dict(letter_counts),
            "frames_per_recording": {
                "mean": sum(frame_counts) / len(frame_counts) if frame_counts else 0,
                "min": min(frame_counts) if frame_counts else 0,
                "max": max(frame_counts) if frame_counts else 0,
                "median": self._median(frame_counts),
            },
            "valid_frames_per_recording": {
                "mean": sum(valid_frame_counts) / len(valid_frame_counts) if valid_frame_counts else 0,
                "min": min(valid_frame_counts) if valid_frame_counts else 0,
                "max": max(valid_frame_counts) if valid_frame_counts else 0,
                "median": self._median(valid_frame_counts),
            },
            "sequence_length_distribution": length_dist,
            "missing_face_percentage": round(missing_face_pct, 2),
            "missing_mouth_percentage": round(missing_face_pct, 2),  # Approximation
            "quality_statistics": quality_stats,
            "split_distribution": dict(split_counts),
            "duration_statistics": {
                "mean_ms": sum(durations) / len(durations) if durations else 0,
                "min_ms": min(durations) if durations else 0,
                "max_ms": max(durations) if durations else 0,
                "total_ms": sum(durations),
            },
        }

        return stats

    def _compute_length_distribution(self, frame_counts: List[int]) -> Dict[str, int]:
        """Compute sequence length distribution in buckets."""
        buckets = {
            "1-10": 0,
            "11-20": 0,
            "21-30": 0,
            "31-40": 0,
            "41-50": 0,
            "51-60": 0,
            "61-76": 0,
            "76+": 0,
        }

        for fc in frame_counts:
            if fc <= 10:
                buckets["1-10"] += 1
            elif fc <= 20:
                buckets["11-20"] += 1
            elif fc <= 30:
                buckets["21-30"] += 1
            elif fc <= 40:
                buckets["31-40"] += 1
            elif fc <= 50:
                buckets["41-50"] += 1
            elif fc <= 60:
                buckets["51-60"] += 1
            elif fc <= 76:
                buckets["61-76"] += 1
            else:
                buckets["76+"] += 1

        return buckets

    def _compute_quality_stats(self, entries: List[ManifestEntry]) -> Dict[str, Any]:
        """Compute quality statistics from manifest entries."""
        brightness_vals = []
        blur_vals = []
        mouth_size_vals = []
        pose_yaw_vals = []
        pose_pitch_vals = []
        pose_roll_vals = []
        duplicate_counts = []
        flag_counts = Counter()

        for entry in entries:
            qs = entry.quality_summary
            brightness_vals.append(qs.get("mean_brightness", 0))
            blur_vals.append(qs.get("mean_blur", 0))
            mouth_size_vals.append(qs.get("mean_mouth_size", 0))
            pose_yaw_vals.append(qs.get("mean_pose_yaw", 0))
            pose_pitch_vals.append(qs.get("mean_pose_pitch", 0))
            pose_roll_vals.append(qs.get("mean_pose_roll", 0))
            duplicate_counts.append(qs.get("duplicate_frame_count", 0))

            for flag in qs.get("flags", []):
                flag_counts[flag] += 1

        def stats(vals: List[float]) -> Dict[str, float]:
            if not vals:
                return {"mean": 0, "min": 0, "max": 0, "std": 0}
            import numpy as np
            arr = np.array(vals)
            return {
                "mean": float(arr.mean()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "std": float(arr.std()),
            }

        return {
            "brightness": stats(brightness_vals),
            "blur": stats(blur_vals),
            "mouth_size": stats(mouth_size_vals),
            "pose_yaw": stats(pose_yaw_vals),
            "pose_pitch": stats(pose_pitch_vals),
            "pose_roll": stats(pose_roll_vals),
            "duplicate_frames": stats(duplicate_counts),
            "quality_flags": dict(flag_counts),
        }

    def _median(self, values: List[float]) -> float:
        """Compute median."""
        if not values:
            return 0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        if n % 2 == 0:
            return (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2
        return sorted_vals[n//2]

    def print_report(self, stats: Dict[str, Any]) -> None:
        """Print formatted inspection report."""
        if "error" in stats:
            print(f"Error: {stats['error']}")
            return

        print("=" * 60)
        print("ALPHABET DATASET INSPECTION REPORT")
        print("=" * 60)

        print(f"\n📊 BASIC STATISTICS")
        print(f"  Total recordings:     {stats['total_recordings']}")
        print(f"  Total speakers:       {stats['total_speakers']}")
        print(f"  Total sessions:       {stats['total_sessions']}")

        print(f"\n🔤 RECORDINGS PER LETTER")
        for letter in sorted(stats['recordings_per_letter'].keys()):
            count = stats['recordings_per_letter'][letter]
            print(f"  {letter}: {count}")

        print(f"\n🎞️  FRAMES PER RECORDING")
        fc = stats['frames_per_recording']
        vfc = stats['valid_frames_per_recording']
        print(f"  All frames:     mean={fc['mean']:.1f}, min={fc['min']}, max={fc['max']}, median={fc['median']:.1f}")
        print(f"  Valid frames:   mean={vfc['mean']:.1f}, min={vfc['min']}, max={vfc['max']}, median={vfc['median']:.1f}")

        print(f"\n📏 SEQUENCE LENGTH DISTRIBUTION")
        for bucket, count in stats['sequence_length_distribution'].items():
            bar = "█" * min(count // 5, 40)  # Scale bar
            print(f"  {bucket:>6}: {count:4d} {bar}")

        print(f"\n❓ MISSING DATA")
        print(f"  Missing face:    {stats['missing_face_percentage']:.2f}%")
        print(f"  Missing mouth:   {stats['missing_mouth_percentage']:.2f}%")

        print(f"\n🎯 QUALITY STATISTICS")
        qs = stats['quality_statistics']
        print(f"  Brightness:  mean={qs['brightness']['mean']:.1f}, range=[{qs['brightness']['min']:.1f}, {qs['brightness']['max']:.1f}], std={qs['brightness']['std']:.1f}")
        print(f"  Blur:        mean={qs['blur']['mean']:.1f}, range=[{qs['blur']['min']:.1f}, {qs['blur']['max']:.1f}], std={qs['blur']['std']:.1f}")
        print(f"  Mouth size:  mean={qs['mouth_size']['mean']:.1f}, range=[{qs['mouth_size']['min']:.1f}, {qs['mouth_size']['max']:.1f}], std={qs['mouth_size']['std']:.1f}")
        print(f"  Pose yaw:    mean={qs['pose_yaw']['mean']:.2f}°, std={qs['pose_yaw']['std']:.2f}°")
        print(f"  Pose pitch:  mean={qs['pose_pitch']['mean']:.2f}°, std={qs['pose_pitch']['std']:.2f}°")
        print(f"  Pose roll:   mean={qs['pose_roll']['mean']:.2f}°, std={qs['pose_roll']['std']:.2f}°")
        print(f"  Duplicates:  mean={qs['duplicate_frames']['mean']:.1f}, max={qs['duplicate_frames']['max']:.0f}")

        if qs['quality_flags']:
            print(f"  Quality flags:")
            for flag, count in sorted(qs['quality_flags'].items()):
                print(f"    {flag}: {count}")

        print(f"\n✂️  SPLIT DISTRIBUTION")
        for split, count in sorted(stats['split_distribution'].items()):
            pct = count / stats['total_recordings'] * 100
            print(f"  {split}: {count} ({pct:.1f}%)")

        print(f"\n⏱️  DURATION STATISTICS")
        ds = stats['duration_statistics']
        print(f"  Mean:   {ds['mean_ms']:.0f} ms")
        print(f"  Min:    {ds['min_ms']:.0f} ms")
        print(f"  Max:    {ds['max_ms']:.0f} ms")
        print(f"  Total:  {ds['total_ms'] / 1000 / 60:.1f} minutes")

        print("\n" + "=" * 60)


def inspect_dataset(dataset_root: str) -> int:
    """
    Main inspection entry point.

    Args:
        dataset_root: Path to dataset root directory

    Returns:
        Exit code (0 = success)
    """
    logging.basicConfig(level=logging.WARNING)

    inspector = DatasetInspector(Path(dataset_root))
    stats = inspector.inspect()
    inspector.print_report(stats)

    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m data.inspect <dataset_root>")
        sys.exit(1)

    sys.exit(inspect_dataset(sys.argv[1]))