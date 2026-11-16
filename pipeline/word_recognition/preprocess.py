"""Authentic annotation-bounded GRID words to lossless canonical mouth PNGs.

No letters, sentence labels, filename-derived classes, fabricated landmarks, or
silent fallback tensors. A failed candidate is logged, with a same-fold
replacement selected deterministically before any model is trained.
"""
from __future__ import annotations
from collections import defaultdict, deque
import hashlib
import math
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from scripts.prepare_grid_dataset import parse_alignment
from src.video_training import resample_temporal
from tracking.mouth_canonicalization import read_aligned_mouth_video
from .common import (ROOT, SPLITS, TASK, atomic_json, fingerprint, load_config, load_dataset,
                     load_json, parser, prepared_root, project_path, read_rows, run_contract, sha256, timestamp, validate_rows, write_rows)


def select_candidates(config):
    root = project_path(config["source_inventory"])
    info = load_json(root / "inventory.json")
    assignments = info["assignments"]
    groups = defaultdict(lambda: defaultdict(list))
    for source in read_rows(root / "video_inventory.jsonl"):
        if source["status"] != "valid":
            continue
        split = assignments[source["speaker"]]
        for word in source["transcript"].split():
            if word in config["words"]:
                groups[split, word][source["speaker"]].append(source)
    result = {}
    # Round-robin speakers so tiny quotas still involve independent people.
    seed = config["training"]["seed"]
    for split in SPLITS:
        for word in config["words"]:
            speakers = groups[split, word]
            if len(speakers) < config["minimum_speakers"][split]:
                raise ValueError(f"Insufficient real speaker coverage: {split}/{word}")
            queues = []
            for speaker, sources in sorted(speakers.items()):
                sources.sort(key=lambda row: fingerprint([seed, word, row["video"]]))
                queues.append(deque(sources))
            ordered = []
            while any(queues):
                for queue in queues:
                    if queue:
                        ordered.append(queue.popleft())
            if len(ordered) < config["quotas"][split]:
                raise ValueError(f"Insufficient real word samples: {split}/{word}")
            result[split, word] = ordered
    return result


def aligned_segment(video, start, end, preprocessing):
    crops, fps = read_aligned_mouth_video(video, max_frames=500, start_frame=start, end_frame=end)
    size = preprocessing["size"]
    resized = [cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR) for crop in crops]
    clip = np.rint(resample_temporal(resized, preprocessing["frames"])).clip(0, 255).astype(np.uint8)
    return clip, fps, len(crops)


def read_frames(run, row, config):
    directory = (prepared_root(config) / row["frames_path"]).resolve()
    if not directory.is_relative_to(prepared_root(config)):
        raise ValueError("Sample frame path escapes the run.")
    n, size = config["preprocessing"]["frames"], config["preprocessing"]["size"]
    paths = sorted(directory.glob("frame_*.png"))
    if [path.name for path in paths] != [f"frame_{i:04d}.png" for i in range(n)]:
        raise ValueError("Missing/extra canonical mouth frames.")
    frames = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in paths]
    if any(frame is None or frame.shape != (size, size) for frame in frames):
        raise ValueError("Cannot decode a canonical mouth PNG with expected dimensions.")
    clip = np.stack(frames)
    if hashlib.sha256(clip.tobytes()).hexdigest() != row["pixels_sha256"]:
        raise ValueError("Canonical mouth pixels changed after preparation.")
    return clip


def prepare(config, run):
    if (run / "dataset.json").exists():
        return load_dataset(run, config, verify_pixels=True)[0]
    acquisition = load_json(run / "acquisition.json")
    inventory = project_path(config["source_inventory"])
    if acquisition["inventory_sha256"] != sha256(inventory / "video_inventory.jsonl") or acquisition["inventory_summary_sha256"] != sha256(inventory / "inventory.json"):
        raise ValueError("Source inventory changed after acquisition verification.")
    candidates = select_candidates(config)
    estimate = sum(config["quotas"].values()) * len(config["words"]) * config["preprocessing"]["frames"] * (64 * 64 + 4096)
    prepared_root(config).mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(prepared_root(config)).free < config["reserve_gib"] * 2**30 + estimate:
        raise OSError("Insufficient free disk for projected PNGs plus reserve.")
    rows, failures, seen_pixels = [], [], set()
    source_root = project_path(config["corpus_root"])
    start_time = time.perf_counter()
    for split in SPLITS:
        for word in config["words"]:
            accepted = 0
            for source in candidates[split, word]:
                if accepted >= config["quotas"][split]:
                    break
                key = f"{source['speaker']}_{Path(source['video']).stem}_{word}"
                try:
                    video = (source_root / source["video"]).resolve()
                    annotation = (source_root / source["alignment"]).resolve()
                    if not video.is_relative_to(source_root) or not annotation.is_relative_to(source_root):
                        raise ValueError("Source paths escape corpus root.")
                    if sha256(video) != source["video_sha256"] or sha256(annotation) != source["alignment_sha256"]:
                        raise ValueError("Source video or annotation changed after full validation.")
                    tokens, _ = parse_alignment(annotation)
                    match = [token for token in tokens if token["word"] == word]
                    if len(match) != 1:
                        raise ValueError("Ambiguous or missing whole-word annotation.")
                    token = match[0]
                    fps = source["fps"]
                    first, last = math.floor(token["start_units"] / 25000 * fps), math.ceil(token["end_units"] / 25000 * fps)
                    if not 0 <= first < last <= source["decoded_frames"] or last - first < 2:
                        raise ValueError("Word interval is too short or outside the real video.")
                    relative = Path(source["speaker"]) / word / key
                    directory = prepared_root(config) / relative
                    marker = directory / "sample.json"
                    expected = {"source_sha256": source["video_sha256"], "annotation_sha256": source["alignment_sha256"],
                                "word": word, "start_frame": first, "end_frame": last, "config_sha256": fingerprint(config),
                                "landmarker_sha256": sha256(ROOT / "frontend/public/models/face_landmarker.task")}
                    if marker.exists():
                        saved = load_json(marker)
                        if saved["contract"] != expected:
                            raise ValueError("Existing canonical sample belongs to another source/contract.")
                        row = saved["row"]
                        read_frames(run, row, config)
                    else:
                        if directory.exists():
                            raise FileExistsError("Unverified sample directory is protected.")
                        clip, measured_fps, tracked = aligned_segment(video, first, last, config["preprocessing"])
                        if abs(measured_fps - fps) > .1:
                            raise ValueError("Source FPS changed.")
                        pixel_hash = hashlib.sha256(clip.tobytes()).hexdigest()
                        row = {"id": key, "speaker": source["speaker"], "split": split, "word": word,
                               "label": config["words"].index(word), "source_video": str(video), "annotation": str(annotation),
                               "source_sha256": source["video_sha256"], "annotation_sha256": source["alignment_sha256"],
                               "start_units": token["start_units"], "end_units": token["end_units"], "timebase_hz": 25000,
                               "start_frame": first, "end_frame": last, "fps": fps, "tracked_frames": tracked,
                               "frames_path": relative.as_posix(), "pixels_sha256": pixel_hash,
                               "data_source": "official_grid_video", "landmark_source": "mediapipe"}
                        if pixel_hash in seen_pixels:
                            raise ValueError("Duplicate mouth pixels; candidate excluded.")
                        temporary = directory.with_name(directory.name + ".partial")
                        temporary.mkdir(parents=True, exist_ok=True)
                        # Recover an interrupted write only when every existing PNG matches.
                        for i, frame in enumerate(clip):
                            target = temporary / f"frame_{i:04d}.png"
                            if target.exists():
                                existing = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
                                if existing is None or not np.array_equal(existing, frame):
                                    raise ValueError("Interrupted canonical sample differs; preserved.")
                            elif not cv2.imwrite(str(target), frame):
                                raise OSError("Cannot write lossless canonical frame.")
                        atomic_json(temporary / "sample.json", {"contract": expected, "row": row})
                        temporary.rename(directory)
                    if row["pixels_sha256"] in seen_pixels:
                        raise ValueError("Duplicate mouth pixels; candidate excluded.")
                    seen_pixels.add(row["pixels_sha256"])
                    rows.append(row)
                    accepted += 1
                except (ValueError, OSError, RuntimeError) as error:
                    if "REAL_FACE_LANDMARKS_UNAVAILABLE" in str(error) or "changed after full validation" in str(error):
                        raise
                    failures.append({"id": key, "split": split, "word": word, "reason": str(error)})
                if len(rows) and len(rows) % 20 == 0:
                    print(f"Prepared {len(rows)} genuine aligned word clips", flush=True)
            # Fail closed rather than silently changing requested class distribution.
            if accepted != config["quotas"][split]:
                write_rows(run / "excluded_candidates.jsonl", failures)
                raise ValueError(f"Failed real-data quota for {split}/{word}; see exclusion log.")
    summary = validate_rows(rows, config)
    write_rows(run / "words.jsonl", rows)
    write_rows(run / "excluded_candidates.jsonl", failures)
    info = {"task": TASK, "checked_at": timestamp(), "config_sha256": fingerprint(config),
            "manifest_sha256": sha256(run / "words.jsonl"), "vocabulary": config["words"],
            "preprocessing": config["preprocessing"], "landmarker_sha256": sha256(ROOT / "frontend/public/models/face_landmarker.task"),
            "protocol": "speaker-disjoint GRID word segments; same source utterance cannot cross partitions",
            "annotation_boundary_policy": "floor(start*fps/25000), exclusive ceil(end*fps/25000); coarticulation possible",
            "samples": len(rows), "excluded_candidates": len(failures), "summary": summary,
            "preprocessing_seconds": time.perf_counter() - start_time,
            "task_limits": "Fixed-vocabulary words segmented using ground-truth timing; not sentence recognition or automatic word boundary detection"}
    atomic_json(run / "dataset.json", info)
    return info


def main():
    args = parser(__doc__).parse_args()
    config = load_config(args.config)
    run = run_contract(config, args.run)
    print(prepare(config, run))


if __name__ == "__main__":
    main()
