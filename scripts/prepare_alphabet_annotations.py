"""Build an audited A-Z JSONL manifest from real recording metadata.

Speaker assignment is explicit; this command does not record, relabel, or train.
"""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.avhubert.classification import audit_records, sha256_file
from scripts.avhubert_classifier import validate_alphabet


def prepare(recordings, assignments, output):
    recordings, output = Path(recordings), Path(output)
    if output.exists():
        raise FileExistsError("Choose a new annotation manifest; existing files are protected.")
    if set(assignments) != {"train", "valid", "test"} or not all(assignments.values()):
        raise ValueError("Assign speakers to all three partitions.")
    speaker_split = {}
    for split, speakers in assignments.items():
        for speaker in speakers:
            if speaker in speaker_split:
                raise ValueError(f"Speaker assigned more than once: {speaker}")
            speaker_split[speaker] = split
    rows = []
    seen_speakers = set()
    for metadata_file in sorted((recordings / "recordings").glob("*/*/*/*/metadata.json")):
        metadata = json.loads(metadata_file.read_text(encoding="utf-8-sig"))
        speaker = metadata.get("speaker_id")
        if speaker not in speaker_split:
            raise ValueError(f"Recording speaker has no explicit partition: {speaker}")
        video = metadata_file.parent / "video.mp4"
        if metadata.get("data_source") != "real_camera" or not video.is_file():
            raise ValueError("Each recording needs real_camera provenance and a video file.")
        label = metadata.get("letter")
        identifier = "/".join(metadata_file.parent.relative_to(recordings / "recordings").parts)
        try:
            relative_video = os.path.relpath(video.resolve(), output.parent.resolve())
        except ValueError:  # Different Windows drives cannot form a relative path.
            relative_video = str(video.resolve())
        rows.append({"id": identifier, "video": relative_video, "label": label,
                     "speaker_id": speaker, "split": speaker_split[speaker], "input_kind": "raw_video",
                     "data_source": "real_camera", "video_sha256": sha256_file(video)})
        seen_speakers.add(speaker)
    if set(speaker_split) - seen_speakers:
        raise ValueError("An assigned speaker has no real recordings.")
    report = audit_records(rows)
    validate_alphabet(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write("".join(json.dumps(row) + "\n" for row in rows))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, default=ROOT / "alphabet_dataset")
    parser.add_argument("--train-speakers", required=True, help="Comma-separated speaker IDs")
    parser.add_argument("--valid-speakers", required=True)
    parser.add_argument("--test-speakers", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assignments = {split: [speaker.strip() for speaker in value.split(",") if speaker.strip()]
                   for split, value in (("train", args.train_speakers), ("valid", args.valid_speakers), ("test", args.test_speakers))}
    print(json.dumps(prepare(args.recordings, assignments, args.output), indent=2))


if __name__ == "__main__":
    main()
