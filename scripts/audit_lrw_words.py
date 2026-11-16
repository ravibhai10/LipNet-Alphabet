"""Read-only local LRW audit plus an optional publicly linked Oxford word example.

The public example is not the licensed corpus, a train/test split, or an accuracy
benchmark. It is kept under .qa and never ingested into training.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import sys
from urllib.parse import urljoin, urlsplit
from urllib.request import urlopen, Request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.lrw import whole_word as words

PAGE = "https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html"


def public_links(html):
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.current, self.label, self.links = None, "", {}

        def handle_starttag(self, tag, attrs):
            if tag == "a":
                self.current, self.label = dict(attrs).get("href"), ""

        def handle_data(self, value):
            if self.current:
                self.label += value

        def handle_endtag(self, tag):
            if tag == "a" and self.current:
                self.links[self.label.strip().lower()] = urljoin(PAGE, self.current)
                self.current = None
    parser = Links()
    parser.feed(html)
    result = {kind: parser.links[label] for kind, label in
              (("video", "example mp4 video"), ("metadata", "example metadata"), ("vocabulary", "here"))}
    for url in result.values():
        if urlsplit(url).hostname != "www.robots.ox.ac.uk" or urlsplit(url).scheme != "https":
            raise ValueError("Unexpected public example host/scheme.")
    return result


def fetch(url, limit=20 * 1024 * 1024):
    with urlopen(Request(url, headers={"User-Agent": "LipNet-LRW-public-example-audit/1.0"}), timeout=30) as response:
        body = response.read(limit + 1)
        if len(body) > limit:
            raise ValueError("Public example exceeds bounded audit download size.")
        return body


def main():
    import argparse
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-example", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Audit outputs are protected; choose a new filename.")
    torch.set_num_threads(2)
    paths = sorted((ROOT / "data/lrw").rglob("*.mp4"))
    hashes = Counter(words.digest(path) for path in paths)
    report = {"checked_utc": datetime.now(timezone.utc).isoformat(),
              "command": sys.executable + " " + " ".join(sys.argv),
              "vocabulary": {"nonempty": len(words.VOCABULARY), "unique": len(set(words.VOCABULARY)),
                             "file_sha256": words.digest(ROOT / "data/vocab.txt")},
              "local_lrw": {"videos": len(paths), "unique_source_sha256": len(hashes), "hash_counts": dict(hashes),
                            "split_counts": dict(Counter(path.relative_to(ROOT / "data/lrw").parts[0] for path in paths)),
                            "annotation_txt_count": len(list((ROOT / "data/lrw").rglob("*.txt"))),
                            "provenance": "generated development clips; see scripts/setup_lrw_dataset.py"},
              "recognition_accuracy_measured": False, "backend_environment_before": os.environ.get("LIPNET_MODEL_BACKEND")}
    checkpoints = [ROOT / "models/alphabet_classifier.pt", ROOT / "models/external/avhubert/base_vox_iter5.pt",
                   *sorted((ROOT / "runs/finetune/lrw_500words").glob("*.pt"))]
    before = {str(path.relative_to(ROOT)): words.digest(path) for path in checkpoints}
    if args.public_example:
        links = public_links(fetch(PAGE).decode())
        folder = ROOT / ".qa/lrw-public-example"
        folder.mkdir(parents=True, exist_ok=True)
        for kind, url in links.items():
            target = folder / {"video": "AFTERNOON.mp4", "metadata": "AFTERNOON.txt", "vocabulary": "official-vocab.txt"}[kind]
            body = fetch(url)
            if target.exists() and target.read_bytes() != body:
                raise FileExistsError("Public sample changed; existing copy is protected.")
            if not target.exists():
                target.write_bytes(body)
        public_words = [line.strip() for line in (folder / "official-vocab.txt").read_text().splitlines() if line.strip()]
        if public_words != words.VOCABULARY:
            raise ValueError("Local and publisher vocabulary/order differ.")
        sample = folder / "AFTERNOON.mp4"
        tensor = words.preprocess_video(sample).float().div(255).unsqueeze(0)
        smokes = []
        for path in checkpoints:
            if path.parent.name != "lrw_500words":
                continue
            state = torch.load(path, map_location="cpu", weights_only=True)
            model = words.AlphabetCNNBiLSTM(num_classes=500).eval()
            model.load_state_dict(state["model_state_dict"], strict=True)
            with torch.no_grad():
                logits = model(tensor)
            smokes.append({"checkpoint": str(path.relative_to(ROOT)), "input_shape": list(tensor.shape),
                           "output_shape": list(logits.shape), "finite_logits": bool(torch.isfinite(logits).all()),
                           "class_mapping_verified": False, "preprocessing_for_historical_training_verified": False})
        report["public_example"] = {"links": links, "source_sha256": words.digest(sample),
                                    "metadata": words.read_metadata(folder / "AFTERNOON.txt"),
                                    "decode": words.decode_check(sample), "vocabulary_order_matches_publisher": True,
                                    "tensor_shape": list(tensor.shape), "legacy_checkpoint_execution": smokes,
                                    "interpretation": "A genuine public single-word clip; execution/shape checks only. Legacy class mapping and preprocessing unverified; no recognition claim."}
    after = {str(path.relative_to(ROOT)): words.digest(path) for path in checkpoints}
    report.update(checkpoint_sha256=before, checkpoint_hashes_unchanged=before == after,
                  backend_environment_unchanged=report["backend_environment_before"] == os.environ.get("LIPNET_MODEL_BACKEND"))
    words.atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
