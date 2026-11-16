"""Predict one explicitly bounded word with this experiment's own selected checkpoint."""
from __future__ import annotations
import json

import torch
from .common import load_config, load_dataset, parser, run_contract, sha256
from .model import WordVideoNet
from .preprocess import aligned_segment
from .train import validate_checkpoint


def predict(config, run, video, first, last):
    if type(first) is not int or type(last) is not int or first < 0 or not 2 <= last - first <= 500:
        raise ValueError("Supply actual word boundaries: 2–500 frames, exclusive end.")
    torch.set_num_threads(config["training"]["threads"])
    info, _ = load_dataset(run, config)
    path = run / "best.pt"
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    validate_checkpoint(artifact, config, info)
    model = WordVideoNet(len(config["words"]), **config["model"]).eval()
    model.load_state_dict(artifact["model_state_dict"], strict=True)
    clip, _, tracked = aligned_segment(video, first, last, config["preprocessing"])
    tensor = torch.from_numpy(clip).float().div(255).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        probabilities = model(tensor).softmax(1)[0]
    if not torch.isfinite(probabilities).all():
        raise ValueError("Nonfinite prediction.")
    values, indices = probabilities.topk(min(5, len(config["words"])))
    return {"vocabulary": config["words"], "input_shape": list(tensor.shape), "tracked_frames": tracked,
            "experimental": True,
            "checkpoint_sha256": sha256(path), "confidence_calibrated": False,
            "boundary_detection": "not performed; explicit word interval supplied",
            "ranked_words": [{"word": config["words"][i], "probability": p} for p, i in zip(values.tolist(), indices.tolist())]}


def main():
    p = parser(__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--start-frame", type=int, required=True)
    p.add_argument("--end-frame", type=int, required=True)
    args = p.parse_args()
    config = load_config(args.config)
    run = run_contract(config, args.run)
    print(json.dumps(predict(config, run, args.video, args.start_frame, args.end_frame), indent=2))


if __name__ == "__main__":
    main()
