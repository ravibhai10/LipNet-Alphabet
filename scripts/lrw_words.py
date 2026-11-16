"""LRW whole-word CLI. Does not select or start any browser backend."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("ingest", help="Audit original WORD/{train,val,test}/ video/metadata pairs; resumable")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--agreement-confirmed", action="store_true")
    p = sub.add_parser("cache", help="Measured mouth alignment to compact uint8 caches; resumable")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, help="Process at most this many NEW clips for a bounded preflight")
    p.add_argument("--reserve-gib", type=float, default=20)
    for name in ("train", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", type=Path, required=True)
        p.add_argument("--cache", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--batch-size", type=int, default=2)
        p.add_argument("--threads", type=int, default=2)
        if name == "train":
            p.add_argument("--epochs", type=int, default=30)
            p.add_argument("--lr", type=float, default=1e-4)
            p.add_argument("--warmup-epochs", type=int, default=5)
            p.add_argument("--seed", type=int, default=42)
            initialization = p.add_mutually_exclusive_group()
            initialization.add_argument("--init-alphabet", type=Path, help="Optional existing 26-class CNN+LSTM; classification layers reset")
            initialization.add_argument("--init-word", type=Path, help="Audited word artifact with the SAME frozen splits/cache; optimizer/scheduler reset")
        else:
            p.add_argument("--checkpoint", type=Path, required=True)
    p = sub.add_parser("predict")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--threads", type=int, default=2)
    p = sub.add_parser("preflight", help="Shape-only CPU training timing; NOT recognition evaluation")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--output", type=Path)
    args = parser.parse_args()
    from pipeline.lrw import whole_word as words
    try:
        if args.command == "ingest":
            result = words.ingest(args.root, args.output, agreement_confirmed=args.agreement_confirmed)
        elif args.command == "cache":
            if args.limit is not None and args.limit < 1 or args.reserve_gib < 0:
                raise ValueError("Positive limit and nonnegative reserve required.")
            result = words.cache_clips(args.dataset, args.output, limit=args.limit, reserve_gib=args.reserve_gib)
        elif args.command == "train":
            result = words.train(args.dataset, args.cache, args.output, epochs=args.epochs,
                                 batch_size=args.batch_size, lr=args.lr, warmup_epochs=args.warmup_epochs,
                                 init=args.init_alphabet, init_word=args.init_word, threads=args.threads, seed=args.seed)
        elif args.command == "evaluate":
            result = words.evaluate(args.dataset, args.cache, args.checkpoint, args.output,
                                    batch_size=args.batch_size, threads=args.threads)
        elif args.command == "predict":
            result = words.predict(args.checkpoint, args.video, threads=args.threads)
        else:
            import torch
            from models.alphabet_model import AlphabetCNNBiLSTM
            if args.threads < 1 or args.batch_size < 1:
                raise ValueError("Positive threads/batch size required.")
            torch.set_num_threads(args.threads)
            model = AlphabetCNNBiLSTM(num_classes=500)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            inputs = torch.rand(args.batch_size, 29, 1, 140, 46)
            targets = torch.zeros(args.batch_size, dtype=torch.long)
            measurements = []
            for index in range(4):
                start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(inputs), targets)
                loss.backward()
                optimizer.step()
                if index:
                    measurements.append(time.perf_counter() - start)
            seconds_per_clip = sum(measurements) / len(measurements) / args.batch_size
            result = {"kind": "random-tensor training throughput ONLY; no recognition metrics",
                      "device": "cpu", "torch": torch.__version__, "threads": args.threads,
                      "batch_size": args.batch_size, "parameters": model.num_parameters(),
                      "input_shape": list(inputs.shape), "output_classes": 500,
                      "measured_seconds_per_training_clip": seconds_per_clip,
                      "rough_training_hours_per_epoch_400k_to_500k": [seconds_per_clip * n / 3600 for n in (400000, 500000)],
                      "estimate_excludes": "decoding, tracking, disk I/O, validation/test, thermal/load changes",
                      "float32_cache_gib_550k": 550000 * 29 * 140 * 46 * 4 / 2**30,
                      "uint8_cache_gib_550k_with_container_allowance": 550000 * (29 * 140 * 46 + 8192) / 2**30,
                      "free_gib": shutil.disk_usage(ROOT).free / 2**30}
            if args.output:
                if args.output.exists():
                    raise FileExistsError("Preflight output is protected.")
                words.atomic_json(args.output, result)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"LRW workflow stopped: {error}\n")


if __name__ == "__main__":
    main()
