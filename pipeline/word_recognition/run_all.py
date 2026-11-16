"""Strict staged orchestration with logs, protected-file fingerprints, and resumption."""
from __future__ import annotations
from datetime import datetime
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import uuid

from .common import ROOT, atomic_json, fingerprint, load_config, load_json, run_contract, sha256, timestamp


def protected_snapshot():
    paths = [ROOT / "models/alphabet_classifier.pt", ROOT / "models/external/avhubert/base_vox_iter5.pt",
             *sorted((ROOT / "runs/finetune/lrw_500words").glob("*.pt"))]
    for folder in (ROOT / "frontend/src", ROOT / "frontend/public"):
        paths.extend(path for path in folder.rglob("*") if path.is_file())
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths if path.is_file()}


def execute(config_path, run, *, resume=False):
    config = load_config(config_path)
    run = run_contract(config, run)
    if (run / "complete.json").exists() and not resume:
        raise FileExistsError("Completed run is protected; choose a new run or explicit --resume.")
    config_path = Path(config_path).resolve()
    snapshot = run / "protected_before.json"
    if not snapshot.exists():
        atomic_json(snapshot, protected_snapshot())
    before = load_json(snapshot)
    environment = {name: os.environ.get(name) for name in ("LIPNET_MODEL_BACKEND", "LIPNET_AVHUBERT_HEAD")}
    commands, completed = [], False
    atomic_json(run / "resources.json", {"checked_at": timestamp(), "python": sys.version, "platform": platform.platform(),
                "logical_processors": os.cpu_count(), "free_gib": shutil.disk_usage(run).free / 2**30,
                "preprocessing": config["preprocessing"], "sample_quotas": config["quotas"], "vocabulary": config["words"],
                "expected_samples": sum(config["quotas"].values()) * len(config["words"]),
                "training_time_budget_minutes": config["training"]["max_minutes"]})
    try:
        for stage in ("download", "preprocess", "smoke", "train", "test", "diagnostics"):
            command = [sys.executable, "-u", "-m", f"pipeline.word_recognition.{stage}", "--config", str(config_path), "--run", str(run)]
            if stage == "train" and resume:
                command.append("--resume")
            entry = {"stage": stage, "command": command, "started_at": timestamp()}
            invocation = uuid.uuid4().hex[:8]
            log = run / f"{stage}-{invocation}.log"
            entry["log"] = str(log)
            commands.append(entry)
            atomic_json(run / "commands.json", commands)
            print(f"Running {stage}; log: {log}", flush=True)
            started = time.perf_counter()
            with log.open("x", encoding="utf-8") as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                        env={**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": str(config["training"]["threads"])})
            entry.update(returncode=result.returncode, seconds=time.perf_counter() - started)
            atomic_json(run / "commands.json", commands)
            if result.returncode:
                raise RuntimeError(f"Stage {stage} failed with exit {result.returncode}; inspect {log}. Subsequent stages were not run.")
        result = load_json(run / "evaluation_summary.json")
        atomic_json(run / "complete.json", {"completed_at": timestamp(), "task": config["dataset"],
                    "config_sha256": fingerprint(config), "evaluation": result, "production_accuracy_validated": False,
                    "recognition_quality": load_json(run / "diagnostics.json")["status"]})
        completed = True
        return result
    finally:
        after = protected_snapshot()
        changed = [name for name in set(before) | set(after) if before.get(name) != after.get(name)]
        current_environment = {name: os.environ.get(name) for name in environment}
        atomic_json(run / "integrity.json", {"checked_at": timestamp(), "completed": completed,
                    "protected_files_checked": len(before), "changed_protected_files": sorted(changed),
                    "backend_selection_unchanged": current_environment == environment,
                    "note": "Protected files are read only; external concurrent changes are reported, never reverted."})
        if changed or current_environment != environment:
            print("Protected workspace state changed during the run. Inspect integrity.json; no automatic restoration was attempted.", file=sys.stderr)


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "pipeline/word_recognition/config.cpu.json")
    p.add_argument("--run", type=Path)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.resume and args.run is None:
        p.error("--resume needs the exact existing --run directory.")
    run = args.run or ROOT / "runs/word_recognition" / (datetime.fromisoformat(timestamp()).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
    try:
        print(execute(args.config, run.resolve(), resume=args.resume))
    except (RuntimeError, ValueError, OSError) as error:
        p.exit(1, f"Word pipeline stopped: {error}\n")


if __name__ == "__main__":
    main()
