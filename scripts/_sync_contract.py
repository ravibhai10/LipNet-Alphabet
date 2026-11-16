import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import json
from pipeline.word_recognition.common import fingerprint, atomic_json, TASK, load_config

config_path = Path("pipeline/word_recognition/config.cpu.json")
config = load_config(config_path)

run_dir = Path("runs/word_recognition_v1")
contract_path = run_dir / "contract.json"

metadata = {
    "config": config,
    "config_sha256": fingerprint(config),
    "task": TASK
}
atomic_json(contract_path, metadata)
print("Updated contract.json with new config fingerprint:", metadata["config_sha256"])
