"""Provision a dedicated Python 3.11 CPU environment without modifying existing ones."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[2]


def run(command):
    subprocess.run([str(value) for value in command], check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, default=ROOT / ".venv-word-recognition")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        parser.error("Use Python 3.11 for these tested CPU/MediaPipe dependency pins.")
    environment = args.environment.resolve()
    if environment in {(ROOT / ".venv").resolve(), (ROOT / ".venv-avhubert-cpu").resolve()}:
        parser.error("Existing application/encoder environments are protected; choose a dedicated environment.")
    if environment.exists() and not (environment / "pyvenv.cfg").exists():
        parser.error("Refusing to provision into a non-environment directory.")
    if not environment.exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    run([python, "-m", "pip", "install", "--disable-pip-version-check", "torch==2.2.2", "torchvision==0.17.2",
         "torchaudio==2.2.2", "--index-url", "https://download.pytorch.org/whl/cpu"])
    run([python, "-m", "pip", "install", "--disable-pip-version-check", "-r", Path(__file__).with_name("requirements.txt")])
    run([python, "-m", "pip", "check"])
    run([python, "-c", "import imageio_ffmpeg,subprocess; subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-version'],check=True)"])
    environment.joinpath("word_pipeline_setup.json").write_text(json.dumps({"python": str(python),
        "requirements": str(Path(__file__).with_name("requirements.txt")), "ffmpeg": "imageio-ffmpeg bundled binary; no system PATH changes"}, indent=2))
    print(f"Ready: {python}")


if __name__ == "__main__":
    main()
