import subprocess, sys
from pathlib import Path

root = Path('.').resolve()
venvs = ['.venv', '.venv-avhubert', '.venv-avhubert-cpu']
for v in venvs:
    python = root / v / 'Scripts' / 'python.exe'
    if not python.exists():
        print(f'{v}: python not found')
        continue
    try:
        r = subprocess.run(
            [str(python), '-c', 'import torch, cv2, numpy, mediapipe; print(f"torch={torch.__version__}, cv2={cv2.__version__}, numpy={numpy.__version__}, mediapipe={mediapipe.__version__}")'],
            capture_output=True, text=True, timeout=15)
        if r.stdout.strip():
            print(f'{v}: {r.stdout.strip()}')
        else:
            print(f'{v}: MISSING packages: {r.stderr.strip()[:200]}')
    except Exception as e:
        print(f'{v}: ERROR: {e}')
