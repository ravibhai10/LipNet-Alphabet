import subprocess
import sys
import os
from pathlib import Path

# Resolve project root dynamically
project_root = Path(__file__).resolve().parent
os.chdir(project_root)

# Run pytest using virtual environment python if present
venv_python = project_root / ".venv" / "Scripts" / "python.exe"
python_bin = str(venv_python) if venv_python.exists() else sys.executable

result = subprocess.run(
    [python_bin, "-m", "pytest", "tests/", "-v", "--tb=short"],
    capture_output=True,
    text=True,
    timeout=300
)

print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print(f"\nReturn code: {result.returncode}")

sys.exit(result.returncode)
