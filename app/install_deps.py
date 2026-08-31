import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn", "python-multipart"], check=True)