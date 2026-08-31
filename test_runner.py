import subprocess
import sys
import os

# Set the working directory
os.chdir(r"C:\Users\ravik\OneDrive\Desktop\lip alphabet")

# Run pytest
result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"], capture_output=True, text=True, timeout=300)

print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print(f"\nReturn code: {result.returncode}")

sys.exit(result.returncode)