import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Test 1: Models endpoint
res_models = client.get("/api/models")
assert res_models.status_code == 200, f"Expected 200, got {res_models.status_code}"
models_data = res_models.json()
print("GET /api/models:", models_data)

# Test 2: Word Health endpoint
res_word_health = client.get("/api/word/health")
assert res_word_health.status_code == 200, f"Expected 200, got {res_word_health.status_code}"
word_health_data = res_word_health.json()
print("GET /api/word/health:", word_health_data)
assert word_health_data.get("inference_ready") == True, "Word model should be ready"

# Test 3: Word Prediction endpoint with genuine synthetic frames
import cv2
import numpy as np
import base64

# Generate 24 dummy jpeg mouth frames
frames = []
for i in range(24):
    img = np.zeros((64, 64), dtype=np.uint8)
    cv2.circle(img, (32, 32), 15 + (i % 5), 255, -1)
    _, buffer = cv2.imencode('.jpg', img)
    encoded = base64.b64encode(buffer).decode('ascii')
    frames.append(encoded)

res_pred = client.post("/api/predict/word", json={"frames": frames})
assert res_pred.status_code == 200, f"Expected 200, got {res_pred.status_code}: {res_pred.text}"
pred_data = res_pred.json()
print("POST /api/predict/word SUCCESS:", pred_data)
assert "predicted_word" in pred_data
assert "confidence" in pred_data
assert "all_probabilities" in pred_data

print("\nALL FASTAPI WORD ENDPOINTS VERIFIED & FUNCTIONAL!")
