from pathlib import Path

path = Path("app/main.py")
content = path.read_text(encoding="utf-8")

target = 'raise HTTPException(status_code=503, detail="The selected classifier is unavailable. Check model readiness and the local backend log.") from exc'

addition = """

@app.get("/api/word/health")
def word_health():
    \"\"\"Return health, readiness, and supported vocabulary for WordVideoNet (Track A).\"\"\"
    return word_inference.health()


@app.post("/api/predict/word")
def predict_word(req: PredictRequest):
    \"\"\"Run the verified WordVideoNet (3D-CNN + BiGRU) on browser mouth crops.\"\"\"
    try:
        return word_inference.predict(req.frames)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except BlockingIOError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except RuntimeError as exc:
        logging.getLogger(__name__).warning("Word inference unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="The word recognition model is unavailable. Check model readiness.") from exc


@app.get("/api/models")
def list_available_models():
    \"\"\"List operational ML tracks and supported vocabularies.\"\"\"
    alpha_status = browser_inference.health()
    word_status = word_inference.health()
    return {
        "tracks": {
            "alphabet": {
                "name": "Alphabet Baseline (Track A0)",
                "task": "isolated_letter",
                "vocabulary_size": 26,
                "classes": [chr(65 + i) for i in range(26)],
                "ready": alpha_status.get("inference_ready", False),
                "endpoint": "/api/predict"
            },
            "word_recognition": {
                "name": "WordVideoNet (Track A)",
                "task": "isolated_word",
                "vocabulary_size": len(word_status.get("vocabulary", [])),
                "classes": word_status.get("vocabulary", []),
                "ready": word_status.get("inference_ready", False),
                "endpoint": "/api/predict/word",
                "checkpoint": word_status.get("checkpoint")
            },
            "video_to_text": {
                "name": "Visual CTC Sequence Decoder (Track B)",
                "task": "continuous_sentence",
                "ready": True,
                "modality": "video_only"
            }
        }
    }"""

if target in content and "/api/word/health" not in content:
    content = content.replace(target, target + "\n" + addition, 1)
    path.write_text(content, encoding="utf-8")
    print("PATCH_APPLIED_SUCCESSFULLY")
else:
    print("TARGET_NOT_FOUND_OR_ALREADY_PATCHED")
