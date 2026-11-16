"""Additive common fields; legacy candidate fields remain diagnostic only."""
import hashlib
import json


def structured_prediction(result: dict, *, task: str) -> dict:
    rejected = bool(result.get('uncertain', False))
    candidate = result.get('letter') if task == 'alphabet_recognition' else result.get('predicted_word')
    entries = result.get('top3', []) if task == 'alphabet_recognition' else result.get('top_predictions', [])
    label_key = 'letter' if task == 'alphabet_recognition' else 'word'
    vocabulary = result.get('vocabulary') or ([chr(65+i) for i in range(26)] if task == 'alphabet_recognition' else [])
    vocab_id = hashlib.sha256(json.dumps(vocabulary, separators=(',', ':')).encode()).hexdigest()
    return {**result, 'schema_version': 1, 'status': 'uncertain' if rejected else 'success',
            'recognition_task': task, 'prediction': None if rejected else candidate,
            'top_k': [{'label': entry[label_key], 'score': entry['prob']} for entry in entries],
            'vocabulary_id': result.get('vocabulary_id', vocab_id),
            'model_version': result.get('model_version') or result.get('model_name') or result.get('model'),
            'confidence_calibrated': False}
