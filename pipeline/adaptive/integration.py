"""Candidate-bound inference integration through the deployed adapter contract.

Uses validation data only. This check establishes software behavior, never
recognition accuracy. Final evaluation remains a separate immutable operation.
"""
import io
import numpy as np
from .models import ModelRegistry
from .quality import digest


def candidate_response(store, model, record):
    raw = store.blob(record['id'], 'rois', model.get('owner'))
    if digest(raw) != record['roi_sha256']:
        raise ValueError('Candidate integration source changed.')
    rois = np.load(io.BytesIO(raw), allow_pickle=False)
    return ModelRegistry(store).predict(model['task'], rois, owner=model.get('owner'),
                                        selection='adaptive', candidate_id=model['id'])


def check_candidate(store, model, version):
    records = [r for r in version['records'] if r['split'] == 'valid']
    if not records:
        raise ValueError('Validation input required for deployed-adapter integration.')
    response = candidate_response(store, model, records[0])
    expected = model['checkpoint_sha256']
    passed = (response['model_id'] == model['id'] and response['model_version'] == expected and
              response['vocabulary'] == version['labels'] and
              np.isfinite(response['confidence']) and 0 <= response['confidence'] <= 1 and
              response['status'] in {'PREDICTED', 'UNCERTAIN'})
    return {'passed': bool(passed), 'model_id': model['id'], 'checkpoint_sha256': expected,
            'source_record': records[0]['id'], 'source_roi_sha256': records[0]['roi_sha256'],
            'scope': 'Real candidate adapter used by private prediction/probe routes; validation input only; not browser or accuracy evidence.'}
