"""Evidence-based track status. Never selects or loads a recognition model."""
from __future__ import annotations
import json
from pathlib import Path
from fastapi import APIRouter

ROOT=Path(__file__).resolve().parents[2]
router=APIRouter(prefix='/api/recognition',tags=['recognition evidence'])


@router.get('/capabilities')
def recognition_capabilities():
    inventory=ROOT/'data/processed/multidataset/miracl_v1.inventory.json'
    evaluation=ROOT/'runs/multidataset/miracl8_cpu_v2/evaluation.json'
    pretrained=ROOT/'runs/word_accuracy/avhubert_miracl8_v1/evaluation.json'
    if pretrained.is_file():evaluation=pretrained
    info=json.loads(inventory.read_text()) if inventory.is_file() else None
    result=json.loads(evaluation.read_text()) if evaluation.is_file() else None
    return {
        'schema_version':1,'modality':'video_only','audio_inference':False,
        'isolated_word':{
            'status':'offline_research_evaluated' if result else 'data_validation_or_training_pending',
            'live_ready':False,'production_ready':False,
            'model':'AV-HuBERT + supervised word head' if pretrained.is_file() else '3D-CNN + two-layer BiGRU; separate MIRACL experiment',
            'vocabulary':result['vocabulary'] if result else (info or {}).get('vocabularies',{}).get('miracl/isolated_word',[]),
            'evaluation':{**{k:result[k] for k in ('test_samples','top1_accuracy','top5_accuracy')},'checkpoint_sha256':result.get('head_sha256',result.get('checkpoint_sha256'))} if result else None,
            'reason':'Limited-vocabulary research evaluation. Webcam and unseen isolated-word rejection have not been validated. Check live model health for availability.',
            'out_of_vocabulary_detection':'unverified; closed-set confidence does not prove vocabulary membership'},
        'lrw500':{'status':'research_checkpoint_available' if (ROOT/'models/external/lrw_tcn/resnet18_dctcn_video.pth').is_file() else 'checkpoint_missing',
                  'vocabulary_size':500,'local_accuracy':None,'webcam_accuracy_validated':False,
                  'health_endpoint':'/api/word/lrw500/health','prediction_endpoint':'/api/predict/word/lrw500',
                  'reason':'Separate publisher-trained research model; the eight-word MIRACL accuracy does not apply.'},
        'video_to_text':{'status':'unavailable','live_ready':False,'vocabulary':None,
                         'reason':'No trained visual-only sentence decoder checkpoint verified. The pretrained AV-HuBERT encoder alone cannot transcribe.'},
        'miracl':{'manifest_available':info is not None,'samples':info['samples'] if info else 0,
                  'counts':info['counts'] if info else {},'speaker_assignments':info['speaker_assignments'] if info else {}},
        'existing_backend':'Unchanged; inspect /api/model/availability and /api/word/health for currently selected services.'}
